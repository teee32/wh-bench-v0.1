#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whbench-eval — wh-bench 中文法律法规大模型自动评测工具

一个面向开发者的「模型工具」：接入任意 OpenAI 兼容的大模型 API，
自动在 wh-bench v0.1（300 条中文法律法规 QA）上跑完整评测，
自动判分，产出结构化评测报告（JSON + Markdown）。

支撑大模型「垂直行业（法律）能力评估」的全流程：
  加载题集 → 调用被测模型作答 → 自动判分 → 汇总报告

用法
----
    # 1) 配置被测模型（OpenAI 兼容端点）
    export EVAL_API_KEY=sk-xxx
    export EVAL_BASE_URL=https://api.openai.com/v1
    export EVAL_MODEL=gpt-4o-mini

    # 2) 跑评测
    python tools/whbench_eval.py --model gpt-4o-mini --max 50
    python tools/whbench_eval.py --model deepseek-chat --judge llm   # 用LLM裁判判分
    python tools/whbench_eval.py --diff hard --scoring char_f1       # 只测困难题

    # 3) 产出
    reports/eval_<model>_<时间戳>.json    # 机器可读：每题明细 + 汇总指标
    reports/eval_<model>_<时间戳>.md      # 人类可读：评测报告

判分模式
--------
  char_f1 (默认)：字符级 F1，快、零额外成本、可复现
  llm     ：用一个更强的「裁判模型」判定语义一致性（1 / 0.5 / 0），更准、需额外API

退出码：0=成功；2=配置错误；3=数据缺失
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "wh_bench_v0.1_all.jsonl"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ── env loader (~/.hermes/.env 兜底，也支持项目 .env) ──────────────────
def _load_env():
    for envpath in (ROOT / ".env", Path.home() / ".hermes" / ".env"):
        if envpath.exists():
            for line in envpath.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── 被测模型 / 裁判模型 客户端 ────────────────────────────────────────
def build_client(api_key_env: str, base_url_env: str, default_base: str):
    from openai import OpenAI
    key = os.getenv(api_key_env)
    if not key:
        return None
    return OpenAI(
        api_key=key,
        base_url=os.getenv(base_url_env, default_base),
        # 部分 OpenAI 兼容网关的 WAF 会拦默认 UA
        default_headers={"User-Agent": "curl/8.4.0"},
    )


# ── 提示词 ────────────────────────────────────────────────────────────
ANSWER_PROMPT = """你是一名中国法律法规领域的专家。请基于你的专业知识简洁、准确地回答下列问题。

问题：{q}

要求：
- 直接给出答案，不要解释推理过程
- 控制在 200 字以内
- 涉及具体数值/期限/责任主体/列表项时，请精确、完整地指出"""

JUDGE_PROMPT = """你是评测官。判断「模型回答」是否与「参考答案」在事实与语义上一致。

问题：{q}
参考答案：{gold}
模型回答：{pred}

判定规则：
- 一致（核心事实匹配，措辞可不同）→ 1
- 部分一致（说出一部分但漏掉关键信息，或附带错误内容）→ 0.5
- 不一致（事实错误或答非所问）→ 0

只输出 JSON：{{"score": 1, "reason": "<=20字理由"}}"""


# ── 判分：字符级 F1 ──────────────────────────────────────────────────
def char_f1(gold: str, pred: str) -> float:
    if not gold or not pred:
        return 0.0
    g, p = Counter(gold), Counter(pred)
    common = sum((g & p).values())
    if common == 0:
        return 0.0
    prec = common / sum(p.values())
    rec = common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


# ── 调模型（带重试） ──────────────────────────────────────────────────
def call_llm(client, model: str, prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"模型调用失败（重试3次）: {last}")


def judge_one(judge_client, judge_model, q, gold, pred) -> tuple[float, str]:
    raw = call_llm(judge_client, judge_model,
                   JUDGE_PROMPT.format(q=q, gold=gold, pred=pred), max_tokens=120)
    # 容错解析：抓第一个 {...}
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            s = float(d.get("score", 0))
            return (s if s in (0, 0.5, 1) else 0.0, str(d.get("reason", ""))[:40])
        except Exception:  # noqa
            pass
    return (0.0, "裁判输出无法解析")


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="whbench-eval — wh-bench 中文法律法规大模型自动评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境变量：EVAL_API_KEY / EVAL_BASE_URL / EVAL_MODEL（被测模型）；"
               "JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL（裁判模型，--judge llm 时用）",
    )
    ap.add_argument("--model", help="被测模型名（覆盖 EVAL_MODEL）")
    ap.add_argument("--scoring", choices=["char_f1", "llm"], default="char_f1",
                    help="判分模式：char_f1（默认，快/免费）| llm（用裁判模型判语义）")
    ap.add_argument("--judge", choices=["char_f1", "llm"], dest="scoring_alias",
                    help="--scoring 的别名")
    ap.add_argument("--judge-model", help="裁判模型名（覆盖 JUDGE_MODEL）")
    ap.add_argument("--diff", choices=["all", "easy", "medium", "hard"], default="all",
                    help="只评测指定难度")
    ap.add_argument("--max", type=int, default=0, help="最多评测多少题（0=全部）")
    ap.add_argument("--workers", type=int, default=4, help="并发数")
    ap.add_argument("--data", default=str(DATA), help="题集 jsonl 路径")
    ap.add_argument("--dry-run", action="store_true", help="只加载数据、不调模型（自检用）")
    args = ap.parse_args()
    if args.scoring_alias:
        args.scoring = args.scoring_alias

    _load_env()

    # 1) 加载题集
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[错误] 题集不存在: {data_path}", file=sys.stderr)
        sys.exit(3)
    items = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    if args.diff != "all":
        items = [x for x in items if x.get("difficulty") == args.diff]
    if args.max > 0:
        items = items[: args.max]
    if not items:
        print("[错误] 没有可评测的题目（检查 --diff / --max）", file=sys.stderr)
        sys.exit(3)
    print(f"[1/4] 已加载题集: {len(items)} 题  (难度={args.diff})")

    if args.dry_run:
        print(f"[dry-run] 数据自检通过。示例题: {items[0]['question'][:40]}…")
        # 自检判分器
        f1 = char_f1(items[0]["answer"], items[0]["answer"])
        print(f"[dry-run] char_f1(完全相同)= {f1:.3f}  (应=1.000)")
        sys.exit(0)

    # 2) 被测模型客户端
    eval_client = build_client("EVAL_API_KEY", "EVAL_BASE_URL", "https://api.openai.com/v1")
    if eval_client is None:
        # 兜底：复用 FOXNIO（本环境可用）
        eval_client = build_client("FOXNIO_API_KEY", "FOXNIO_BASE_URL", "https://api.foxnio.com/v1")
        if eval_client is None:
            print("[错误] 未配置被测模型 API key。请设置 EVAL_API_KEY（或 FOXNIO_API_KEY）。",
                  file=sys.stderr)
            sys.exit(2)
    model = args.model or os.getenv("EVAL_MODEL") or os.getenv("FOXNIO_MODEL", "gpt-5.5")
    print(f"[2/4] 被测模型: {model}  | 判分模式: {args.scoring}")

    judge_client, judge_model = None, None
    if args.scoring == "llm":
        judge_client = build_client("JUDGE_API_KEY", "JUDGE_BASE_URL", "https://api.openai.com/v1") \
            or build_client("FOXNIO_API_KEY", "FOXNIO_BASE_URL", "https://api.foxnio.com/v1")
        if judge_client is None:
            print("[错误] --judge llm 需配置裁判模型 JUDGE_API_KEY（或 FOXNIO_API_KEY）。", file=sys.stderr)
            sys.exit(2)
        judge_model = args.judge_model or os.getenv("JUDGE_MODEL") or os.getenv("FOXNIO_MODEL", "gpt-5.5")
        print(f"      裁判模型: {judge_model}")

    # 3) 跑评测（并发）
    print(f"[3/4] 开始评测（并发 {args.workers}）…")
    results = [None] * len(items)
    t0 = time.time()

    def work(i, it):
        q, gold = it["question"], it["answer"]
        pred = call_llm(eval_client, model, ANSWER_PROMPT.format(q=q))
        if args.scoring == "char_f1":
            score = round(char_f1(gold, pred), 4)
            reason = "char-F1"
        else:
            score, reason = judge_one(judge_client, judge_model, q, gold, pred)
        return i, {
            "id": it.get("id"), "question": q, "gold": gold, "pred": pred,
            "score": score, "reason": reason,
            "difficulty": it.get("difficulty"), "category": it.get("category"),
            "source_doc": it.get("source_doc"),
        }

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, it) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            try:
                i, rec = fut.result()
                results[i] = rec
            except Exception as e:  # noqa
                done += 1
                print(f"    [警告] 第{done}题评测失败: {str(e)[:80]}", file=sys.stderr)
                continue
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"    进度 {done}/{len(items)}  用时 {time.time()-t0:.0f}s")

    results = [r for r in results if r is not None]
    if not results:
        print("[错误] 全部题目评测失败，请检查模型 API 配置/网络。", file=sys.stderr)
        sys.exit(2)

    # 4) 汇总 + 报告
    n = len(results)
    total = sum(r["score"] for r in results)
    overall = total / n
    by_diff = defaultdict(lambda: [0.0, 0])
    by_cat = defaultdict(lambda: [0.0, 0])
    for r in results:
        by_diff[r["difficulty"]][0] += r["score"]; by_diff[r["difficulty"]][1] += 1
        by_cat[r["category"]][0] += r["score"]; by_cat[r["category"]][1] += 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w.-]", "_", model)
    report = {
        "tool": "whbench-eval",
        "benchmark": "wh-bench v0.1",
        "model": model,
        "scoring": args.scoring,
        "judge_model": judge_model,
        "timestamp": ts,
        "num_questions": n,
        "difficulty_filter": args.diff,
        "overall_score": round(overall, 4),
        "by_difficulty": {k: round(v[0]/v[1], 4) for k, v in sorted(by_diff.items())},
        "by_category": {k: round(v[0]/v[1], 4) for k, v in sorted(by_cat.items(), key=lambda x:-x[1][1])},
        "elapsed_sec": round(time.time()-t0, 1),
        "details": results,
    }
    json_path = REPORTS / f"eval_{safe_model}_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 报告
    md = []
    md.append(f"# wh-bench 评测报告 · {model}\n")
    md.append(f"- 评测工具：`whbench-eval`")
    md.append(f"- 评测集：wh-bench v0.1（{n} 题，难度={args.diff}）")
    md.append(f"- 判分模式：`{args.scoring}`" + (f"（裁判模型 `{judge_model}`）" if judge_model else ""))
    md.append(f"- 时间：{ts}  | 用时：{report['elapsed_sec']}s\n")
    md.append(f"## 总分：**{overall*100:.1f}%**（{total:.1f} / {n}）\n")
    md.append("### 分难度")
    md.append("| 难度 | 得分率 | 题数 |")
    md.append("| --- | --- | --- |")
    for k, v in sorted(by_diff.items()):
        md.append(f"| {k} | {v[0]/v[1]*100:.1f}% | {v[1]} |")
    md.append("\n### 分领域（题数前10）")
    md.append("| 领域 | 得分率 | 题数 |")
    md.append("| --- | --- | --- |")
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1][1])[:10]:
        md.append(f"| {k} | {v[0]/v[1]*100:.1f}% | {v[1]} |")
    md.append("\n### 失分样本（score < 1，前10）")
    bad = [r for r in results if r["score"] < 1][:10]
    for r in bad:
        md.append(f"- **[{r['score']}] {r['question'][:40]}**")
        md.append(f"  - 参考：{r['gold'][:60]}")
        md.append(f"  - 模型：{r['pred'][:60]}")
    md_path = REPORTS / f"eval_{safe_model}_{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[4/4] 完成。总分 {overall*100:.1f}%（{n}题）")
    print(f"      JSON 报告: {json_path}")
    print(f"      MD   报告: {md_path}")


if __name__ == "__main__":
    main()
