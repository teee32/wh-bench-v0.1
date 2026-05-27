"""Stage 5: baseline evaluation — run a target LLM on the benchmark.

Reads:  data/filtered/qa_v0.1_reviewed.jsonl  (or --input)
Writes: reports/baseline_<model>_<ts>.json
        reports/baseline_<model>_<ts>.jsonl   (per-question detail)

Two scoring modes:
  - char_f1     :  character-level F1 between gold answer and prediction (default)
  - llm_judge   :  use a stronger LLM as judge (--judge); slower but more accurate

Run:
    python scripts/05_baseline.py                            # uses default model
    python scripts/05_baseline.py --model gpt-4o-mini
    python scripts/05_baseline.py --judge --judge-model gpt-4o
    python scripts/05_baseline.py --input data/filtered/qa_v0.1_auto.jsonl --max 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from wh_bench.utils import (
    DATA_FILTERED, REPORTS, get_logger, jsonl_read, jsonl_write,
    get_llm_client,
)

log = get_logger("05_baseline")


PROMPT = """你是一名工业安全规程领域的专家。请基于你的专业知识简洁、准确地回答下列问题。

问题：{question}

要求：
- 直接给出答案，不要解释推理过程
- 控制在 200 字以内
- 如果题目涉及具体数值/步骤/责任主体，请精确指出"""


JUDGE_PROMPT = """你是评测官。判断"模型回答"是否与"参考答案"在事实/语义上一致。

问题：{question}
参考答案：{gold}
模型回答：{pred}

判定规则：
- 一致（核心事实匹配，措辞可不同）→ 1
- 部分一致（说出了一部分但漏掉关键信息或加了错误内容）→ 0.5
- 不一致（事实错误或答非所问）→ 0

只输出 JSON：{{"score": 1|0.5|0, "reason": "<=20字理由"}}"""


# ── scoring ─────────────────────────────────────────────────────────────
def char_f1(gold: str, pred: str) -> float:
    """Char-level F1 (set-based, fast). Decent proxy for Chinese QA."""
    if not gold or not pred:
        return 0.0
    g = Counter(gold); p = Counter(pred)
    common = sum((g & p).values())
    if common == 0:
        return 0.0
    prec = common / sum(p.values())
    rec = common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def call_model(client, model: str, prompt: str, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def llm_judge(client, model: str, question: str, gold: str, pred: str) -> dict:
    raw = call_model(client, model,
                     JUDGE_PROMPT.format(question=question, gold=gold, pred=pred),
                     max_tokens=120)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"score": 0, "reason": f"parse_fail: {raw[:30]}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DATA_FILTERED / "qa_v0.1_reviewed.jsonl"))
    p.add_argument("--model", default="",
                   help="override model name (default: from env)")
    p.add_argument("--max", type=int, default=0,
                   help="0 = all; else limit for smoke test")
    p.add_argument("--judge", action="store_true",
                   help="use LLM-judge in addition to char_f1")
    p.add_argument("--judge-model", default="",
                   help="judge model (default: same as eval)")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        # fall back to auto-filtered if reviewed not present
        alt = DATA_FILTERED / "qa_v0.1_auto.jsonl"
        if alt.exists():
            log.warning(f"{src} missing; falling back to {alt}")
            src = alt
        else:
            log.error(f"no input found: {args.input}")
            return

    qas = jsonl_read(src)
    if args.max:
        qas = qas[: args.max]
    log.info(f"evaluating {len(qas)} QA from {src.name}")

    client, default_model = get_llm_client()
    model = args.model or default_model
    log.info(f"target model: {model}")
    if args.judge:
        judge_model = args.judge_model or default_model
        log.info(f"judge model:  {judge_model}")

    rows: list[dict] = []
    t0 = time.time()
    for q in tqdm(qas, desc="evaluating", unit="q"):
        try:
            pred = call_model(client, model, PROMPT.format(question=q["question"]))
        except Exception as e:
            log.warning(f"  model fail {q['id']}: {e}")
            pred = ""
        f1 = char_f1(q["answer"], pred)
        row = {
            "id": q["id"],
            "question": q["question"],
            "gold": q["answer"],
            "pred": pred,
            "char_f1": round(f1, 3),
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
        }
        if args.judge:
            try:
                v = llm_judge(client, judge_model, q["question"], q["answer"], pred)
                row["judge_score"] = float(v.get("score", 0))
                row["judge_reason"] = v.get("reason", "")
            except Exception as e:
                row["judge_score"] = 0.0
                row["judge_reason"] = f"err: {e}"
        rows.append(row)

    elapsed = time.time() - t0

    # ── aggregate ────────────────────────────────────────────────────────
    n = len(rows)
    avg_f1 = sum(r["char_f1"] for r in rows) / max(1, n)
    summary = {
        "model": model,
        "n_questions": n,
        "char_f1_mean": round(avg_f1, 3),
        "char_f1_above_0.5": sum(1 for r in rows if r["char_f1"] >= 0.5),
        "by_difficulty_mean_f1": {
            diff: round(
                sum(r["char_f1"] for r in rows if r["difficulty"] == diff)
                / max(1, sum(1 for r in rows if r["difficulty"] == diff)),
                3)
            for diff in {r["difficulty"] for r in rows} if diff
        },
        "elapsed_sec": round(elapsed, 1),
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if args.judge:
        summary["judge_model"] = judge_model
        summary["judge_score_mean"] = round(
            sum(r.get("judge_score", 0) for r in rows) / max(1, n), 3)
        summary["judge_score_above_0.5"] = sum(
            1 for r in rows if r.get("judge_score", 0) >= 0.5)

    safe_model = model.replace("/", "_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS / f"baseline_{safe_model}_{ts}.json"
    detail_path = REPORTS / f"baseline_{safe_model}_{ts}.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    jsonl_write(detail_path, rows)

    log.info(f"✓ summary → {summary_path}")
    log.info(f"  detail  → {detail_path}")
    log.info(f"  char_f1 mean = {avg_f1:.3f}  ({elapsed:.1f}s for {n} q)")
    if args.judge:
        log.info(f"  judge mean   = {summary['judge_score_mean']:.3f}")


if __name__ == "__main__":
    main()
