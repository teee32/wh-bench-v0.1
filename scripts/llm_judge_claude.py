#!/usr/bin/env python3
"""
LLM-as-Judge (Claude jury): 用 claude-sonnet-4-6 复审 wh-bench-v0.1 全 303 条。

跟 llm_judge.py 对照使用 — 同 prompt、同四维、同 verdict 标签。
两个 judge 都 fail 才剔除（jury 模式）。
"""
import os, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 加载 .env
for line in Path("/root/.hermes/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

CLIENT = OpenAI(
    api_key=os.environ["FOXNIO_OPUS_API_KEY"],
    base_url="https://sub.foxnio.com/v1",
    default_headers={"User-Agent": "curl/8.4.0"},
)
MODEL = "claude-sonnet-4-6"

JUDGE_PROMPT = """你是中文法律法规问答评测集的质量审查员。给你一条 QA 样本及其原文片段，按以下四维严格打分。

【原文片段】
法规：{source_doc}
章节：{source_section}
直接引用：{source_quote}
原文上下文：
---
{source_text}
---

【待审 QA】
问题：{question}
答案：{answer}

【四维评估】（每维 1-5 分，5=完美，1=不可用）
1. groundedness   答案是否能从原文直接得出（不能编造、不能扩展原文没说的内容）
2. accuracy       答案是否准确（数值正确、主体明确、列表完整、无遗漏关键项）
3. clarity        问题表述是否清晰、有客观答案（不模糊、不歧义）
4. specificity    问题是否具体到能在原文定位（不是"是什么/有哪些"这种泛问）

【总评】
- pass        四维都≥4 且无重大问题
- borderline  有 1 维=3 或答案部分不准但主体对
- fail        任一维≤2 或答案与原文矛盾/编造/严重漏项

只输出严格 JSON，不要解释、不要 markdown 代码块：
{{"groundedness":int,"accuracy":int,"clarity":int,"specificity":int,"verdict":"pass|borderline|fail","issues":["简短issue1","issue2"],"reason":"一句话总结"}}"""


def judge_one(sample, max_retries=3):
    src = sample["source_text"][:2500]
    prompt = JUDGE_PROMPT.format(
        source_doc=sample["source_doc"],
        source_section=sample["source_section"],
        source_quote=sample["source_quote"],
        source_text=src,
        question=sample["question"],
        answer=sample["answer"],
    )
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            # 防御性剥离 markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            # Claude 偶尔会在 JSON 前后加注释，取首个 { 到末尾的 }
            if not raw.startswith("{"):
                start = raw.find("{")
                if start >= 0:
                    raw = raw[start:]
            if not raw.endswith("}"):
                end = raw.rfind("}")
                if end >= 0:
                    raw = raw[:end+1]
            data = json.loads(raw)
            for k in ("groundedness", "accuracy", "clarity", "specificity", "verdict"):
                if k not in data:
                    raise ValueError(f"missing field: {k}")
            data["id"] = sample["id"]
            data["question"] = sample["question"]
            data["answer"] = sample["answer"]
            data["source_doc"] = sample["source_doc"]
            data["source_quote"] = sample["source_quote"]
            data["difficulty"] = sample["difficulty"]
            data["_usage"] = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                return {"id": sample["id"], "verdict": "ERROR", "error": str(e)[:200]}
            time.sleep(2 ** attempt)


def main():
    src = Path("/root/projects/wh-bench-v0.1/data/filtered/qa_v0.1_auto.jsonl")
    out_dir = Path("/root/projects/wh-bench-v0.1/data/judged")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_judged_claude.jsonl"

    samples = [json.loads(l) for l in src.read_text().splitlines()]
    print(f"[judge-claude] 共 {len(samples)} 条，并发 6，model={MODEL}", flush=True)
    t0 = time.time()

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=6) as ex, out_path.open("w") as f:
        futures = {ex.submit(judge_one, s): s for s in samples}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            if completed % 20 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(samples) - completed) / rate
                print(f"[judge-claude] {completed}/{len(samples)}  rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[judge-claude] 完成 {len(results)} 条，耗时 {elapsed:.1f}s", flush=True)

    from collections import Counter
    verdict_count = Counter(r.get("verdict", "ERROR") for r in results)
    print("\n=== Verdict 分布 ===")
    for v, n in verdict_count.most_common():
        print(f"  {v}: {n} ({n/len(results)*100:.1f}%)")

    valid = [r for r in results if r.get("verdict") in ("pass", "borderline", "fail")]
    if valid:
        print("\n=== 平均分 ===")
        for k in ("groundedness", "accuracy", "clarity", "specificity"):
            avg = sum(r.get(k, 0) for r in valid) / len(valid)
            print(f"  {k}: {avg:.2f}")

    in_tok = sum(r.get("_usage", {}).get("in", 0) for r in results if "_usage" in r)
    out_tok = sum(r.get("_usage", {}).get("out", 0) for r in results if "_usage" in r)
    print(f"\n=== Tokens === in={in_tok:,}  out={out_tok:,}  total={in_tok+out_tok:,}")

    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
