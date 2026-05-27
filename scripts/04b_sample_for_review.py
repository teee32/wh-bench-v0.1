"""Stage 4b: sample N QA items and export to a Telegram-readable markdown file.

Use this when you can't open the Flask web UI (e.g. remote SSH, Telegram bot).
Reviewer reads the .md, then sends back a list of REJECT ids; the merge step
turns those into `data/filtered/qa_v0.1_reviewed.jsonl`.

Run:
    python scripts/04b_sample_for_review.py --n 30
    python scripts/04b_sample_for_review.py --n 30 --seed 42

Then, after the reviewer responds (e.g. "reject 3, 7, 12"):
    python scripts/04b_sample_for_review.py --merge --reject 3,7,12
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from wh_bench.utils import (
    DATA_FILTERED, REPORTS, get_logger, jsonl_read, jsonl_write,
)

log = get_logger("04b_sample")
SAMPLE_JSONL = REPORTS / "review_sample.jsonl"
SAMPLE_MD = REPORTS / "review_sample.md"


def make_sample(n: int, seed: int) -> None:
    src = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    if not src.exists():
        log.error(f"找不到 {src}，请先跑 03_filter.py")
        return
    pool = jsonl_read(src)
    log.info(f"pool size: {len(pool)}")
    if n >= len(pool):
        sample = pool
    else:
        rng = random.Random(seed)
        sample = rng.sample(pool, n)
    # 加上 review_idx（1..N）便于用户引用
    for i, q in enumerate(sample, 1):
        q["review_idx"] = i
    jsonl_write(SAMPLE_JSONL, sample)
    log.info(f"✓ 抽样 {len(sample)} 条 → {SAMPLE_JSONL}")

    # 渲染 markdown
    lines = [
        f"# wh-bench v0.1 抽检 — {n} 条样本",
        "",
        f"*抽样时间：{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*  ",
        f"*池大小：{len(pool)} 条，抽样数：{len(sample)}*",
        "",
        "**抽检说明**：每条看 Q + A + 原文出处。**只标需要 REJECT 的编号**，其余默认 ACCEPT。",
        "回复格式：`reject 3, 7, 12` 或 `全部通过`。",
        "",
        "标准：",
        "1. 答案是否能从原文直接引用或紧密改写？（不能瞎编/外推）",
        "2. 问题是否表述清晰、有客观答案？",
        "3. 答案是否完整准确？（数值、主体、步骤）",
        "",
        "---",
        "",
    ]
    for i, q in enumerate(sample, 1):
        lines.extend([
            f"## #{i}  `{q.get('category','?')}` / `{q.get('difficulty','?')}`",
            "",
            f"**Q**: {q['question']}",
            "",
            f"**A**: {q['answer']}",
            "",
            f"**来源**：《{q.get('source_doc','?')}》— {q.get('source_section','?')}  ({q.get('authority','?')})",
            "",
            f"<details><summary>原文支撑</summary>",
            "",
            f"> {q.get('source_quote', q.get('source_text',''))[:400]}",
            "",
            "</details>",
            "",
            "---",
            "",
        ])
    SAMPLE_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"✓ markdown → {SAMPLE_MD}")
    log.info(f"  字符数: {len(SAMPLE_MD.read_text(encoding='utf-8'))}")


def merge_reviews(reject_ids: set[int]) -> None:
    """合并 reviewer 的判定 → qa_v0.1_reviewed.jsonl"""
    if not SAMPLE_JSONL.exists():
        log.error(f"找不到 {SAMPLE_JSONL}，请先跑 sample 步骤")
        return
    sample = jsonl_read(SAMPLE_JSONL)
    accepted = []
    rejected = []
    for q in sample:
        idx = q.get("review_idx", -1)
        if idx in reject_ids:
            q["review_status"] = "human_rejected"
            q["review_note"] = "rejected via Telegram review"
            rejected.append(q)
        else:
            q["review_status"] = "human_verified"
            q["review_note"] = ""
            accepted.append(q)

    # 已抽检（accept）的样本 + 未抽检的剩余条目 = 最终 reviewed.jsonl
    sampled_ids = {q["id"] for q in sample}
    src = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    full = jsonl_read(src)
    untouched = [q for q in full if q["id"] not in sampled_ids]
    for q in untouched:
        q["review_status"] = "auto_filtered"
        q["review_note"] = ""

    out = accepted + untouched   # 拒绝的不进最终集
    dest = DATA_FILTERED / "qa_v0.1_reviewed.jsonl"
    jsonl_write(dest, out)

    # 写状态
    state = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sample_n": len(sample),
        "accepted_n": len(accepted),
        "rejected_n": len(rejected),
        "acceptance_rate": round(len(accepted) / max(1, len(sample)), 3),
        "rejected_ids": sorted(reject_ids),
        "final_size": len(out),
    }
    (REPORTS / "review_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(f"✓ 抽检合并完成：")
    log.info(f"  抽样 {len(sample)}：accept {len(accepted)}  reject {len(rejected)}")
    log.info(f"  通过率 {100*state['acceptance_rate']:.1f}%")
    log.info(f"  → {dest}  (含未抽样 {len(untouched)} 条，标记 auto_filtered)")
    log.info(f"  → reports/review_state.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge", action="store_true",
                   help="合并 reviewer 的反馈到 reviewed.jsonl")
    p.add_argument("--reject", default="",
                   help="逗号分隔的拒绝 idx，如 '3,7,12'")
    args = p.parse_args()

    if args.merge:
        reject_ids = {int(x.strip()) for x in args.reject.split(",") if x.strip()}
        merge_reviews(reject_ids)
    else:
        make_sample(args.n, args.seed)


if __name__ == "__main__":
    main()
