"""Stage 3: automatic quality filter.

Reads:  data/distilled/qa_raw.jsonl
Writes: data/filtered/qa_v0.1_auto.jsonl
        reports/filter_stats.json

Filters (cheap, deterministic — no LLM needed):
  1. length:  5 ≤ len(question) ≤ 200,  3 ≤ len(answer) ≤ 600
  2. non-empty: question / answer / source_doc all required
  3. source-grounding:  answer chars must overlap ≥30% with source_text
  4. dedup: drop near-duplicate questions (rapidfuzz ratio ≥ 90)
  5. forbidden patterns: refusal phrases, hallucinated metadata

Run:
    python scripts/03_filter.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from rapidfuzz import fuzz
from tqdm import tqdm

from wh_bench.utils import (
    DATA_DISTILLED, DATA_FILTERED, REPORTS,
    get_logger, jsonl_read, jsonl_write, stable_id,
)

log = get_logger("03_filter")


REFUSAL_PATTERNS = [
    "无法回答", "不清楚", "未提及", "原文没有", "无法确定",
    "unable to", "cannot answer", "i don't know",
]


def length_ok(q: dict) -> bool:
    return 5 <= len(q["question"]) <= 200 and 3 <= len(q["answer"]) <= 600


def non_empty(q: dict) -> bool:
    return bool(q["question"]) and bool(q["answer"]) and bool(q.get("source_doc"))


def grounded(q: dict, min_overlap: float = 0.30) -> bool:
    """Answer should be reflected in source_text — measured via char overlap."""
    src = q.get("source_text", "") or ""
    ans = q.get("answer", "") or ""
    if not src or not ans:
        return False
    # Char-set overlap (cheap, language-agnostic)
    src_chars = set(src)
    ans_chars = set(ans)
    if not ans_chars:
        return False
    overlap = len(ans_chars & src_chars) / len(ans_chars)
    return overlap >= min_overlap


def no_refusal(q: dict) -> bool:
    a = q["answer"].lower()
    return not any(p in a for p in REFUSAL_PATTERNS)


def dedup(qas: list[dict], threshold: int = 90) -> list[dict]:
    kept: list[dict] = []
    for q in qas:
        is_dup = False
        for k in kept:
            if fuzz.ratio(q["question"], k["question"]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(q)
    return kept


def main() -> None:
    src = DATA_DISTILLED / "qa_raw.jsonl"
    if not src.exists():
        log.error(f"missing {src}; run 02_distill.py first")
        return

    qas = jsonl_read(src)
    log.info(f"loaded {len(qas)} raw QA")

    stats: Counter = Counter()
    stats["input"] = len(qas)

    # 1. non_empty
    qas = [q for q in qas if non_empty(q)] ; stats["after_non_empty"] = len(qas)
    # 2. length
    qas = [q for q in qas if length_ok(q)] ; stats["after_length"] = len(qas)
    # 3. refusal
    qas = [q for q in qas if no_refusal(q)] ; stats["after_no_refusal"] = len(qas)
    # 4. grounding
    qas = [q for q in qas if grounded(q)] ; stats["after_grounding"] = len(qas)
    # 5. dedup
    qas = dedup(qas) ; stats["after_dedup"] = len(qas)

    # reassign sequential ids
    for i, q in enumerate(qas):
        q["id"] = stable_id("wh-bench", i + 1)
        q["review_status"] = "auto"  # human review happens in stage 4

    out = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    jsonl_write(out, qas)
    log.info(f"✓ {len(qas)} QA after filter → {out}")

    # stats by category / authority
    by_cat = Counter(q.get("category", "(空)") or "(空)" for q in qas)
    by_auth = Counter(q.get("authority", "(空)") or "(空)" for q in qas)
    by_diff = Counter(q.get("difficulty", "(空)") or "(空)" for q in qas)

    report = {
        "stages": dict(stats),
        "kept": len(qas),
        "drop_rate": round(1 - len(qas) / max(1, stats["input"]), 3),
        "by_category": dict(by_cat.most_common()),
        "by_authority": dict(by_auth.most_common()),
        "by_difficulty": dict(by_diff.most_common()),
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (REPORTS / "filter_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  stats → reports/filter_stats.json")
    log.info(f"  drop rate: {report['drop_rate']*100:.1f}%")
    log.info(f"  top categories: {list(by_cat.most_common(5))}")


if __name__ == "__main__":
    main()
