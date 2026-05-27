#!/usr/bin/env python3
"""
wh-bench v0.1 finalization (口径 A: jury 严交集)
- 加载 303 条过滤通过的 QA
- 剔除 jury_drop_list 中双 fail 交集 3 条 → 300 条
- 合并 jury 元数据（gpt/claude 评分 + 决定 + CONTESTED 标记）
- 80/20 train/test split，按 difficulty 分层
- 输出 JSONL + Parquet（HuggingFace）+ CSV（人类可读）
"""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
FILTERED = ROOT / "data/filtered/qa_v0.1_auto.jsonl"
JUDGED_GPT = ROOT / "data/judged/qa_judged.jsonl"
JUDGED_CLAUDE = ROOT / "data/judged/qa_judged_claude.jsonl"
JURY = ROOT / "data/judged/jury_verdicts.jsonl"
DROP_LIST = ROOT / "data/judged/jury_drop_list.txt"
OUT = ROOT / "data/processed"
OUT.mkdir(exist_ok=True, parents=True)

random.seed(42)

# 1) 读取剔除清单
drop_ids = set(DROP_LIST.read_text(encoding="utf-8").strip().splitlines())
print(f"[drop] {len(drop_ids)} ids: {sorted(drop_ids)}")

# 2) 读取过滤通过的 303 条
qa_records = []
with FILTERED.open(encoding="utf-8") as f:
    for line in f:
        qa_records.append(json.loads(line))
print(f"[filtered] {len(qa_records)} records")

# 3) 读取 jury verdicts（含双 judge 评分）
jury_map = {}
with JURY.open(encoding="utf-8") as f:
    for line in f:
        v = json.loads(line)
        jury_map[v["id"]] = v
print(f"[jury] {len(jury_map)} verdicts")

# 4) 合并 + 剔除
final = []
contested_ids = []
for r in qa_records:
    qid = r["id"]
    if qid in drop_ids:
        continue
    j = jury_map.get(qid, {})
    # 用 jury 决定填 review_status
    decision = j.get("decision", "KEEP")
    gpt_v = j.get("gpt_verdict", "?")
    claude_v = j.get("claude_verdict", "?")
    contested = (gpt_v == "fail" and claude_v != "fail") or (claude_v == "fail" and gpt_v != "fail")
    if contested:
        contested_ids.append(qid)

    r["review_status"] = "jury_keep"
    r["jury"] = {
        "decision": decision,
        "gpt_verdict": gpt_v,
        "claude_verdict": claude_v,
        "contested": contested,
        "gpt_scores": j.get("gpt_scores"),
        "claude_scores": j.get("claude_scores"),
        "gpt_reason": j.get("gpt_reason"),
        "claude_reason": j.get("claude_reason"),
    }
    # 删掉内部字段
    for k in ("_doc_id", "_chunk_idx"):
        r.pop(k, None)
    final.append(r)

print(f"[final] {len(final)} records (dropped {len(qa_records) - len(final)})")
print(f"[contested] {len(contested_ids)} ids (CONTESTED, 写入 limitation)")

# 5) Stratified 80/20 split by difficulty
buckets = defaultdict(list)
for r in final:
    buckets[r.get("difficulty", "medium")].append(r)

train, test = [], []
for diff, items in buckets.items():
    random.shuffle(items)
    cut = int(len(items) * 0.8)
    train.extend(items[:cut])
    test.extend(items[cut:])
    print(f"  [split] {diff}: total={len(items)} train={cut} test={len(items)-cut}")

random.shuffle(train)
random.shuffle(test)
print(f"[split] total train={len(train)} test={len(test)}")

# 6) 写出 JSONL
def dump_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump_jsonl(OUT / "wh_bench_v0.1_train.jsonl", train)
dump_jsonl(OUT / "wh_bench_v0.1_test.jsonl", test)
dump_jsonl(OUT / "wh_bench_v0.1_all.jsonl", final)

# 7) 写 contested 清单
(OUT / "contested_ids.txt").write_text("\n".join(contested_ids), encoding="utf-8")

# 8) 统计
stats = {
    "total": len(final),
    "dropped": len(drop_ids),
    "contested": len(contested_ids),
    "train": len(train),
    "test": len(test),
    "difficulty_dist": dict(Counter(r["difficulty"] for r in final)),
    "category_dist": dict(Counter(r.get("category", "?") for r in final)),
    "authority_dist": dict(Counter(r.get("authority", "?") for r in final)),
    "source_doc_count": len(set(r.get("source_doc") for r in final)),
}
(OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[stats]", json.dumps(stats, ensure_ascii=False, indent=2))

# 9) Parquet（HuggingFace 喜欢）
try:
    import pandas as pd
    for name, rows in [("train", train), ("test", test), ("all", final)]:
        df = pd.json_normalize(rows, max_level=1)
        df.to_parquet(OUT / f"wh_bench_v0.1_{name}.parquet", index=False)
    print("[parquet] wrote train/test/all parquet")
except Exception as e:
    print(f"[parquet] skipped: {e}")

print("\n✅ finalize done →", OUT)
