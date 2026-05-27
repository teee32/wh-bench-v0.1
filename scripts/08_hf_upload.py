#!/usr/bin/env python3
"""
Upload wh-bench v0.1 to HuggingFace dataset repo kksk2312/wh-bench-v0.1.
- Parquet files at root (HF Datasets auto-discovery)
- README.md with YAML frontmatter at root
- DATASHEET.md, TECH_REPORT.md as documentation
- jsonl + stats.json for human-readable inspection
"""
from huggingface_hub import HfApi, upload_file
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "kksk2312/wh-bench-v0.1"
api = HfApi()

# README first (has YAML frontmatter, becomes dataset card)
print(f"[upload] README → {REPO}/README.md")
api.upload_file(
    path_or_fileobj=ROOT / "hf_README.md",
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="dataset",
    commit_message="docs: add dataset card with YAML frontmatter",
)

# Auxiliary docs
for src, dest in [
    ("DATASHEET.md", "DATASHEET.md"),
    ("TECH_REPORT.md", "TECH_REPORT.md"),
    ("LICENSE", "LICENSE"),
]:
    print(f"[upload] {src} → {REPO}/{dest}")
    api.upload_file(
        path_or_fileobj=ROOT / src,
        path_in_repo=dest,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"docs: add {dest}",
    )

# Data: parquet at root for HF auto-discovery
data_files = [
    ("data/processed/wh_bench_v0.1_train.parquet", "wh_bench_v0.1_train.parquet"),
    ("data/processed/wh_bench_v0.1_test.parquet", "wh_bench_v0.1_test.parquet"),
    ("data/processed/wh_bench_v0.1_all.parquet", "wh_bench_v0.1_all.parquet"),
    ("data/processed/wh_bench_v0.1_train.jsonl", "wh_bench_v0.1_train.jsonl"),
    ("data/processed/wh_bench_v0.1_test.jsonl", "wh_bench_v0.1_test.jsonl"),
    ("data/processed/wh_bench_v0.1_all.jsonl", "wh_bench_v0.1_all.jsonl"),
    ("data/processed/stats.json", "stats.json"),
    ("data/processed/contested_ids.txt", "contested_ids.txt"),
]
for src, dest in data_files:
    print(f"[upload] {src} → {REPO}/{dest}")
    api.upload_file(
        path_or_fileobj=ROOT / src,
        path_in_repo=dest,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"data: upload {dest}",
    )

print(f"\n✅ Done → https://huggingface.co/datasets/{REPO}")
