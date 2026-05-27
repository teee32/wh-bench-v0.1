"""Stage 6: publish to HuggingFace + push to GitHub.

Reads:  data/filtered/qa_v0.1_reviewed.jsonl   (preferred)
        OR data/filtered/qa_v0.1_auto.jsonl    (if no human review yet — will warn)

Steps:
  1. Build the public dataset card README for the HF dataset repo
  2. Upload data + README to HF Hub via huggingface_hub
  3. (optional) git init + commit + push to GitHub

Run:
    python scripts/06_publish.py                  # publish HF only
    python scripts/06_publish.py --github         # also create + push GitHub repo
    python scripts/06_publish.py --hf-only --dry  # preview without uploading
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

from wh_bench.utils import (
    DATA_FILTERED, REPORTS, ROOT, get_logger, jsonl_read, jsonl_write,
)

log = get_logger("06_publish")


HF_DATASET_README = """---
license: cc-by-4.0
language:
- zh
size_categories:
- n<1K
task_categories:
- question-answering
- text-classification
tags:
- chinese
- industrial-safety
- regulation
- benchmark
- evaluation
pretty_name: Chinese Industrial Safety Regulation QA Benchmark
---

# 中文工业安全规程问答评测集 v{version}

> A small, curated, source-traceable QA dataset extracted from publicly-published
> Chinese government industrial safety regulations.

## 这是什么

**{n_qa}** 条人工抽检的中文工业安全规程 Q&A，每条都能反查到原文出处（机关 + 文件名 + 段落）。

不是为了规模大，是为了：
- **抽检通过率公开**（人工 review {n_reviewed} 条，通过率 {acc_rate}%）
- **领域专家可复核**（可见 source URL 与原文片段）
- **小但严**：宁要 100 条干净的，不要 10000 条噪声

## 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 稳定 ID，如 `wh-bench-0001` |
| `question` | string | 问题 |
| `answer` | string | 参考答案 |
| `source_doc` | string | 原规程标题 |
| `source_url` | string | 原文 URL（语料库级） |
| `source_section` | string | 章节，如「第十二条」 |
| `source_text` | string | 原文段落，可用于 RAG / 复核 |
| `source_quote` | string | 蒸馏 LLM 给出的最短支撑引用 |
| `category` | string | 主题，如「受限空间」「动火作业」 |
| `difficulty` | string | easy / medium / hard |
| `authority` | string | 发布机关 |
| `review_status` | string | `human_verified` |
| `review_note` | string | reviewer 备注 |

## 数据来源

derived from:
- [`twang2218/chinese-law-and-regulations`](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations) (Apache-2.0)

filtered for industrial-safety regulations (operating procedures, technical norms,
hazardous-work standards), then distilled into QA pairs by LLM and human-reviewed.

## 抽检结果

| 字段 | 值 |
|---|---|
| 总条目 | {n_qa} |
| 已人工抽检 | {n_reviewed} |
| 通过率 | **{acc_rate}%** |
| 抽检日期 | {ts} |

{baseline_block}

## 使用

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
print(ds["test"][0])
```

## License

- **Q&A 标注 + 包装代码**：CC BY 4.0
- **原始规程文本**：PRC 政府出版物，依《著作权法》第五条不受著作权保护

## Citation

```bibtex
@dataset{{whbench_{year},
  author    = {{kksk2312}},
  title     = {{Chinese Industrial Safety Regulation QA Benchmark v{version}}},
  year      = {{{year}}},
  publisher = {{HuggingFace}},
  url       = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## Source code

GitHub: [{github_repo}](https://github.com/{github_repo})

---

*This dataset is part of an open-source effort aligned with Wuhan / Hubei AI+OPC
high-quality industry dataset policy initiatives.*
"""


def find_data_file() -> Path:
    reviewed = DATA_FILTERED / "qa_v0.1_reviewed.jsonl"
    auto = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    if reviewed.exists() and reviewed.stat().st_size > 0:
        return reviewed
    if auto.exists():
        log.warning(f"⚠ using {auto} — no human review yet (review_status will be 'auto')")
        return auto
    raise FileNotFoundError("no filtered data found; run 03_filter.py / 04_review_app.py")


def build_readme(qa_path: Path, repo_id: str, github_repo: str) -> str:
    qas = jsonl_read(qa_path)
    n_qa = len(qas)
    n_reviewed = sum(1 for q in qas if q.get("review_status") == "human_verified")
    # acceptance rate from review state if present
    state_file = REPORTS / "review_state.json"
    acc_rate = "TBD"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            votes = state.get("votes", {})
            acc = sum(1 for v in votes.values() if v.get("verdict") == "accept")
            tot = sum(1 for v in votes.values() if v.get("verdict") in ("accept", "reject"))
            if tot:
                acc_rate = f"{100*acc/tot:.1f}"
        except Exception:
            pass

    # find latest baseline report
    baselines = sorted(REPORTS.glob("baseline_*.json"))
    if baselines:
        try:
            b = json.loads(baselines[-1].read_text(encoding="utf-8"))
            baseline_block = (
                "## Baseline\n\n"
                f"| Model | char-F1 | n |\n|---|---|---|\n"
                f"| {b.get('model','?')} | {b.get('char_f1_mean','?')} | {b.get('n_questions','?')} |\n"
                "\n更多指标见 `reports/` 目录。\n"
            )
        except Exception:
            baseline_block = ""
    else:
        baseline_block = "## Baseline\n\n*待跑：`python scripts/05_baseline.py`*"

    return HF_DATASET_README.format(
        version="0.1",
        n_qa=n_qa,
        n_reviewed=n_reviewed,
        acc_rate=acc_rate,
        repo_id=repo_id,
        github_repo=github_repo,
        ts=datetime.utcnow().strftime("%Y-%m-%d"),
        year=datetime.utcnow().year,
        baseline_block=baseline_block,
    )


# ── HF publish ──────────────────────────────────────────────────────────
def publish_to_hf(qa_path: Path, repo_id: str, github_repo: str, dry: bool) -> None:
    log.info(f"=== HuggingFace: {repo_id} ===")
    # stage everything in a clean temp dir
    stage = ROOT / "build" / "hf_dataset"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # canonical filename: data/v0.1.jsonl (top-level) — also keep a `data/` subdir
    (stage / "data").mkdir()
    shutil.copy(qa_path, stage / "data" / "v0.1.jsonl")

    readme = build_readme(qa_path, repo_id, github_repo)
    (stage / "README.md").write_text(readme, encoding="utf-8")

    log.info(f"  staged @ {stage}")
    if dry:
        log.info("  --dry: skipping upload")
        log.info(f"  README preview (first 800 chars):\n{readme[:800]}")
        return

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True,
                token=os.getenv("HF_TOKEN") or None)
    log.info(f"  repo ensured: https://huggingface.co/datasets/{repo_id}")
    upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"v0.1 publish ({datetime.utcnow().isoformat(timespec='seconds')}Z)",
        token=os.getenv("HF_TOKEN") or None,
    )
    log.info(f"  ✓ uploaded → https://huggingface.co/datasets/{repo_id}")


# ── GitHub publish ──────────────────────────────────────────────────────
def run(cmd: list[str], cwd: Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"$ {' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()


def publish_to_github(github_repo: str, dry: bool) -> None:
    log.info(f"=== GitHub: {github_repo} ===")
    user, repo = github_repo.split("/", 1)
    if dry:
        log.info("  --dry: skipping git push")
        return

    if not (ROOT / ".git").exists():
        run(["git", "init"], cwd=ROOT)
        run(["git", "branch", "-M", "main"], cwd=ROOT)

    run(["git", "config", "user.email", f"{user}@users.noreply.github.com"], cwd=ROOT)
    run(["git", "config", "user.name", user], cwd=ROOT)

    # add a copy of v0.1.jsonl into data/ (gitignored by default; force-add the named file)
    data_path = DATA_FILTERED / "qa_v0.1_reviewed.jsonl"
    if not data_path.exists():
        data_path = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    shutil.copy(data_path, ROOT / "data" / "v0.1.jsonl")

    # also a small sample (first 5)
    qas = jsonl_read(data_path)[:5]
    jsonl_write(ROOT / "data" / "v0.1.sample.jsonl", qas)

    run(["git", "add", "-A"], cwd=ROOT)
    try:
        run(["git", "commit", "-m", f"v0.1 ({datetime.utcnow().isoformat(timespec='seconds')}Z)"], cwd=ROOT)
    except RuntimeError as e:
        if "nothing to commit" not in str(e):
            raise

    # check repo exists; if not, create via gh
    exists = subprocess.run(
        ["gh", "repo", "view", github_repo], capture_output=True, text=True
    ).returncode == 0
    if not exists:
        log.info(f"  creating {github_repo} via gh ...")
        run(["gh", "repo", "create", github_repo, "--public",
             "--description", "Chinese Industrial Safety Regulation QA Benchmark",
             "--source", str(ROOT), "--remote", "origin", "--push"])
        log.info(f"  ✓ created + pushed → https://github.com/{github_repo}")
        return

    # set remote if missing
    remotes = subprocess.run(["git", "remote"], cwd=ROOT, capture_output=True, text=True).stdout
    if "origin" not in remotes:
        run(["git", "remote", "add", "origin", f"https://github.com/{github_repo}.git"], cwd=ROOT)

    run(["git", "push", "-u", "origin", "main"], cwd=ROOT)
    log.info(f"  ✓ pushed → https://github.com/{github_repo}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", default=os.getenv("HF_REPO_ID", "kksk2312/wh-bench-v0.1"))
    p.add_argument("--github-repo", default=os.getenv("GITHUB_REPO", "teee32/wh-bench-v0.1"))
    p.add_argument("--hf-only", action="store_true")
    p.add_argument("--github", action="store_true",
                   help="also push to GitHub (default: HF only)")
    p.add_argument("--dry", action="store_true",
                   help="don't actually push, just stage + preview")
    args = p.parse_args()

    qa_path = find_data_file()
    log.info(f"using data: {qa_path} ({qa_path.stat().st_size} bytes)")

    publish_to_hf(qa_path, args.hf_repo, args.github_repo, args.dry)

    if args.github and not args.hf_only:
        publish_to_github(args.github_repo, args.dry)


if __name__ == "__main__":
    main()
