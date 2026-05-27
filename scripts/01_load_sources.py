"""Stage 1: load source corpora from HuggingFace and filter to topical regulations.

Reads:  HF datasets in src/wh_bench/sources.py (HF_SOURCES)
Writes: data/raw/sources.jsonl   (1 doc per line, fields: doc_id/title/authority/url/content/...)
        data/raw/manifest.json   (summary)

Run:
    python scripts/01_load_sources.py
    python scripts/01_load_sources.py --target 30   # smaller v0.1 cut
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from wh_bench.utils import DATA_RAW, get_logger, jsonl_write, stable_id
from wh_bench.sources import (
    HF_SOURCES,
    TITLE_INCLUDE_KEYWORDS,
    TITLE_EXCLUDE_KEYWORDS,
    PREFERRED_OFFICE_CATEGORIES,
    V01_TARGET_DOCS,
    V01_MIN_CONTENT_LEN,
    V01_MAX_CONTENT_LEN,
)

log = get_logger("01_load_sources")


def title_matches(title: str) -> bool:
    if not title:
        return False
    if any(k in title for k in TITLE_EXCLUDE_KEYWORDS):
        return False
    return any(k in title for k in TITLE_INCLUDE_KEYWORDS)


def normalize_content(text: str) -> str:
    """Strip excess whitespace, keep paragraph breaks."""
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# 关键词分级（数字越大越偏好）
HIGH_VALUE = [
    "安全规程", "操作规程", "作业规程", "技术规范", "技术规程",
    "受限空间", "动火", "高处作业", "粉尘防爆",
    "危险化学品", "易燃易爆", "压力容器", "锅炉",
    "安全技术", "安全防护", "电气安全", "机械安全",
    "起重机械", "矿山安全",
]
MED_VALUE = [
    "安全生产", "安全管理", "应急管理", "应急预案",
    "职业病", "职业健康", "防火", "防爆", "化学品",
    "施工安全",
]
# 地方法规通常含的前缀关键词
LOCAL_MARKERS = [
    "市", "省", "自治区", "县", "区",
    "来宾", "许昌", "商丘", "呼和浩特", "淮南", "南阳", "盘锦", "郑州",
    "内蒙古", "广西", "云南", "贵州", "陕西", "山西", "河北",
]


def score_doc(item: dict) -> float:
    """Higher = more likely a high-quality central-level industrial-safety regulation."""
    score = 0.0
    title = item.get("title", "") or ""

    # 1. 分级关键词加权
    for k in HIGH_VALUE:
        if k in title:
            score += 3.0
    for k in MED_VALUE:
        if k in title:
            score += 1.5

    # 2. 地方法规：标题前 4 个字含地名 → 减分
    title_head = title[:4]
    if any(m in title_head for m in LOCAL_MARKERS):
        score -= 3.0

    # 3. office_category 偏好
    oc = item.get("office_category") or ""
    if oc in PREFERRED_OFFICE_CATEGORIES:
        score += 3.0

    # 4. 部委强加分
    office = item.get("office", "") or ""
    for k, w in (("应急管理", 4.0), ("市场监督", 4.0), ("工业和信息化", 3.5),
                 ("安全生产监督", 4.0), ("住房和城乡建设", 3.0),
                 ("交通运输", 2.5), ("国务院", 2.0)):
        if k in office:
            score += w
            break

    # 5. 长度适中
    content_len = len(item.get("content", "") or "")
    if 3000 <= content_len <= 30000:
        score += 1.0

    # 6. 现行有效
    status = item.get("status", "") or ""
    if "现行有效" in status or "有效" in status:
        score += 0.5

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Load + filter source regulations")
    parser.add_argument("--target", type=int, default=V01_TARGET_DOCS,
                        help=f"max docs to keep (default {V01_TARGET_DOCS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="just print stats, don't write data")
    args = parser.parse_args()

    all_picked: list[dict] = []

    for src in HF_SOURCES:
        log.info(f"Loading {src['hf_repo']} ({src['split']}) ...")
        try:
            ds = load_dataset(src["hf_repo"], src.get("config", "default"),
                              split=src["split"])
        except Exception as e:
            log.error(f"  failed: {e}")
            continue
        log.info(f"  → {len(ds)} rows")

        # Stage 1: keyword filter
        candidates = []
        for row in tqdm(ds, desc=f"  filter {src['name']}", unit="row"):
            row = dict(row)  # in case it's lazy
            title = row.get(src["fields"]["title"], "") or ""
            content = row.get(src["fields"]["content"], "") or ""

            if not title_matches(title):
                continue
            if not (V01_MIN_CONTENT_LEN <= len(content) <= V01_MAX_CONTENT_LEN):
                continue
            candidates.append(row)

        log.info(f"  keyword-filtered: {len(candidates)} docs")

        # Stage 2: score + take top-N
        for row in candidates:
            row["_score"] = score_doc({
                "title": row.get(src["fields"]["title"], "") or "",
                "content": row.get(src["fields"]["content"], "") or "",
                "office": row.get(src["fields"]["office"], "") or "",
                "office_category": row.get(src["fields"]["office_category"], "") or "",
                "status": row.get("status", "") or "",
            })
        candidates.sort(key=lambda r: r["_score"], reverse=True)

        keep = candidates[: args.target]
        log.info(f"  keeping top {len(keep)} by score")

        for i, row in enumerate(keep):
            content = normalize_content(row.get(src["fields"]["content"], ""))
            doc = {
                "doc_id": stable_id(f"src{src['name']}", i + 1),
                "title": row.get(src["fields"]["title"], "") or "",
                "authority": row.get(src["fields"]["office"], "") or "未知",
                "office_category": row.get(src["fields"]["office_category"], "") or "",
                "doc_type": row.get(src["fields"]["type"], "") or "",
                "publish_date": str(row.get(src["fields"]["publish_date"], "") or ""),
                "url": f"https://huggingface.co/datasets/{src['hf_repo']}",  # corpus-level
                "content": content,
                "content_len": len(content),
                "source_corpus": src["hf_repo"],
                "license": src["license"],
                "score": row["_score"],
                "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            all_picked.append(doc)

    # ── Output ────────────────────────────────────────────────────────────
    log.info(f"Total picked: {len(all_picked)} docs")

    if all_picked:
        # show top-10 titles for sanity
        log.info("Top 10 picked titles:")
        for d in all_picked[:10]:
            log.info(f"  [{d['score']:.1f}] {d['title']}  ({d['content_len']} chars)")

    if args.dry_run:
        log.info("--dry-run: not writing files")
        return

    out = DATA_RAW / "sources.jsonl"
    jsonl_write(out, all_picked)
    log.info(f"✓ wrote {out}")

    manifest = {
        "version": "0.1",
        "n_docs": len(all_picked),
        "by_authority": {},
        "by_office_category": {},
        "total_chars": sum(d["content_len"] for d in all_picked),
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    for d in all_picked:
        manifest["by_authority"][d["authority"]] = manifest["by_authority"].get(d["authority"], 0) + 1
        oc = d["office_category"]
        manifest["by_office_category"][oc] = manifest["by_office_category"].get(oc, 0) + 1

    (DATA_RAW / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"✓ wrote {DATA_RAW / 'manifest.json'}")


if __name__ == "__main__":
    main()
