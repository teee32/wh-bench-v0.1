"""Stage 2: distill QA pairs from source regulations using an LLM.

Reads:  data/raw/sources.jsonl
Writes: data/distilled/qa_raw.jsonl

Strategy:
  - chunk each doc into ~1500-char paragraphs (respecting article boundaries)
  - prompt the LLM with strict instructions: answer MUST be quotable from the chunk
  - parse JSON list of {question, answer, source_text}
  - dedupe trivially within doc

Run:
    python scripts/02_distill.py                 # full pass
    python scripts/02_distill.py --max-docs 5    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from wh_bench.utils import (
    DATA_RAW, DATA_DISTILLED, get_logger, jsonl_read, jsonl_write,
    get_llm_client, stable_id,
)

log = get_logger("02_distill")

CHUNK_TARGET = 1500
CHUNK_MAX = 2200
CHUNK_MIN = 400


# ── Chunking ────────────────────────────────────────────────────────────
ARTICLE_RE = re.compile(r"(第[一二三四五六七八九十百千零〇\d]+条)")

def chunk_doc(text: str) -> list[dict]:
    """Split content into chunks, prefer article boundaries.

    Returns list of {section, text}.
    """
    if not text:
        return []

    # split on 第X条 markers; keep the marker as section header
    parts = ARTICLE_RE.split(text)
    chunks: list[dict] = []

    if len(parts) >= 3:
        # parts = [preamble, '第一条', body, '第二条', body, ...]
        preamble = parts[0].strip()
        if len(preamble) >= CHUNK_MIN:
            chunks.append({"section": "前言", "text": preamble[:CHUNK_MAX]})
        for i in range(1, len(parts) - 1, 2):
            section = parts[i].strip()
            body = parts[i + 1].strip()
            if not body:
                continue
            chunks.append({"section": section, "text": (section + " " + body)[:CHUNK_MAX]})
    else:
        # no article markers: split by paragraphs greedily
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 2 > CHUNK_TARGET and len(buf) >= CHUNK_MIN:
                chunks.append({"section": "", "text": buf[:CHUNK_MAX]})
                buf = p
            else:
                buf = (buf + "\n\n" + p) if buf else p
        if buf and len(buf) >= CHUNK_MIN:
            chunks.append({"section": "", "text": buf[:CHUNK_MAX]})

    # merge tiny chunks with neighbor — also extend section range
    merged: list[dict] = []
    for c in chunks:
        if merged and len(c["text"]) < CHUNK_MIN:
            merged[-1]["text"] += "\n" + c["text"]
            # extend section: "第一条" + "第三条"  →  "第一条—第三条"
            prev_sec = merged[-1]["section"]
            cur_sec = c["section"]
            if cur_sec and cur_sec != prev_sec:
                if "—" in prev_sec:
                    head = prev_sec.split("—")[0]
                    merged[-1]["section"] = f"{head}—{cur_sec}"
                else:
                    merged[-1]["section"] = f"{prev_sec}—{cur_sec}"
        else:
            merged.append(dict(c))
    return merged


# ── LLM prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是中文工业安全规程评测集的标注助手。

任务：给你一段中文规程原文，请生成 1-3 个高质量的问答对（QA pair）。

严格要求：
1. **答案必须可以直接从原文引用或紧密改写**，不可推断、不可外部知识
2. 问题应清晰、具体，能考察是否真正理解了规程的关键约束
3. 优先考察：必须做的步骤 / 禁止行为 / 数值阈值 / 责任主体 / 适用场景
4. 不要生成过于宽泛的问题（如"什么是安全生产"）
5. 不要生成无答案题（如果原文里找不到清晰答案，少生成或不生成）

输出 JSON 格式：
{
  "qa_pairs": [
    {
      "question": "...",
      "answer": "...",
      "source_quote": "原文中支撑该答案的最短引用片段",
      "category": "受限空间/动火作业/危险化学品/...",
      "difficulty": "easy|medium|hard"
    }
  ]
}

如果该段不适合出题（如纯目录、纯定义术语、纯过渡段），返回 {"qa_pairs": []}。"""


USER_TEMPLATE = """规程标题：{title}
章节：{section}

原文：
{chunk}

请生成 1-3 个问答对（如果文本质量低则少出或不出）。直接输出 JSON。"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_llm(client, model: str, title: str, section: str, chunk: str,
             temperature: float) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                title=title, section=section or "(未标注)", chunk=chunk)},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=1500,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # try to salvage
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    return data.get("qa_pairs", [])


# ── Worker ──────────────────────────────────────────────────────────────
def distill_doc(client, model: str, doc: dict, max_chunks: int,
                max_qa_per_chunk: int, temperature: float) -> list[dict]:
    chunks = chunk_doc(doc["content"])
    if not chunks:
        return []
    chunks = chunks[:max_chunks]
    out: list[dict] = []
    for ch_idx, ch in enumerate(chunks):
        try:
            qas = call_llm(client, model, doc["title"], ch["section"],
                           ch["text"], temperature)
        except Exception as e:
            log.warning(f"  chunk fail {doc['doc_id']}[{ch_idx}]: {e}")
            continue
        for q in qas[:max_qa_per_chunk]:
            if not isinstance(q, dict):
                continue
            out.append({
                "_doc_id": doc["doc_id"],
                "_chunk_idx": ch_idx,
                "question": (q.get("question") or "").strip(),
                "answer": (q.get("answer") or "").strip(),
                "source_doc": doc["title"],
                "source_url": doc["url"],
                "source_section": ch["section"],
                "source_text": ch["text"],
                "source_quote": (q.get("source_quote") or "").strip(),
                "category": (q.get("category") or "").strip(),
                "difficulty": (q.get("difficulty") or "medium").strip(),
                "authority": doc["authority"],
                "review_status": "auto",
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=0,
                   help="0 = all (default); else limit for smoke test")
    p.add_argument("--max-chunks", type=int,
                   default=int(os.getenv("DISTILL_MAX_CHUNKS_PER_DOC", "8")))
    p.add_argument("--max-qa", type=int,
                   default=int(os.getenv("DISTILL_MAX_QA_PER_CHUNK", "3")))
    p.add_argument("--temperature", type=float,
                   default=float(os.getenv("DISTILL_TEMPERATURE", "0.3")))
    p.add_argument("--parallel", type=int,
                   default=int(os.getenv("DISTILL_MAX_PARALLEL", "4")))
    args = p.parse_args()

    src_path = DATA_RAW / "sources.jsonl"
    if not src_path.exists():
        log.error(f"missing {src_path}; run 01_load_sources.py first")
        sys.exit(1)

    docs = jsonl_read(src_path)
    if args.max_docs:
        docs = docs[: args.max_docs]
    log.info(f"distilling {len(docs)} docs | chunks≤{args.max_chunks} qa≤{args.max_qa}/chunk parallel={args.parallel}")

    client, model = get_llm_client()
    log.info(f"LLM client ready: model={model}")

    all_qa: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {
            ex.submit(distill_doc, client, model, doc,
                      args.max_chunks, args.max_qa, args.temperature): doc
            for doc in docs
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="distilling"):
            doc = futs[fut]
            try:
                qas = fut.result()
                all_qa.extend(qas)
                tqdm.write(f"  {doc['doc_id']}  +{len(qas)} QA  ({doc['title'][:30]})")
            except Exception as e:
                log.error(f"  doc fail {doc['doc_id']}: {e}")

    # assign global ids
    for i, q in enumerate(all_qa):
        q["id"] = stable_id("wh-bench", i + 1)
    # reorder fields
    ordered_keys = ["id", "question", "answer", "source_doc", "source_url",
                    "source_section", "source_text", "source_quote",
                    "category", "difficulty", "authority",
                    "review_status", "_doc_id", "_chunk_idx"]
    all_qa = [{k: q.get(k, "") for k in ordered_keys} for q in all_qa]

    out = DATA_DISTILLED / "qa_raw.jsonl"
    jsonl_write(out, all_qa)
    elapsed = time.time() - t0
    log.info(f"✓ {len(all_qa)} QA pairs → {out}  ({elapsed:.1f}s)")

    stats = {
        "n_docs": len(docs),
        "n_qa": len(all_qa),
        "qa_per_doc_avg": round(len(all_qa) / max(1, len(docs)), 2),
        "model": model,
        "temperature": args.temperature,
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (DATA_DISTILLED / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  stats: {stats}")


if __name__ == "__main__":
    main()
