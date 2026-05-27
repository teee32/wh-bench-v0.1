"""Stage 4: human spot-check via local Flask web UI.

Reads:  data/filtered/qa_v0.1_auto.jsonl
Writes: data/filtered/qa_v0.1_reviewed.jsonl
        reports/review_<timestamp>.csv

Workflow:
  1. python scripts/04_review_app.py
  2. open http://127.0.0.1:7860/
  3. for each QA: click ✓ accept / ✗ reject / ⤼ skip
  4. progress is auto-saved every action (~/projects/wh-bench-v0.1/reports/review_state.json)
  5. when done, hit "Finalize" → writes reviewed.jsonl + final CSV

You can resume by re-running this script; state persists.

Run:
    python scripts/04_review_app.py
    python scripts/04_review_app.py --sample 30   # only review 30 random
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string

from wh_bench.utils import (
    DATA_FILTERED, REPORTS, get_logger, jsonl_read, jsonl_write,
)

log = get_logger("04_review")

STATE_FILE = REPORTS / "review_state.json"

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>wh-bench v0.1 抽检</title>
<style>
  body { font-family: -apple-system, "Helvetica Neue", "PingFang SC", sans-serif;
         max-width: 880px; margin: 30px auto; padding: 0 20px; color: #222;
         background: #fafafa; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .progress { color: #888; margin-bottom: 18px; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
          padding: 22px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .label { font-size: 12px; color: #888; text-transform: uppercase;
           letter-spacing: .5px; margin-bottom: 4px; }
  .q { font-size: 17px; font-weight: 600; margin-bottom: 12px; }
  .a { font-size: 15px; padding: 10px 14px; background: #f5f8ff;
       border-left: 3px solid #5b8def; margin-bottom: 16px;
       white-space: pre-wrap; word-break: break-word; }
  .src { font-size: 13px; padding: 10px 14px; background: #f7f7f7;
         border-left: 3px solid #bbb; max-height: 220px; overflow-y: auto;
         white-space: pre-wrap; word-break: break-word; color: #555; }
  .meta { font-size: 12px; color: #888; margin-top: 10px; }
  .btn-row { margin-top: 18px; display: flex; gap: 10px; }
  button { font-size: 15px; padding: 10px 22px; border-radius: 6px;
           border: 1px solid #ccc; background: #fff; cursor: pointer; }
  button.accept { background: #2ea043; color: #fff; border-color: #2ea043; }
  button.reject { background: #d1242f; color: #fff; border-color: #d1242f; }
  button.skip   { background: #6e7781; color: #fff; border-color: #6e7781; }
  button:hover { opacity: .9; }
  .done { padding: 30px; background: #fff; border-radius: 8px; text-align: center; }
  .done h2 { color: #2ea043; }
  .stats { font-size: 14px; color: #555; margin-top: 12px; }
  textarea { width: 100%; padding: 8px; font-size: 13px; border-radius: 4px;
             border: 1px solid #ddd; }
  kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px;
        padding: 2px 6px; font-size: 12px; font-family: monospace; }
</style>
</head>
<body>
  <h1>wh-bench v0.1 · 人工抽检</h1>
  <div class="progress" id="progress"></div>
  <div id="content"></div>
  <div style="margin-top:30px; font-size:12px; color:#888;">
    快捷键：<kbd>1</kbd> 通过 · <kbd>2</kbd> 否决 · <kbd>3</kbd> 跳过
  </div>

<script>
let state = { current: null, queue: [], reviewed: 0, accepted: 0, rejected: 0, total: 0 };

async function loadNext() {
  const r = await fetch('/api/next');
  const data = await r.json();
  state = data;
  render();
}

function render() {
  document.getElementById('progress').textContent =
    `进度 ${state.reviewed}/${state.total}  ·  通过 ${state.accepted}  ·  否决 ${state.rejected}`;
  const c = document.getElementById('content');
  if (!state.current) {
    c.innerHTML = `<div class="done">
      <h2>抽检完成 ✓</h2>
      <p class="stats">通过率: ${state.accepted}/${state.reviewed}
        = ${(100*state.accepted/Math.max(1,state.reviewed)).toFixed(1)}%</p>
      <button onclick="finalize()">写入 reviewed.jsonl 和 CSV</button>
      <div id="finalmsg"></div>
    </div>`;
    return;
  }
  const q = state.current;
  c.innerHTML = `
    <div class="card">
      <div class="label">问题 (${q.id})</div>
      <div class="q">${escapeHtml(q.question)}</div>
      <div class="label">答案</div>
      <div class="a">${escapeHtml(q.answer)}</div>
      <div class="label">原文出处 — ${escapeHtml(q.source_doc)} · ${escapeHtml(q.source_section || '')}</div>
      <div class="src">${escapeHtml(q.source_text)}</div>
      <div class="meta">类别: ${escapeHtml(q.category||'-')}  ·  难度: ${escapeHtml(q.difficulty||'-')}
        ·  机关: ${escapeHtml(q.authority||'-')}</div>
      <div class="label" style="margin-top:14px">备注（可选，否决时建议写）</div>
      <textarea id="note" rows="2" placeholder="例如：答案过于宽泛 / 与原文不符 / ..."></textarea>
      <div class="btn-row">
        <button class="accept" onclick="vote('accept')">✓ 通过 (1)</button>
        <button class="reject" onclick="vote('reject')">✗ 否决 (2)</button>
        <button class="skip"   onclick="vote('skip')">⤼ 跳过 (3)</button>
      </div>
    </div>`;
}

async function vote(verdict) {
  const note = document.getElementById('note')?.value || '';
  await fetch('/api/vote', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ id: state.current.id, verdict, note })
  });
  loadNext();
}

async function finalize() {
  const r = await fetch('/api/finalize', { method: 'POST' });
  const d = await r.json();
  document.getElementById('finalmsg').innerHTML =
    `<p style="margin-top:14px">✓ 写入完成<br>
       reviewed: <code>${d.jsonl}</code><br>
       csv:      <code>${d.csv}</code><br>
       通过率:    <strong>${(100*d.acceptance_rate).toFixed(1)}%</strong></p>`;
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',
    '"':'&quot;',"'":'&#39;'})[c]);
}

document.addEventListener('keydown', e => {
  if (!state.current) return;
  if (e.key === '1') vote('accept');
  if (e.key === '2') vote('reject');
  if (e.key === '3') vote('skip');
});

loadNext();
</script>
</body>
</html>
"""


# ── State management ────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"votes": {}}  # id -> {verdict, note, ts}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


# ── Flask app ───────────────────────────────────────────────────────────
def make_app(qa_pool: list[dict]) -> Flask:
    app = Flask(__name__)
    qa_by_id = {q["id"]: q for q in qa_pool}

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/api/next")
    def api_next():
        state = load_state()
        votes = state["votes"]
        # find next un-voted (accept skip is "vote", will be re-shown if just skipped this session)
        remaining = [q for q in qa_pool if q["id"] not in votes]
        accepted = sum(1 for v in votes.values() if v["verdict"] == "accept")
        rejected = sum(1 for v in votes.values() if v["verdict"] == "reject")
        return jsonify({
            "current": remaining[0] if remaining else None,
            "queue": [q["id"] for q in remaining[:3]],
            "reviewed": accepted + rejected,  # skipped doesn't count
            "accepted": accepted,
            "rejected": rejected,
            "total": len(qa_pool),
        })

    @app.post("/api/vote")
    def api_vote():
        body = request.get_json(force=True)
        qid = body["id"]
        verdict = body["verdict"]
        note = body.get("note", "")
        state = load_state()
        # skip = remove any prior vote, but mark recently-skipped to push to back
        if verdict == "skip":
            # actually we just don't record it, but to avoid infinite loop we
            # rotate: pop from pool and push to end
            for i, q in enumerate(qa_pool):
                if q["id"] == qid:
                    qa_pool.append(qa_pool.pop(i))
                    break
            return jsonify({"ok": True})
        state["votes"][qid] = {
            "verdict": verdict, "note": note,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        save_state(state)
        return jsonify({"ok": True})

    @app.post("/api/finalize")
    def api_finalize():
        state = load_state()
        votes = state["votes"]
        accepted_qa: list[dict] = []
        all_rows: list[dict] = []
        for q in qa_pool:
            v = votes.get(q["id"])
            if not v:
                continue
            row = {**q, "review_verdict": v["verdict"], "review_note": v["note"]}
            all_rows.append(row)
            if v["verdict"] == "accept":
                qcopy = {**q, "review_status": "human_verified", "review_note": v["note"]}
                qcopy.pop("_doc_id", None)
                qcopy.pop("_chunk_idx", None)
                accepted_qa.append(qcopy)

        out_jsonl = DATA_FILTERED / "qa_v0.1_reviewed.jsonl"
        jsonl_write(out_jsonl, accepted_qa)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_csv = REPORTS / f"review_{ts}.csv"
        if all_rows:
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

        n_total = len(all_rows)
        n_accept = sum(1 for r in all_rows if r["review_verdict"] == "accept")
        return jsonify({
            "jsonl": str(out_jsonl),
            "csv": str(out_csv),
            "n_reviewed": n_total,
            "n_accepted": n_accept,
            "acceptance_rate": n_accept / max(1, n_total),
        })

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0,
                   help="if >0, randomly subsample N QA for review")
    p.add_argument("--port", type=int,
                   default=int(os.getenv("REVIEW_PORT", "7860")))
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    src = DATA_FILTERED / "qa_v0.1_auto.jsonl"
    if not src.exists():
        log.error(f"missing {src}; run 03_filter.py first")
        return

    qas = jsonl_read(src)
    if args.sample and args.sample < len(qas):
        random.seed(42)
        qas = random.sample(qas, args.sample)
        log.info(f"sampled {len(qas)} for review")
    log.info(f"loaded {len(qas)} QA into review pool")
    log.info(f"open http://127.0.0.1:{args.port}/")

    app = make_app(qas)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
