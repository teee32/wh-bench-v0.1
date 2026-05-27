#!/usr/bin/env python3
"""
Jury 决策：合并 gpt-5.5 + claude-sonnet-4-6 两个 judge 的评分。

策略（用户选项 C）：两个 judge 都 fail 才剔除。
- 双 fail   → DROP（强证据数据有问题）
- 单 fail   → KEEP（保留，但标记为 contested）
- 双 pass   → KEEP
- 其他组合  → KEEP（borderline 容忍）

输出：
- jury_verdicts.jsonl       每条 QA 的双 judge 评分对照 + 最终 jury 决定
- jury_drop_list.txt        要剔除的 QA id（双 fail）
- jury_contested.md         单 fail 但保留的争议条目（透明披露用）
- reports/jury_summary.md   人类可读总结报告
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path("/root/projects/wh-bench-v0.1")
GPT_PATH = ROOT / "data/judged/qa_judged.jsonl"
CLAUDE_PATH = ROOT / "data/judged/qa_judged_claude.jsonl"
OUT_DIR = ROOT / "data/judged"
REPORT_DIR = ROOT / "reports"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    gpt = {r["id"]: r for r in load_jsonl(GPT_PATH)}
    claude = {r["id"]: r for r in load_jsonl(CLAUDE_PATH)}
    ids = sorted(set(gpt) | set(claude))
    print(f"GPT-5.5 评了 {len(gpt)} 条；Claude-Sonnet-4-6 评了 {len(claude)} 条；联合 {len(ids)} 条")

    jury = []
    for qid in ids:
        g = gpt.get(qid, {"verdict": "MISSING"})
        c = claude.get(qid, {"verdict": "MISSING"})
        gv, cv = g.get("verdict", "ERROR"), c.get("verdict", "ERROR")

        # Jury 决策
        if gv == "fail" and cv == "fail":
            decision = "DROP"
            note = "双 judge 都判 fail，强证据剔除"
        elif gv == "fail" or cv == "fail":
            decision = "KEEP_CONTESTED"
            note = f"单 judge fail（gpt={gv}, claude={cv}），保留 + dataset card 披露"
        elif gv == "pass" and cv == "pass":
            decision = "KEEP_CLEAN"
            note = "双 pass"
        else:
            decision = "KEEP"
            note = f"borderline 容忍（gpt={gv}, claude={cv}）"

        # 拼合两份分数
        merged = {
            "id": qid,
            "decision": decision,
            "note": note,
            "gpt_verdict": gv,
            "claude_verdict": cv,
            "gpt_scores": {k: g.get(k) for k in ("groundedness","accuracy","clarity","specificity")} if gv not in ("ERROR","MISSING") else None,
            "claude_scores": {k: c.get(k) for k in ("groundedness","accuracy","clarity","specificity")} if cv not in ("ERROR","MISSING") else None,
            "gpt_issues": g.get("issues", []),
            "claude_issues": c.get("issues", []),
            "gpt_reason": g.get("reason", ""),
            "claude_reason": c.get("reason", ""),
            "question": g.get("question") or c.get("question", ""),
            "answer": g.get("answer") or c.get("answer", ""),
            "source_doc": g.get("source_doc") or c.get("source_doc", ""),
            "source_quote": g.get("source_quote") or c.get("source_quote", ""),
            "difficulty": g.get("difficulty") or c.get("difficulty", ""),
        }
        jury.append(merged)

    # 写 jury 全表
    jury_path = OUT_DIR / "jury_verdicts.jsonl"
    with jury_path.open("w") as f:
        for r in jury:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 决策分布
    decision_count = Counter(r["decision"] for r in jury)
    print("\n=== Jury 决策分布 ===")
    for k, n in decision_count.most_common():
        print(f"  {k}: {n} ({n/len(jury)*100:.1f}%)")

    # 双 judge verdict cross table
    cross = Counter((r["gpt_verdict"], r["claude_verdict"]) for r in jury)
    print("\n=== 双 judge 一致性矩阵 ===")
    print("                  claude:pass  claude:borderline  claude:fail   claude:其他")
    for gv in ("pass", "borderline", "fail"):
        row = f"  gpt:{gv:<11}"
        for cv in ("pass", "borderline", "fail"):
            row += f"  {cross.get((gv,cv),0):>10}"
        # 其他 = error/missing
        other = sum(v for (g,c),v in cross.items() if g==gv and c not in ("pass","borderline","fail"))
        row += f"  {other:>11}"
        print(row)

    # Drop list
    drop_ids = sorted([r["id"] for r in jury if r["decision"] == "DROP"])
    drop_path = OUT_DIR / "jury_drop_list.txt"
    drop_path.write_text("\n".join(drop_ids))
    print(f"\nDROP（双 fail 剔除）{len(drop_ids)} 条: {drop_ids}")

    # Contested = 单 fail 但保留
    contested = [r for r in jury if r["decision"] == "KEEP_CONTESTED"]
    print(f"CONTESTED（单 fail 保留）{len(contested)} 条")

    # 人类可读报告
    REPORT_DIR.mkdir(exist_ok=True)
    report = REPORT_DIR / "jury_summary.md"
    lines = ["# Jury 复审总结：gpt-5.5 + claude-sonnet-4-6 双 judge",
             "",
             f"- 总样本：{len(jury)}",
             f"- 决策：DROP {decision_count.get('DROP',0)} / KEEP_CONTESTED {decision_count.get('KEEP_CONTESTED',0)} / KEEP_CLEAN {decision_count.get('KEEP_CLEAN',0)} / KEEP {decision_count.get('KEEP',0)}",
             f"- 一致 pass 率：{cross.get(('pass','pass'),0)/len(jury)*100:.1f}%",
             "",
             "## 一致性矩阵",
             "",
             "| GPT \\\\ Claude | pass | borderline | fail | 其他 |",
             "|---|---|---|---|---|"]
    for gv in ("pass", "borderline", "fail"):
        row = f"| **{gv}** |"
        for cv in ("pass", "borderline", "fail"):
            row += f" {cross.get((gv,cv),0)} |"
        other = sum(v for (g,c),v in cross.items() if g==gv and c not in ("pass","borderline","fail"))
        row += f" {other} |"
        lines.append(row)

    lines.extend([
        "",
        f"## 双 judge 一致剔除（DROP）：{len(drop_ids)} 条",
        ""
    ])
    drop_set = set(drop_ids)
    for r in jury:
        if r["id"] in drop_set:
            lines.append(f"### {r['id']} | {r['source_doc']}")
            lines.append(f"- 难度: {r['difficulty']}")
            lines.append(f"- 问题: {r['question']}")
            lines.append(f"- 答案: {r['answer']}")
            lines.append(f"- 原文引用: {r['source_quote'][:200]}")
            lines.append(f"- GPT issues: {r['gpt_issues']} — {r['gpt_reason']}")
            lines.append(f"- Claude issues: {r['claude_issues']} — {r['claude_reason']}")
            lines.append("")

    lines.extend([
        "",
        f"## 单 judge fail 但保留（CONTESTED）：{len(contested)} 条",
        "",
        "*这些条目一个 judge 认为 fail，另一个认为 pass/borderline — 数据集 card 中作为已知 limitation 透明披露。*",
        ""
    ])
    for r in contested:
        lines.append(f"### {r['id']} | gpt={r['gpt_verdict']} / claude={r['claude_verdict']}")
        lines.append(f"- 问题: {r['question']}")
        lines.append(f"- 答案: {r['answer']}")
        lines.append(f"- GPT: {r['gpt_reason']}")
        lines.append(f"- Claude: {r['claude_reason']}")
        lines.append("")

    report.write_text("\n".join(lines))
    print(f"\n报告：{report}")
    print(f"Drop list: {drop_path}")
    print(f"Jury verdicts: {jury_path}")


if __name__ == "__main__":
    main()
