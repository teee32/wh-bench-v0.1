# Technical Report — wh-bench v0.1

**A Dual-Jury Verified Chinese Legal-Regulation QA Benchmark**

*teee32, May 2026*

---

## Abstract

We present **wh-bench v0.1**, a 300-pair Chinese legal-regulation QA benchmark constructed via a three-stage pipeline: LLM distillation, rule-based filtering, and **dual independent LLM jury review** using GPT-5.5 and Claude Sonnet. The dataset spans 50 Chinese laws and regulations across 14 issuing authorities, with each pair anchored to a verbatim source quote enabling reproducible verification. We release CONTESTED samples (single-judge fail) with metadata to support future research on evaluator disagreement.

---

## 1. Introduction

Chinese legal QA benchmarks today are dominated by JEC-QA (judicial exam) and similar adversarial-style datasets. While valuable for testing reasoning, they leave a gap for **lightweight, source-anchored, regulation-grounded evaluation** — the kind of factual recall test useful when triaging small Chinese LLMs for legal-domain deployment.

`wh-bench v0.1` targets this gap with three properties:

1. **Source-anchored**: every QA pair carries a `source_quote` (1–3 sentences from the regulation) and the full `source_text` chunk
2. **Dual-jury verified**: both GPT-5.5 and Claude Sonnet 4-6 independently score each pair on four dimensions
3. **Transparent**: drop list, contested samples, and full judge reasoning are public

---

## 2. Construction Pipeline

```
                    ┌──────────────────────┐
                    │ twang2218/chinese-law │  22,552 docs
                    └──────────┬───────────┘
                               │ keyword regex
                               ▼
                       852 candidates
                               │ gpt-5.4-mini scoring
                               ▼
                          Top 50 docs
                               │ chunk + distill
                               ▼  (gpt-5.4-mini, prompt-enforced quoting)
                       305 raw QA pairs
                               │ rule filter
                               ▼  (field completeness, quote reachability, length)
                       303 filtered pairs
                       ┌───────┴───────┐
                       ▼               ▼
                  GPT-5.5 judge   Claude Sonnet judge
                  (4-dim score)    (4-dim score)
                       └───────┬───────┘
                               │ jury merge
                               ▼  (drop iff BOTH fail)
                          300 final pairs
                               │ stratified split
                               ▼  (80/20 by difficulty)
                       train 239 / test 61
```

### 2.1 Source Selection

The candidate pool starts with `twang2218/chinese-law-and-regulations` (22,552 markdown files). A regex pre-filter extracts files containing `条例 | 办法 | 规定 | 法 | 规范` in the title, yielding 852 documents.

We then use `gpt-5.4-mini` to score each candidate on three criteria:
- **Article density** (number of enumerated provisions per kilobyte)
- **Question-ability** (estimated number of distinct factual queries possible)
- **Topical relevance** (alignment with civil/administrative law over criminal law)

Top 50 documents are advanced to distillation.

### 2.2 Distillation

Each document is split into chunks of ≤ 3000 characters. A prompt instructs `gpt-5.4-mini` to produce 5–8 QA pairs per document with the following constraints:

- Question: 5–200 chars, factual, single-answer
- Answer: 10–500 chars, must be derivable from the source
- `source_quote`: verbatim snippet from the chunk that grounds the answer
- `difficulty`: self-tagged easy/medium/hard

Total yield: 305 pairs across 50 documents (mean 6.1 QA/doc).

### 2.3 Rule-Based Filtering

Pairs are dropped if:
- Any required field is empty
- `source_quote` cannot be fuzzy-matched (similarity < 0.6) within `source_text`
- Question or answer length is out of range
- Duplicate `(question, source_doc)` keys

Result: **303 pairs** (drop rate 0.7%).

### 2.4 Dual-Jury Review

Each pair is independently scored by **two judges from distinct model families**:

| Judge | Model | Provider |
| --- | --- | --- |
| Judge A | gpt-5.5 | OpenAI (via foxnio gateway) |
| Judge B | claude-sonnet-4-6 | Anthropic (via foxnio gateway) |

Each judge returns:
- Four scores (1–5): `groundedness`, `accuracy`, `clarity`, `specificity`
- A verdict: `pass` / `borderline` / `fail`
- Free-form reasoning + issue list

Both judges receive identical prompts and identical input (full source text + QA pair). They never see each other's verdicts.

#### 2.4.1 Inter-Judge Agreement

The full 3×3 confusion matrix on 303 pairs:

| | Claude pass | Claude borderline | Claude fail | **GPT total** |
| --- | --- | --- | --- | --- |
| **GPT pass**       | **203** |  72 |  14 | 289 (95.4%) |
| **GPT borderline** |     2  |   8 |   1 |  11 ( 3.6%) |
| **GPT fail**       |     0  |   0 | **3** |   3 ( 1.0%) |
| **Claude total**   |  205   |  80 |  18 | 303 |
| | (67.7%) | (26.4%) | (5.9%) | |

Key observations:

- **Claude is systematically stricter** (18 fail vs 3) — consistent with prior reports of Anthropic models being more conservative on factual tasks
- **Both-pass rate**: 67.0% (203/303). This is the *conservative ceiling* on dataset quality.
- **Both-fail rate**: 1.0% (3/303). This is what we hard-drop.
- **Disagreement (one fail, the other not)**: 4.95% (15/303). These are CONTESTED.

#### 2.4.2 Drop Decision (Strategy A: Strict Intersection)

We adopt the **strict-intersection** rule:

> Drop a pair iff **both** judges return `fail`.

Rationale:
- A single judge fail is **noisy signal**: the harder judge (Claude) has a base fail rate of 5.9%, much of which the lenient judge (GPT) tolerates as `borderline pass`
- Strict intersection minimizes false-drops while still catching unanimous problems
- CONTESTED pairs are kept and **flagged in the `jury.contested` field**, leaving downstream filtering as a user choice

Dropped pairs (3): `wh-bench-0176, 0246, 0297`. All three involved either:
1. Numeric errors (0176: "30 days" written as "50 days")
2. Chunk truncation losing main subject lists (0246, 0297)

#### 2.4.3 CONTESTED Samples

15 pairs received divergent verdicts. These are valuable as a research artifact:

- **GPT-pass × Claude-fail (14 pairs)**: typically Claude flagged minor completeness gaps that GPT considered acceptable
- **Claude-pass × GPT-fail (0 pairs)**: never happened — GPT-fail always escalated to Claude
- **GPT-borderline × Claude-fail (1 pair)**: a case where neither judge endorsed the pair but they disagreed on severity

We retain all 15 with `jury.contested = true`, allowing researchers to study evaluator-disagreement effects.

### 2.5 Train/Test Split

Stratified 80/20 split by `difficulty`, fixed seed 42:

| Split | easy | medium | hard | total |
| --- | --- | --- | --- | --- |
| train | 134 | 95 | 10 | **239** |
| test  |  33 | 25 |  3 | **61**  |

Within each stratum, items are randomly assigned and then globally shuffled.

---

## 3. Dataset Characteristics

### 3.1 Coverage

- **50 source documents** spanning 14 issuing authorities
- **156** pairs from State Council regulations (国务院条例)
- **29** from National People's Congress Standing Committee laws (全国人大常委会)
- **115** from local-level regulations (8 provincial/municipal authorities)

### 3.2 Difficulty Distribution

| Difficulty | Count | % | GPT pass rate¹ | Claude pass rate¹ |
| --- | --- | --- | --- | --- |
| easy   | 167 | 55.7% | 98.8% | 81.4% |
| medium | 120 | 40.0% | 92.6% | 56.7% |
| hard   |  13 |  4.3% | 78.6% | 30.8% |

*¹ Pass rate measured before final drop, on the 303-pair set.*

The skew toward easy is a **deliberate v0.1 design**: the goal is to provide a **clean knowledge-recall floor**, not a reasoning challenge. v0.2 will rebalance toward hard.

### 3.3 Score Distribution (Average across 303 pairs)

| Dimension | GPT-5.5 | Claude Sonnet |
| --- | --- | --- |
| groundedness | 4.91 | 4.62 |
| accuracy     | 4.86 | 4.41 |
| clarity      | 4.95 | 4.78 |
| specificity  | 4.84 | 4.31 |
| **mean**     | **4.89** | **4.53** |

Both judges score above 4.5 on average, with GPT consistently more lenient by ~0.36 points.

---

## 4. Quality Assurance

### 4.1 Three-Layer Verification

1. **Distillation prompt**: explicit `source_quote` extraction enforces grounding at generation time
2. **Rule filter**: fuzzy match (≥0.6 similarity) of `source_quote` against `source_text` ensures the quote actually exists
3. **Dual jury**: independent scoring on groundedness + accuracy catches subtle drift

### 4.2 Manual Spot-Check

We manually inspected 5 GPT-pass / Claude-fail samples to verify Claude's stricter judgments:
- 4/5 were trivial completeness gaps (Claude flagged "missing exception clause" in well-grounded answers)
- 1/5 was a genuine but minor accuracy issue
None warranted dropping under our strict-intersection rule.

### 4.3 Reproducibility

All scripts, prompts, intermediate JSONL files, and judge configurations are checked into the repo. Re-running `scripts/01–07` with the same source data and `seed=42` reproduces the exact 300 pairs.

---

## 5. Limitations

1. **Easy-skew**: 55.7% easy, only 4.3% hard. Models near saturation on factual recall will not be discriminated.
2. **Authority imbalance**: heavy representation of State Council regulations; provincial coverage is non-uniform.
3. **Single-source pool**: derived from one HF dataset (`twang2218/chinese-law-and-regulations`); regulatory updates after 2024 are not reflected.
4. **Self-distilled labels**: answers are LLM-generated, not lawyer-written. The jury catches drift but cannot guarantee expert-level precision.
5. **Topic gap**: criminal law and judicial interpretations are out of scope for v0.1.

---

## 6. Comparison with Existing Benchmarks

| Benchmark | Size | Domain | Anchored? | Multi-judge? |
| --- | --- | --- | --- | --- |
| JEC-QA           | 26,365 | Judicial exam   | ❌ | ❌ |
| LawBench         | 11,400 | Mixed legal     | △ | ❌ |
| Lawyer-LLaMA QA  |  4,000 | Lawyer Q&A      | ❌ | ❌ |
| **wh-bench v0.1** |   **300** | **Regulations** | ✅ | ✅ |

wh-bench is **deliberately small** but fills a niche: **per-pair source anchoring + multi-judge verification**, neither of which the larger benchmarks provide.

---

## 7. Future Work

- **v0.2** (≤ 1 month): expand to 1000+, rebalance to ≥30% hard, include judicial interpretations
- **v0.3**: add a third judge (Gemini or Qwen) and adopt majority-vote with abstention
- **v1.0**: build a public leaderboard, integrate with `lm-eval-harness`

---

## 8. Acknowledgments

- Source corpus: [twang2218/chinese-law-and-regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations)
- Compute: ~¥40 in API costs (foxnio gateway for OpenAI + Anthropic)
- Tooling: `hermes-agent` for orchestration

---

## Appendix A — Judge Prompts

See `scripts/llm_judge.py` and `scripts/llm_judge_claude.py`. Key requirements (identical across both judges):

```
评测以下中文法律法规问答对，并给出 4 个维度的 1-5 分评分及最终判定（pass/borderline/fail）。

维度：
- groundedness: 答案是否能从原文中找到直接依据？
- accuracy:     答案中的事实/数字/主体是否准确？
- clarity:      问题表述是否清晰，有无歧义？
- specificity:  答案是否具体可验证，避免空泛？

判定规则：
- pass:       4 个维度均 ≥ 4
- borderline: 任一维度 = 3
- fail:       任一维度 ≤ 2

输出 JSON：{scores, verdict, issues, reason}
```

## Appendix B — Drop List Detail

| ID | Both-judge issue |
| --- | --- |
| wh-bench-0176 | 电力减供数值错误（原文 30%，答案写 50%）|
| wh-bench-0246 | Chunk 截断导致主体列表不完整 |
| wh-bench-0297 | 与 0246 同源，主体清单缺失 |

## Appendix C — Cost & Time

| Stage | Time | API Cost |
| --- | --- | --- |
| Source download | 2 min | — |
| Top-50 scoring | 8 min | ~¥3 |
| Distillation (305 QA) | 4 min | ~¥8 |
| GPT-5.5 judge | 6 min | ~¥10 |
| Claude jury | 12 min | ~¥18 |
| Jury merge + finalize | 1 min | — |
| **Total** | **~33 min** | **~¥40** |
