---
license: cc-by-sa-4.0
language:
- zh
task_categories:
- question-answering
- text-classification
tags:
- legal
- chinese-law
- regulations
- qa
- benchmark
- evaluation
- llm-as-judge
size_categories:
- n<1K
pretty_name: "wh-bench v0.1: Chinese Legal-Regulation QA Benchmark"
configs:
- config_name: default
  data_files:
  - split: train
    path: wh_bench_v0.1_train.parquet
  - split: test
    path: wh_bench_v0.1_test.parquet
---

# wh-bench v0.1 — 中文法律法规问答评测集

> A small, high-quality, **dual-LLM-jury verified** Chinese legal-regulation QA benchmark for evaluating instruction-tuned LLMs on Chinese law knowledge.

📦 **Code & full reports**: [github.com/teee32/wh-bench-v0.1](https://github.com/teee32/wh-bench-v0.1)
📊 **Tech report**: [TECH_REPORT.md](https://github.com/teee32/wh-bench-v0.1/blob/main/TECH_REPORT.md)
📋 **Datasheet**: [DATASHEET.md](https://github.com/teee32/wh-bench-v0.1/blob/main/DATASHEET.md)

## TL;DR

300 条中文法律法规问答评测集，由公开法规文本经 LLM 蒸馏 + 规则过滤 + **双 LLM 评委独立评审（GPT-5.5 + Claude Sonnet）** 三阶段构建。每条 QA 都带原文引用，可追溯到 50 部国家级与地方性法规。

| Split | Count | easy / medium / hard |
| --- | --- | --- |
| train | 239 | 134 / 95 / 10 |
| test  |  61 |  33 / 25 /  3 |
| **all** | **300** | **167 / 120 / 13** |

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("kksk2312/wh-bench-v0.1")
print(ds)
# DatasetDict({
#   train: Dataset({features: [...], num_rows: 239})
#   test:  Dataset({features: [...], num_rows:  61})
# })

sample = ds["test"][0]
print(sample["question"])     # 哪些船舶在内河航行前必须向引航机构申请引航？
print(sample["answer"])       # 外国籍船舶、1000总吨以上的海上机动船舶...
print(sample["source_quote"]) # 下列船舶在内河航行，应当向引航机构申请引航：
```

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | unique ID, e.g. `wh-bench-0001` |
| `question` | string | Chinese factual question |
| `answer` | string | concise answer derivable from source |
| `source_doc` | string | regulation title |
| `source_url` | string | upstream HF dataset URL |
| `source_section` | string | section range |
| `source_text` | string | full regulation chunk (context) |
| `source_quote` | string | verbatim quote anchoring the answer |
| `category` | string | content category |
| `difficulty` | string | easy / medium / hard |
| `authority` | string | issuing authority |
| `review_status` | string | always `jury_keep` for v0.1 |
| `jury.decision` | string | KEEP / DROP (all kept here) |
| `jury.gpt_verdict` | string | pass / borderline / fail (GPT-5.5) |
| `jury.claude_verdict` | string | pass / borderline / fail (Claude Sonnet) |
| `jury.contested` | bool | true if judges disagree (15 pairs) |
| `jury.gpt_scores.*` | int | 1-5 scores: groundedness/accuracy/clarity/specificity |
| `jury.claude_scores.*` | int | 1-5 scores from Claude |
| `jury.gpt_reason` | string | judge's free-form reasoning |
| `jury.claude_reason` | string | judge's free-form reasoning |

## Dual-LLM Jury Method

**Construction pipeline:**

1. **Distill**: `gpt-5.4-mini` generates QA from chunked regulations, prompt-enforced source quoting
2. **Filter**: rule-based field/quote/length checks → 303 pairs
3. **Dual jury**: `gpt-5.5` and `claude-sonnet-4-6` independently score on 4 dimensions (groundedness, accuracy, clarity, specificity) and emit `pass/borderline/fail`
4. **Drop strategy A (strict intersection)**: drop iff *both* judges fail → 3 dropped, 300 kept

**Inter-judge confusion matrix (303 pairs):**

| | Claude pass | Claude borderline | Claude fail |
| --- | --- | --- | --- |
| **GPT pass**       | **203** | 72 | 14 |
| **GPT borderline** |   2 |  8 |  1 |
| **GPT fail**       |   0 |  0 | **3** |

- Both-pass: **67.0%** (conservative quality ceiling)
- Both-fail (dropped): **1.0%** (3 pairs)
- CONTESTED (one fail, the other not): **15 pairs** — kept with `jury.contested=true`

## Limitations

- **Easy-skew**: 55.7% easy, 4.3% hard. Better suited to factual recall than reasoning evaluation.
- **15 CONTESTED samples** retained: filter via `jury.contested == True` for stricter use.
- **Authority imbalance**: State Council regulations dominate; provincial coverage non-uniform.
- **Self-distilled labels**: LLM-generated, not lawyer-written. Triple-locked by quoting + jury, but not expert-grade.
- **Topic gap**: criminal law and judicial interpretations out of scope for v0.1.

## License

- Dataset: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Code: [MIT](https://github.com/teee32/wh-bench-v0.1/blob/main/LICENSE)
- Source regulations: public domain (PRC Copyright Law Art. 5)

## Citation

```bibtex
@dataset{teee32_wh_bench_2026,
  title  = {wh-bench v0.1: A Dual-Jury Verified Chinese Legal-Regulation QA Benchmark},
  author = {teee32},
  year   = {2026},
  url    = {https://huggingface.co/datasets/kksk2312/wh-bench-v0.1},
  note   = {300 QA pairs across 50 Chinese laws and regulations,
            verified by GPT-5.5 + Claude Sonnet jury}
}
```

## Acknowledgments

- Source corpus: [twang2218/chinese-law-and-regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations)
- Constructed in May 2026 as an open-source data-pipeline experiment under Wuhan's "AI + One-Person-Company" policy initiative.
