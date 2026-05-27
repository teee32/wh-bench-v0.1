# wh-bench v0.1 — 中文法律法规问答评测集

> A small, high-quality, **dual-LLM-jury verified** Chinese legal-regulation QA benchmark for evaluating instruction-tuned LLMs on Chinese law knowledge.

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/teee32/wh-bench-v0.1)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/teee32/wh-bench-v0.1)

## 概览

**wh-bench v0.1** 是一份 300 条中文法律法规问答评测集，由公开法规文本经 LLM 蒸馏 + 规则过滤 + **双 LLM 评委独立评审（GPT-5.5 + Claude Sonnet）** 三阶段构建。每条 QA 都带原文引用（`source_quote` + `source_text`），可追溯到 50 部国家级与地方性法规。

设计目标：用最小开销给小规模中文 LLM 提供一个 **可信、可复现、字段完整** 的法律领域 QA 评测起点。

| Split | 数量 | 难度分布（easy/medium/hard） |
| --- | --- | --- |
| train | 239 | 134 / 95 / 10 |
| test  |  61 |  33 / 25 /  3 |
| **all**   | **300** | **167 / 120 / 13** |

- **源文档**：50 部（覆盖国务院条例、人大常委会法律、地方性法规）
- **发文机关**：14 个层级（中央 + 14 省/市人大常委会）
- **双评委一致 pass**：67.0%（保守口径）
- **双评委一致 fail（已剔除）**：3 条
- **争议条目（CONTESTED，单方 fail）**：15 条（保留并在 `jury.contested=true` 标记）

## 字段

```json
{
  "id": "wh-bench-0001",
  "question": "哪些船舶在内河航行前必须向引航机构申请引航？",
  "answer": "外国籍船舶、1000总吨以上的海上机动船舶、通航条件受限制的船舶...",
  "source_doc": "中华人民共和国内河交通安全管理条例",
  "source_url": "https://huggingface.co/datasets/twang2218/chinese-law-and-regulations",
  "source_section": "第一条—第九十五条",
  "source_text": "<原始法规全文 chunk>",
  "source_quote": "下列船舶在内河航行，应当向引航机构申请引航：",
  "category": "航行管理",
  "difficulty": "medium",
  "authority": "国务院",
  "review_status": "jury_keep",
  "jury": {
    "decision": "KEEP",
    "gpt_verdict": "pass",
    "claude_verdict": "pass",
    "contested": false,
    "gpt_scores":   {"groundedness": 5, "accuracy": 5, "clarity": 5, "specificity": 5},
    "claude_scores":{"groundedness": 5, "accuracy": 5, "clarity": 5, "specificity": 5},
    "gpt_reason":    "...",
    "claude_reason": "..."
  }
}
```

## 快速开始

```python
from datasets import load_dataset

ds = load_dataset("teee32/wh-bench-v0.1")
print(ds)
# DatasetDict({
#   train: Dataset({features: [...], num_rows: 239})
#   test:  Dataset({features: [...], num_rows:  61})
# })

# closed-book 评测：把 question 喂给模型，对照 answer 评分
sample = ds["test"][0]
print(sample["question"])
print(sample["answer"])
print(sample["source_quote"])  # 原文锚点，便于人审

# open-book 评测：把 source_text 当作上下文
prompt = f"参考下文回答问题。\n\n参考资料：\n{sample['source_text'][:2000]}\n\n问题：{sample['question']}"
```

## 双 LLM 评委（Dual-Jury）方法

每条 QA 通过以下三阶段筛选：

1. **蒸馏阶段** — `gpt-5.4-mini` 基于源文档 chunk 生成 QA，要求严格引用原文
2. **规则过滤** — 字段完整性、引用可达性、长度/重复检测
3. **双评委独立评审** — `gpt-5.5` 和 `claude-sonnet-4-6` 分别按四维评分（groundedness / accuracy / clarity / specificity）

**剔除口径（口径 A，严交集）**：仅当 **两位评委同时判定 fail** 才剔除。

| | Claude pass | Claude borderline | Claude fail |
| --- | --- | --- | --- |
| **GPT pass**       | 203 | 72 | 14 |
| **GPT borderline** |   2 |  8 |  1 |
| **GPT fail**       |   0 |  0 |  3 |

- 双 pass 一致率：**67.0%**
- 单方 fail（CONTESTED）：**15 条**，已在数据集中保留并打标，便于研究模型对争议样本的处理
- 双 fail：**3 条**（`wh-bench-0176/0246/0297`），已从最终发布版本中剔除

更详细的方法论与统计：[TECH_REPORT.md](./TECH_REPORT.md)、[DATASHEET.md](./DATASHEET.md)

## 已知限制

- **CONTESTED 15 条** 被保留，主要原因是单方评委严苛而另一方判 pass 或 borderline；使用者若追求最严格质量可结合 `jury.contested` 字段过滤
- **难度分布偏 easy**（167/120/13）：本版本主要测查事实回忆与条款定位，不适合作为推理能力评测的唯一基准
- **机关分布不均**：国务院 156 条最多；地方人大 100+ 条主要来自 8 个省市，未全省覆盖
- **答案为蒸馏生成**：人工抽检 + 双 LLM 评审 + 原文引用三重锁定，但仍可能存在边缘表述偏差

## 复现

```bash
git clone https://github.com/teee32/wh-bench-v0.1.git
cd wh-bench-v0.1
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # 填入 OPENAI_API_KEY（兼容 OpenAI 格式）

# 完整流水线
python scripts/01_load_sources.py    # 拉取源数据
python scripts/02_distill.py         # LLM 蒸馏 QA
python scripts/03_filter.py          # 规则过滤
python scripts/llm_judge.py          # GPT-5.5 评审
python scripts/llm_judge_claude.py   # Claude Sonnet 评审
python scripts/jury_decide.py        # 合并 jury 决定
python scripts/07_finalize.py        # 剔除 + split + 落地
```

## 引用

```bibtex
@dataset{teee32_wh_bench_2026,
  title  = {wh-bench v0.1: A Dual-Jury Verified Chinese Legal-Regulation QA Benchmark},
  author = {teee32},
  year   = {2026},
  url    = {https://huggingface.co/datasets/teee32/wh-bench-v0.1},
  note   = {300 QA pairs across 50 Chinese laws and regulations, verified by GPT-5.5 + Claude Sonnet jury}
}
```

## 数据源 & 致谢

- 源法规文本：[twang2218/chinese-law-and-regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations)（22,552 篇）
- 蒸馏模型：OpenAI gpt-5.4-mini
- 评审模型：OpenAI gpt-5.5 + Anthropic Claude Sonnet 4-6

## License

数据集本身采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，构建脚本采用 [MIT](./LICENSE)。

源法规文本属于公有领域（中国法律法规不受著作权保护，《著作权法》第五条）。
