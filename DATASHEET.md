# Datasheet — wh-bench v0.1

> Datasheet for the **wh-bench v0.1** Chinese legal-regulation QA benchmark.
> Format adapted from Gebru et al. 2018, *"Datasheets for Datasets"*.

---

## 1. Motivation

### 1.1 For what purpose was the dataset created?

为评估中文 LLM 在**法律法规知识**上的事实准确性而构建。当前中文法律 QA 评测多偏向司法考试题或律师函风格，缺少**直接对照法规原文条款**的轻量评测集。wh-bench 设计为：

- 每条 QA 都带原文引用（`source_quote`），评分时可对照原文判定
- 规模小（300 条）但**字段完整**、**可追溯**、**有双评委验证**
- 适用于：closed-book 知识评测、open-book RAG 评测、SFT 数据冷启动

### 1.2 Who created the dataset?

由独立开发者 `teee32` 在 2026 年 5 月构建，作为武汉「AI+OPC（一人公司）」政策风口下的数据集开源实践。

### 1.3 Who funded the dataset?

无机构资助。API 成本（约 ¥40）由作者自付。

---

## 2. Composition

### 2.1 What do the instances represent?

每个实例为一条**法规知识问答对**：

- `question`: 自然语言问题（中文）
- `answer`: 简明答案（中文，可直接引用或紧密改写自原文）
- `source_text`: 原始法规文本 chunk（含上下文）
- `source_quote`: 答案最直接对应的原文锚点（1-3 句）
- `source_doc`: 法规标题
- `source_section`: 章节范围
- `category`: 内容分类（蒸馏阶段自动生成）
- `difficulty`: easy / medium / hard
- `authority`: 发文机关
- `jury`: 双评委评分与决定（详见下文）

### 2.2 How many instances are there?

**300 条**（train 239 + test 61），按 difficulty 分层 80/20 划分。

### 2.3 Does the dataset contain all possible instances or is it a sample?

是从 **22,552 篇** 中文法规中抽样蒸馏得到，覆盖 **50 部** 法规文档。抽样方式：

1. 关键词命中（含「条例/办法/规定/法/规范」）→ 命中 852 篇
2. LLM 评分排序 Top-50（评分基于条款密度、可问性、与目标场景相关度）
3. 每部蒸馏 6.1 条 QA（平均），共 305 条
4. 规则过滤 → 303 条
5. 双 jury 剔除 3 条双 fail → **最终 300 条**

### 2.4 Is there any missing information?

- `answer` 字段为 LLM 蒸馏生成，可能与原文存在**轻微改写**，已由双评委的 `groundedness` 维度评分把关
- 15 条 CONTESTED 样本保留了**单方评委 fail 的存疑判断**（详见 `jury` 字段）

### 2.5 Does the dataset contain confidential / personal information?

否。源数据全部为**公开发布的中国法律法规**（《著作权法》第五条规定不受著作权保护）。

### 2.6 Does the dataset contain offensive content?

否。所有内容为法律条文，无攻击性表述。

---

## 3. Collection Process

### 3.1 How was the data acquired?

| 阶段 | 工具 / 模型 | 输入 | 输出 | 数量 |
| --- | --- | --- | --- | --- |
| 数据源 | HuggingFace `twang2218/chinese-law-and-regulations` | — | raw markdown | 22,552 |
| 关键词命中 | regex / `01_load_sources.py` | raw | candidates | 852 |
| LLM 评分排序 | `gpt-5.4-mini` | candidates | top-50 | 50 |
| 蒸馏 QA | `gpt-5.4-mini`，prompt 强制原文引用 | top-50 + chunk | raw QA | 305 |
| 规则过滤 | `03_filter.py` 字段完整性 / 引用可达性 / 长度 | raw QA | filtered | 303 |
| GPT 评审 | `gpt-5.5` 四维评分 + verdict | filtered | judged_gpt | 303 |
| Claude 评审 | `claude-sonnet-4-6` 四维评分 + verdict | filtered | judged_claude | 303 |
| Jury 合并 | `jury_decide.py` 双 fail 才剔 | both | jury_verdicts | 303 |
| 最终化 | `07_finalize.py` 剔除 + split | jury | **v0.1** | **300** |

### 3.2 What was the sampling strategy?

非概率抽样。LLM 评分 Top-50，保证覆盖**高质量、条款密集、有评测价值**的法规。本版本明确**不覆盖** **司法解释** 与 **行政规章**，仅含**法律 + 行政法规 + 地方性法规**。

### 3.3 Time frame

2026-05-27 一日内完成全流水线（约 4 小时蒸馏 + 30 分钟双 jury 评审）。

### 3.4 Were any ethical review processes conducted?

无机构 IRB，但作者执行了以下质量保证：

- **双 LLM 独立评审**（不同模型族系，避免单一偏见）
- **保留 CONTESTED 样本**而非简单剔除，供研究者审视争议判断
- **每条都带原文引用**，下游评测者可独立验证
- **公开全部失败案例**（drop list、jury reasons），便于复现与质询

---

## 4. Preprocessing / Cleaning / Labeling

### 4.1 Was the data preprocessed?

是。具体步骤：

1. **Chunking**：长法规按章节切分，每 chunk ≤ 3000 字
2. **字段必填**：question/answer/source_text/source_quote 任一为空即丢弃
3. **引用可达性**：`source_quote` 必须能在 `source_text` 中匹配到（fuzzy match）
4. **去重**：相同 `(question, source_doc)` 对仅保留一条
5. **长度阈值**：question 5-200 字，answer 10-500 字

### 4.2 Was the "raw" data saved?

是。所有中间产物均在 git 仓库内：

- `data/raw/sources.jsonl` — 命中候选 852 篇
- `data/distilled/qa_raw.jsonl` — 蒸馏产出 305 条
- `data/filtered/qa_v0.1_auto.jsonl` — 过滤后 303 条
- `data/judged/qa_judged.jsonl` — GPT 评审结果
- `data/judged/qa_judged_claude.jsonl` — Claude 评审结果
- `data/judged/jury_verdicts.jsonl` — Jury 合并结果
- `data/judged/jury_drop_list.txt` — 剔除清单

---

## 5. Uses

### 5.1 What tasks could the dataset be used for?

- **Closed-book QA 评测**：测查模型对中国法规条款的事实回忆能力
- **Open-book RAG 评测**：把 `source_text` 当作检索上下文，评测 grounding 能力
- **SFT / DPO 数据冷启动**：300 条带原文引用的高质量种子
- **评测方法学研究**：jury 的 CONTESTED 15 条可作为「评委不一致」的研究样本

### 5.2 Are there tasks the dataset should not be used for?

- ❌ **法律咨询替代**：本数据集不提供个案咨询能力评测
- ❌ **司法判决预测**：不含案例事实与判决
- ❌ **跨语言评测**：仅中文，不含翻译对
- ❌ **推理能力主基准**：难度偏 easy（167/300），适合作为**事实知识**而非**推理能力**评测

---

## 6. Distribution

### 6.1 How will the dataset be distributed?

- HuggingFace Datasets：`teee32/wh-bench-v0.1`
- GitHub：`teee32/wh-bench-v0.1`（源码 + 中间产物）

### 6.2 License

- 数据集：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 构建脚本：[MIT](./LICENSE)
- 源法规文本：公有领域（中国《著作权法》第五条）

---

## 7. Maintenance

### 7.1 Who is supporting / hosting / maintaining the dataset?

`teee32`（独立开发者）。Issue 与 PR 通过 [GitHub](https://github.com/teee32/wh-bench-v0.1/issues) 接收。

### 7.2 Will the dataset be updated?

计划路线：

- **v0.2**（≤1 个月）：扩到 1000+ 条，增加 hard 难度比例（目标 hard ≥ 30%），加入司法解释
- **v0.3**：引入第三位评委（Gemini / Qwen），多轮 jury 投票
- **v1.0**：覆盖 30 部以上一线常用法律，建立 leaderboard

### 7.3 Will older versions continue to be supported?

是。每个版本以独立 tag 存档于 GitHub 与 HuggingFace，永不撤回。

### 7.4 If others want to contribute, how?

GitHub issue / PR。重点欢迎：

- 新法规候选提名
- CONTESTED 样本的第三方人工裁定
- 评委 prompt 改进与新模型评委加入
