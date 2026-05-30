# whbench-eval — wh-bench 中文法律法规大模型自动评测工具

一个面向开发者的**模型工具（model tool）**：接入任意 OpenAI 兼容的大模型 API，自动在 wh-bench v0.1（300 条中文法律法规 QA）上跑完整评测、自动判分、产出结构化评测报告。

> 用一句话说清它是什么：**给大模型测「中国法律知识」能力的自动化评测工具**，支撑垂直行业（法律）AI 模型的开发、选型与优化全流程。

## 它解决什么问题

做法律方向 AI 应用的开发者，需要回答「我该用哪个大模型？它的中文法律知识到底行不行？」——whbench-eval 让你一条命令拿到答案：自动让被测模型回答 300 道法规题，自动判分，输出分难度、分领域的能力画像。

## 安装

```bash
pip install openai            # 唯一硬依赖
# 或用项目自带 .venv
```

## 快速开始

```bash
# 1) 配置被测模型（任意 OpenAI 兼容端点）
export EVAL_API_KEY=sk-xxxxx
export EVAL_BASE_URL=https://api.openai.com/v1
export EVAL_MODEL=gpt-4o-mini

# 2) 跑评测
python tools/whbench_eval.py --model gpt-4o-mini            # 全 300 题
python tools/whbench_eval.py --max 50                       # 快速试 50 题
python tools/whbench_eval.py --diff hard                    # 只测困难题
python tools/whbench_eval.py --scoring llm --judge-model gpt-4o   # 用 LLM 裁判判分
```

## 判分模式

| 模式 | 说明 | 速度 | 成本 |
| --- | --- | --- | --- |
| `char_f1`（默认） | 字符级 F1，被测答案与标准答案的重合度 | 快 | 仅被测模型调用 |
| `llm` | 用一个更强的「裁判模型」判定语义一致性（1 / 0.5 / 0） | 慢 | 额外裁判模型调用 |

## 产出

每次评测产出两份报告到 `reports/`：

- `eval_<model>_<时间戳>.json` — 机器可读：每题明细（问题/标准答案/模型回答/得分）+ 汇总指标
- `eval_<model>_<时间戳>.md` — 人类可读：总分、分难度、分领域、失分样本

示例汇总：

```
## 总分：69.6%（5 题）
### 分难度
| 难度 | 得分率 | 题数 |
| easy   | 72.1% | 3 |
| medium | 65.8% | 2 |
```

## 环境变量

| 变量 | 用途 |
| --- | --- |
| `EVAL_API_KEY` / `EVAL_BASE_URL` / `EVAL_MODEL` | 被测模型 |
| `JUDGE_API_KEY` / `JUDGE_BASE_URL` / `JUDGE_MODEL` | 裁判模型（`--scoring llm` 时） |

## 自检

```bash
python tools/whbench_eval.py --dry-run    # 只加载数据、校验判分器，不调模型
```

## 许可

工具代码 MIT；wh-bench 数据集 CC BY-SA 4.0。
