# wh-bench 在线评测台 (GitHub Pages)

交互式中文法律法规问答评测 demo，把 wh-bench v0.1 数据集变成「看得见、摸得着」的可用 AI 产品。

- **在线访问**：https://teee32.github.io/wh-bench-v0.1/
- 纯静态单页（`index.html` + `data.json`），零依赖、零后端
- 玩法：选难度/题数 → 系统出题 → 你或你的大模型作答 → 对照标准答案与法条出处 → 自评打分

数据来自 `../data/processed/wh_bench_v0.1_all.jsonl`，由 `../scripts/build_demo_data.py` 生成（如有）。
