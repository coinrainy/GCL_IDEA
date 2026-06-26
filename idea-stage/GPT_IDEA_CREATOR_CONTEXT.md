# GPT Idea Creator Context

生成时间：2026-06-26  
阶段决策：`READY_FOR_GPT_IDEA_CREATOR`

## GPT 下一步应读取的文件

1. `literature/LIT_REVIEW_GCL_CORE.md`
2. `literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`
3. `literature/GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
4. `AGENTS.md`
5. `research-wiki/log.md`
6. `research-wiki/experiments/grace_baseline_1_1_8.md`
7. `research-wiki/experiments/gcn_baseline_1_1_8.md`
8. `research-wiki/experiments/gat_baseline_1_1_8.md`

## 项目边界

- 研究主题：图对比学习用于普通节点分类。
- 主协议：stratified random `1:1:8`，即 train/validation/test = 10%/10%/80%。
- 默认 seeds：0-9。
- GCL 下游评估：frozen encoder + Logistic Regression，使用 train/val 或 train 内部 CV 选择 evaluator 超参，禁止 test 调参。
- 当前阶段只允许生成 idea 和实验计划，不允许声称 SOTA 或 robust。
- smoke、pilot、development 结果都不能进入论文性能 claim。

## 本次文献数量

- GCL / Graph SSL 直接相关论文：21 篇。
- 外部可迁移方法论文：18 篇。

## 最重要的 5 个 gap

1. **语义保持视图缺失**：GRACE/GCA 式随机或中心性增强不保证保留节点类别语义。
2. **false negative 与 hard negative 混淆**：同类或语义相近节点常被当作负样本推远。
3. **instance discrimination 与 class-level node classification 不一致**：实例均匀化不等于类别边界清晰。
4. **masked modeling 与 contrastive learning 缺少轻量统一**：重构学局部上下文，对比学判别结构，但现有组合常复杂。
5. **协议不一致导致文献 claim 难迁移**：public split、random split、不同 evaluator 与 test 调参会污染比较。

## 推荐 GPT 生成方向

优先生成 3-5 个轻量、可证伪、可在当前仓库实现的 idea。推荐排序：

1. False-negative-aware GRACE/GCA loss weighting。
2. Prototype-guided positive expansion。
3. Feature-space semantic augmentation。
4. Masked-contrastive node pretraining。
5. Degree/class-aware hardness reweighting。

## GPT 输出格式建议

每个 idea 至少包含：

- 名称；
- 解决的 gap ID；
- 核心假设；
- 方法机制；
- 与 GRACE/GCA/BGRL/CCA-SSG/GraphMAE 的区别；
- 最小实现改动；
- 最小验证实验；
- 失败判据；
- 风险；
- 阶段决策建议：`GO` / `REVISE` / `PIVOT` / `KILL` / `BLOCKED`。

## 严禁 GPT 下一步做的事

- 不要运行 `/idea-discovery`。
- 不要实现代码。
- 不要跑 pilot 或 GPU 实验。
- 不要把文献中的 SOTA 数字当成本仓库 claim。
- 不要使用 test set 选择 evaluator 或方法超参。
