# Query Pack

_Auto-generated for /idea-creator. Max 8000 chars._

## Current Idea Context: GCL Node Classification

Decision: `READY_FOR_GPT_IDEA_CREATOR`

Read first:

- `idea-stage/GPT_IDEA_CREATOR_CONTEXT.md`
- `literature/GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
- `literature/LIT_REVIEW_GCL_CORE.md`
- `literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`

Project constraints:

- 主任务是普通节点分类，主线是图对比学习。
- 主协议为 stratified random `1:1:8`；默认 seeds 0-9。
- GCL evaluator 使用 frozen encoder + Logistic Regression；evaluator 超参只能由 train/val 或 train 内部 CV 决定。
- smoke/pilot/development 结果不能产生 SOTA、robust、comprehensive 等性能 claim。

Top gaps:

- 语义保持视图缺失。
- false negative 与 hard negative 混淆。
- instance discrimination 与 class-level node classification 不一致。
- masked modeling 与 contrastive learning 缺少轻量统一。
- 协议不一致导致文献 claim 难迁移。

Promising directions:

- False-negative-aware GRACE/GCA loss weighting.
- Prototype-guided positive expansion.
- Feature-space semantic augmentation.
- Masked-contrastive node pretraining.
- Degree/class-aware hardness reweighting.
