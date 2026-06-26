# Gap Map

_Field gaps with stable IDs._

## 2026-06-26 /research-lit: GCL Node Classification

Decision: `READY_FOR_GPT_IDEA_CREATOR`

Canonical files:

- `literature/LIT_REVIEW_GCL_CORE.md`
- `literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`
- `literature/GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
- `idea-stage/GPT_IDEA_CREATOR_CONTEXT.md`

Key gaps:

- `G1`: 视图增强缺少类别语义保证。
- `G2`: false negative / hard negative 定义混乱。
- `G3`: instance-level 对比与 class-level 节点分类目标错位。
- `G4`: masked modeling 与 contrastive objective 未形成简洁统一。
- `G5`: public split、random split、不同 evaluator 与 test 调参导致文献 claim 不可直接迁移。
- `G6`: degree / class imbalance / heterophily 弱群体被平均结果掩盖。
- `G7`: LLM/TAG 方法与普通图节点分类之间存在语义与复现鸿沟。
- `G8`: 复杂模块可能引入不公平调参空间和计算预算。

Recommended idea directions:

- False-negative-aware GRACE/GCA loss weighting.
- Prototype-guided positive expansion.
- Feature-space semantic augmentation.
- Masked-contrastive node pretraining.
- Degree/class-aware hardness reweighting.
