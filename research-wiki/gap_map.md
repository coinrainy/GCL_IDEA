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

## 2026-06-26 /research-lit: False Negatives in GCL

Decision: `READY_FOR_FALSE_NEGATIVE_IDEA_CREATOR`

Canonical files:

- `literature/LIT_REVIEW_GCL_FALSE_NEGATIVES.md`
- `literature/FALSE_NEGATIVE_GAP_MAP_FOR_GPT_IDEA_CREATOR.md`

Focused gaps:

- `FN-G1`: hard negative 与 false negative 没有分离。
- `FN-G2`: 负样本只降权，正样本不扩展。
- `FN-G3`: 训练早期 pseudo semantics 不可靠。
- `FN-G4`: 图结构相似度在异配/噪声图上失效。
- `FN-G5`: clustering/prototype 容易吞掉少数类。
- `FN-G6`: 生成式 hard negatives 复杂且可能泄露语义。
- `FN-G7`: 少标签信息边界模糊。
- `FN-G8`: 缺少统一 `1:1:8` + logreg_val 协议验证。

Recommended false-negative directions:

- Reliability-weighted InfoNCE.
- Curriculum false-negative filter.
- Joint positive expansion and negative debias.
- Reliability-gated hard true negative mining.
- Train/Val-calibrated false-negative debias.

## 2026-06-26 /idea-creator: False-Negative Ideas Beyond IGT-GCL

Decision: `GO_FOR_TOP2_REFINEMENT`

Canonical files:

- `idea-stage/FN_IDEAS_BEYOND_IGT.md`
- `idea-stage/FN_IDEAS_BEYOND_IGT_20260626_092155.md`
- `research-wiki/ideas/dsr_gcl.md`
- `research-wiki/ideas/cng_gcl.md`

Updated constraint:

- Treat `IGT-GCL` as a low-novelty counterexample. Avoid point posterior, interval targets, abstention, and simple sample reweighting variants as primary contributions.

Top recommendations:

- `DSR-GCL`: decouple class-semantic representation from residual instance-uniformity so false-negative repulsion does not directly shape the downstream semantic subspace. Targets `G2`, `G3`, `FN-G1`, `FN-G2`.
- `CNG-GCL`: select negatives through an anchor-local conflict graph rather than pair posterior or weight estimation. Targets `FN-G1`, `FN-G3`, `FN-G4`.

Backlog directions:

- masked positive completion;
- balanced optimal-transport assignment;
- gradient orthogonalization;
- semantic-shell counterfactual negatives;
- environment-invariant positive learning;
- frequency-diversified positives;
- mutual-information budgeted contrast;
- honest train/val-calibrated semi-supervised constraints.
