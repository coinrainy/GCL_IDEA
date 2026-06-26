# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 在普通节点分类的图对比学习中，标准 GRACE/GCA 式预训练默认每个节点都适合强 positive alignment；我们需要验证并利用“哪些节点适合强 alignment，哪些节点不适合”这一节点级差异，而不能把它偷换成 boundary-aware GNN 或 false-negative mining。
- Must-solve bottleneck: 当前 idea 的最大瓶颈是 novelty collision。BCE-GCL 的 boundary 叙事已被 BES/UGCL/ASPECT/HAR 等近邻强烈包围；必须收缩为无监督 contrastive eligibility diagnostic + 最小 objective routing，并证明它不是 degree、uncertainty、augmentation strength 或普通 loss weighting 的代理。
- Non-goals: 不追求 SOTA claim；不做 LLM/KAN/大架构替换；不使用 test label 或 test set 调参；不声称 first boundary-aware GCL；不把 pilot/development 结果写成 formal claim。
- Constraints: 基于现有 GRACE/GCA 代码路线；节点分类主协议为 stratified random `1:1:8`；GCL evaluator 为 frozen encoder + Logistic Regression；首轮只允许 pilot/development 证据；优先 Cora/CiteSeer 3 seeds pilot，再决定是否 10 seeds development。
- Success condition: 如果一个纯无监督 eligibility signal 能预测节点对强 alignment 的受益/受损差异，并且 objective routing 在控制 degree/uncertainty/augmentation strength 与 scalar weighting 后仍有独立增益，则方向可继续；否则 `PIVOT/KILL`。

## Anchor Check

- Original bottleneck: 标准 GCL 的一刀切强 positive alignment 可能不适合所有节点，但 BCE-GCL 的 boundary 叙事 novelty 风险太高。
- Why the revised method still addresses it: 修订版只保留 “alignment eligibility” 问题，并用二路 objective intervention 检验强 alignment 是否应该被替换，而不是重新包装 boundary-aware GNN。
- Reviewer suggestions rejected as drift: 不加入 LLM/VLM/KAN；不把 method 扩展为 adaptive augmentation generator；不转成 pair-level false-negative reliability。

## Simplicity Check

- Dominant contribution after revision: 二路 objective routing：eligible nodes 用完整 InfoNCE；ineligible nodes 从 anchor-level contrastive discrimination 中退出，改用 positive-only stop-gradient consistency。
- Components removed or merged: 删除三路 routing；删除 `stability-only` 第三目标；删除 `s_loss` gate；固定 pilot quantile。
- Reviewer suggestions rejected as unnecessary complexity: 不新增神经 gating network；不引入 prototypes；不做 label-aware calibration。
- Why the remaining mechanism is smallest adequate route: 只改 loss 中每个节点作为 anchor 时接受哪种 objective，不改 view、encoder、evaluator、split 或训练范式。

## Changes Made

### 1. Route space changed from three routes to two routes

- Reviewer said: 三路 objective 没有数学闭合，容易退化为 scalar reweighting。
- Action: 改成 `InfoNCE` vs `positive-only stop-gradient consistency`。
- Reasoning: 二路 objective type 切换更清晰，和 loss weighting 的差异更可检验。
- Impact on core method: 方法更小，pilot 更可证伪。

### 2. Gate changed from proxy stacking to alignment treatment-risk

- Reviewer said: `s_instability + s_fs + s_loss` 像 proxy 拼接。
- Action: 删除 `s_loss` 作为 gate 输入；gate 只用多增强 alignment instability 和 degree-residualized feature-structure agreement。
- Reasoning: `s_loss` 太接近 hard-node weighting；保留为 diagnostic 即可。
- Impact on core method: 更远离 HAR/SHARP 与 uncertainty weighting。

### 3. Delta against prior work moved into method definition

- Reviewer said: 与 BES/UGCL/ASPECT/HAR/SHARP 差异还靠文字解释。
- Action: 在 revised proposal 中明确 CER-GCL 不生成 view、不做 spectral fusion、不做 uncertainty-weighted loss、不做 pair false-negative correction；唯一动作是切换 objective type。
- Reasoning: 若公式上仍可化约为权重，方法应被 kill。
- Impact on core method: 让 route-vs-weighting 成为第一证伪实验。

## Revised Proposal

# Research Proposal: CER-GCL — Contrastive Eligibility Routing for Graph SSL

## Problem Anchor

- Bottom-line problem: 在普通节点分类的图对比学习中，标准 GRACE/GCA 式预训练默认每个节点都适合强 positive alignment；我们需要验证并利用“哪些节点适合强 alignment，哪些节点不适合”这一节点级差异，而不能把它偷换成 boundary-aware GNN 或 false-negative mining。
- Must-solve bottleneck: 当前 idea 的最大瓶颈是 novelty collision。BCE-GCL 的 boundary 叙事已被 BES/UGCL/ASPECT/HAR 等近邻强烈包围；必须收缩为无监督 contrastive eligibility diagnostic + 最小 objective routing，并证明它不是 degree、uncertainty、augmentation strength 或普通 loss weighting 的代理。
- Non-goals: 不追求 SOTA claim；不做 LLM/KAN/大架构替换；不使用 test label 或 test set 调参；不声称 first boundary-aware GCL；不把 pilot/development 结果写成 formal claim。
- Constraints: 基于现有 GRACE/GCA 代码路线；节点分类主协议为 stratified random `1:1:8`；GCL evaluator 为 frozen encoder + Logistic Regression；首轮只允许 pilot/development 证据；优先 Cora/CiteSeer 3 seeds pilot，再决定是否 10 seeds development。
- Success condition: 如果一个纯无监督 eligibility signal 能预测节点对强 alignment 的受益/受损差异，并且 objective routing 在控制 degree/uncertainty/augmentation strength 与 scalar weighting 后仍有独立增益，则方向可继续；否则 `PIVOT/KILL`。

## Technical Gap

已有 GCL 近邻主要改变 view、pair sampling、sample weighting、spectral fusion、degree bias 或 supervised boundary shaping。CER-GCL 只检验一个更窄的干预问题：**当节点作为 contrastive anchor 时，它是否应该接受 full discrimination objective，还是只保留 positive-view consistency？**

该问题和 scalar weighting 不同：scalar weighting 仍保留同一个 InfoNCE objective，只改变强度；CER-GCL 改变 objective type。对于 ineligible anchors，它不再参与 full denominator discrimination，而是用 stop-gradient positive consistency 保留跨视图不变性。

## Method Thesis

- One-sentence thesis: A node's eligibility for strong contrastive discrimination can be estimated from unsupervised alignment treatment-risk signals, and ineligible anchors should switch objective type from InfoNCE to positive-only stop-gradient consistency rather than merely receive a smaller weight.
- Why this is the smallest adequate intervention: 只替换 anchor-level objective，不改 backbone、view generator、negative sampler、evaluator 或数据协议。
- Why this route is timely: 近期工作已显示节点异质性、uncertainty、degree bias、spectral preference 和 boundary vulnerability 都重要；CER-GCL 将这些压力收缩为一个可证伪的 objective-intervention question。

## Contribution Focus

- Dominant contribution: Contrastive eligibility as an anchor-level objective intervention problem in Graph SSL。
- Optional supporting contribution: A falsification protocol proving whether routing is more than degree/uncertainty/augmentation proxy or scalar weighting。
- Explicit non-contributions: boundary-aware GNN、adaptive augmentation、pair false-negative correction、spectral policy、uncertainty-weighted loss、new encoder。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: GRACE/GCA-style encoder, standard stochastic views, existing split/evaluator。
- New trainable components: none。
- Tempting additions intentionally not used: neural gate、prototype、label-aware calibration、LLM semantic views。

### System Overview

```text
warmup GRACE
  -> collect multi-view node statistics
  -> compute treatment-risk score q_i
  -> fixed quantile gate m_i
       m_i = 1: full InfoNCE anchor
       m_i = 0: positive-only stop-gradient consistency anchor
  -> train encoder with mixed objective
  -> frozen Logistic Regression evaluation
```

### Core Mechanism

Let `z_i^a` and `z_i^b` be normalized embeddings for node `i` under two views. Let `sg(.)` denote stop-gradient.

For eligible anchors:

```text
L_i^NCE = -log exp(sim(z_i^a, z_i^b)/tau) /
          sum_j exp(sim(z_i^a, z_j^b)/tau)
```

For ineligible anchors:

```text
L_i^PO = 1 - sim(z_i^a, sg(z_i^b))
```

Mixed objective:

```text
L = mean_i [ m_i * L_i^NCE + (1 - m_i) * L_i^PO ]
```

Important distinction from scalar weighting:

- Scalar weighting baseline: `L = mean_i w_i * L_i^NCE`。
- CER-GCL: ineligible anchors no longer optimize denominator discrimination as anchors; they optimize a different positive-only objective。

Eligibility gate:

```text
q_i = rank_norm(drift_i) + rank_norm(fs_residual_i)
m_i = 1[q_i <= quantile(q, rho)]
```

where:

- `drift_i`: warmup-period multi-view positive drift, averaged over repeated standard augmentations。
- `fs_residual_i`: feature-structure disagreement residualized against degree, computed from feature-neighbor agreement minus structure-neighbor expectation。
- `rho`: fixed pilot quantile, e.g. `0.8`; no pilot threshold search。

`s_loss` / node hardness is not used in the gate. It is logged only as diagnostic to test whether CER-GCL collapses into hard-node weighting.

### Training Plan

1. Warmup: train vanilla GRACE for `w` epochs and collect EMA statistics。
2. Gate freeze: compute fixed route mask `m_i` from `q_i` using fixed quantile。
3. Mixed training: continue training with `L_i^NCE` for eligible anchors and `L_i^PO` for ineligible anchors。
4. Evaluation: freeze encoder and run Logistic Regression selected by train/val only。

### Failure Modes and Diagnostics

- Routing equals weighting:
  - Test: same `q_i` scalar weighting baseline。
  - Stop rule: if scalar weighting matches route on bucket diagnostics and accuracy, kill routing claim。
- Gate equals degree/uncertainty:
  - Test: degree-only, uncertainty-only, residualized gate。
  - Stop rule: if residual gate has no predictive value, kill method。
- Positive-only consistency causes collapse:
  - Test: embedding rank/variance and per-route representation norms。
  - Stop rule: if rank/variance collapse appears in smoke, kill or add collapse guard only after documenting it。

## Novelty and Elegance Argument

CER-GCL is not boundary-aware GNN, adaptive augmentation, uncertainty weighting, spectral fusion, or false-negative correction. It is a minimal intervention on the anchor objective. The paper lives or dies on whether changing objective type for ineligible anchors produces evidence that scalar weighting cannot explain.

## Claim-Driven Validation Sketch

### Claim 1: Treatment-risk eligibility is predictive beyond known proxies

- Minimal experiment: Cora/CiteSeer 3 seeds pilot；compute q_i and post-hoc alignment benefit / correctness change。
- Baselines / ablations: degree-only、uncertainty-only、drift-only、fs-residual-only、random gate。
- Metric: route bucket accuracy delta、alignment drift delta、Spearman correlation with post-hoc benefit。
- Expected evidence: q_i retains residual predictive power after degree/uncertainty controls。

### Claim 2: Objective routing is not scalar weighting

- Minimal experiment: vanilla GRACE vs global weak pressure vs same-score scalar weighting vs CER-GCL two-route objective。
- Metrics: frozen Logistic Regression, bucket-wise diagnostics, embedding rank/variance。
- Expected evidence: CER-GCL improves targeted bucket behavior where scalar weighting fails；if only overall accuracy changes, claim remains weak。

## Experiment Handoff Inputs

- Must-prove claims:
  - C1: treatment-risk q_i predicts alignment suitability beyond proxies。
  - C2: objective type switching is not reducible to scalar weighting。
- Must-run ablations:
  - random gate、degree-only、uncertainty-only、drift-only、fs-residual-only。
  - same-score scalar weighting、global weak pressure、vanilla GRACE。
- Critical datasets / metrics:
  - Cora/CiteSeer seeds 0-2 pilot first。
  - frozen encoder + Logistic Regression。
  - route bucket diagnostics and collapse checks。
- Highest-risk assumptions:
  - q_i has residual predictive value。
  - positive-only objective does not collapse and differs from downweighting。

## Compute & Timeline Estimate

- Estimated GPU-hours: smoke < 1 GPU-hour；pilot 4-8 GPU-hours。
- Data / annotation cost: none。
- Timeline: one implementation day plus one pilot day if GRACE script remains stable。

