# Research Proposal: CER-GCL — Contrastive Eligibility Routing for Graph SSL

## Problem Anchor

- Bottom-line problem: 在普通节点分类的图对比学习中，标准 GRACE/GCA 式预训练默认每个节点都适合强 positive alignment；我们需要验证并利用“哪些节点适合强 alignment，哪些节点不适合”这一节点级差异，而不能把它偷换成 boundary-aware GNN 或 false-negative mining。
- Must-solve bottleneck: 当前 idea 的最大瓶颈是 novelty collision。BCE-GCL 的 boundary 叙事已被 BES/UGCL/ASPECT/HAR 等近邻强烈包围；必须收缩为无监督 contrastive eligibility diagnostic + 最小 objective routing，并证明它不是 degree、uncertainty、augmentation strength 或普通 loss weighting 的代理。
- Non-goals: 不追求 SOTA claim；不做 LLM/KAN/大架构替换；不使用 test label 或 test set 调参；不声称 first boundary-aware GCL；不把 pilot/development 结果写成 formal claim。
- Constraints: 基于现有 GRACE/GCA 代码路线；节点分类主协议为 stratified random `1:1:8`；GCL evaluator 为 frozen encoder + Logistic Regression；首轮只允许 pilot/development 证据；优先 Cora/CiteSeer 3 seeds pilot，再决定是否 10 seeds development。
- Success condition: 如果一个纯无监督 eligibility signal 能预测节点对强 alignment 的受益/受损差异，并且 objective routing 在控制 degree/uncertainty/augmentation strength 与 scalar weighting 后仍有独立增益，则方向可继续；否则 `PIVOT/KILL`。

## Technical Gap

GRACE/GCA 把同一节点的两个增强视图作为正对，并对所有节点使用相同 InfoNCE 目标。已有工作已经分别处理了 adaptive augmentation、false negatives、uncertainty weighting、degree bias、node-wise spectral fusion、semantic drift 和 boundary embedding shaping，但这些工作通常问的是：

- 如何生成更好的 view；
- 如何处理 pair-level false negatives；
- 如何用 uncertainty / degree / spectral policy 做加权或通道融合；
- 如何在 supervised/end-to-end GNN 中塑造 boundary embeddings。

CER-GCL 的保留问题更窄：**在无监督 GCL 预训练中，是否存在可证伪的 node-wise eligibility signal，能判断某个节点是否应该接受强 positive alignment？** 如果有，最小机制不是新 backbone，而是一个 objective router，用同一个 score 比较三种处理：strong InfoNCE、weak consistency、stability-only/downweight。

Naive fixes 不够：

- 只降低全局 temperature 或 loss weight 无法说明节点级差异。
- degree-only / uncertainty-only 可能只是已有 HAR/UGCL/GRADE 的复现。
- adaptive augmentation 解决的是 view 生成，不直接检验 positive alignment 是否适合每个节点。
- boundary 叙事会撞上 BES，因此不能作为主问题。

## Method Thesis

- One-sentence thesis: CER-GCL tests whether an unsupervised contrastive eligibility signal can route nodes among strong alignment, weak consistency, and stability-only objectives more effectively than scalar weighting or known proxies.
- Why this is the smallest adequate intervention: 它只改 GRACE/GCA 的 loss path，不改 encoder、数据集、split、evaluator 或 backbone capacity。
- Why this route is timely: 2025-2026 的近邻工作已经证明 node-wise heterogeneity、uncertainty、stability 与 boundary vulnerability 是真实问题；CER-GCL 的机会在于把它们收缩成一个可证伪的无监督 routing question，而不是堆新模块。

## Contribution Focus

- Dominant contribution: A diagnostic-first formulation of contrastive eligibility in node-level Graph SSL, plus a minimal objective router to test whether eligibility is actionable.
- Optional supporting contribution: A strict falsification suite that separates eligibility from degree, uncertainty, augmentation instability, scalar loss weighting, and weaker global contrastive pressure.
- Explicit non-contributions: 新 encoder、新 augmentation generator、prototype pseudo-labeling、LLM semantic view、SOTA benchmark sweep、supervised boundary shaping。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: GRACE/GCA-style GCN encoder and existing project data/evaluation scripts.
- New trainable components: none in the first version. Eligibility uses stop-gradient statistics.
- Tempting additions intentionally not used: prototype clustering, adversarial view generator, labels for routing, neural score network, LLM/TAG features.

### System Overview

```text
graph + features
  -> two standard GRACE views
  -> shared encoder
  -> node statistics:
       s_instability = cross-view embedding instability
       s_fs = feature-structure disagreement proxy
       s_loss = contrastive loss stability / hardness
  -> stop-gradient eligibility score e_i
  -> route node i:
       high eligibility     -> standard InfoNCE
       medium eligibility   -> weak consistency
       low eligibility      -> stability-only / downweighted objective
  -> frozen encoder
  -> Logistic Regression evaluator
```

### Core Mechanism

- Input / output: 输入两个增强视图的节点 embedding 与图/特征统计；输出每个节点的 route mask 和 loss coefficient。
- Score construction:
  - `s_instability`: 两个增强视图中同一节点 embedding 的 cosine distance 或 normalized L2 drift。
  - `s_fs`: 原始特征邻域相似度与结构邻域相似度的差异，避免只用 degree。
  - `s_loss`: 节点级 contrastive loss 的 EMA rank，作为 hard-to-align proxy。
  - `e_i = stopgrad(normalize(-a*s_instability - b*s_fs - c*s_loss))`，首轮固定 `a=b=c=1`，不通过 test 调参。
- Routing:
  - Top eligibility: standard node-wise InfoNCE。
  - Middle eligibility: weak consistency，降低 positive alignment sharpness，不强化 uniformity pressure。
  - Low eligibility: stability-only / downweight，只保留 collapse guard 或小权重 consistency。
- Why this is the main novelty: 关键不是 score 某一项，而是把 “是否适合强 positive alignment” 作为可证伪对象，并用 route-vs-weighting 实验检验它是否有独立机制价值。

### Modern Primitive Usage

- No LLM / VLM / Diffusion / RL primitive is used.
- Justification: 该项目主线是普通图节点分类，数据不一定有文本语义；引入 LLM 会扩大复现和适用性风险，不解决当前 novelty collision。

### Integration into Downstream Pipeline

- Attach point: GRACE loss computation after embeddings are produced.
- Frozen parts: encoder architecture, splits, evaluator, dataset preprocessing.
- Trainable new parts: none initially.
- Inference: no router is needed at downstream evaluation time; only frozen embeddings are evaluated.

### Training Plan

1. Warmup: 前 `w` epochs 使用 vanilla GRACE，收集 node statistics 的 EMA。
2. Routing phase: 每 `k` epochs 更新 eligibility quantile thresholds。
3. Loss:
   - `L = mean_i r_strong(i) * L_InfoNCE(i) + r_weak(i) * L_weak(i) + r_low(i) * L_stability(i)`
   - `r_*` 是 stop-gradient route mask。
4. Hyperparameters:
   - 首轮固定 quantile，例如 high 60%、middle 30%、low 10%。
   - 如果需要选择阈值，只能用 train/val 或无标签 criterion，禁止 test。

### Failure Modes and Diagnostics

- Failure mode: score 只是 degree proxy。
  - Detect: score-degree Spearman correlation；degree-only routing baseline。
  - Fallback: 若控制 degree 后无独立信号，停止该方向。
- Failure mode: gain 来自全局弱 contrastive pressure。
  - Detect: global lower temperature / scalar downweight baseline。
  - Fallback: 若同等 global pressure 匹配，停止。
- Failure mode: routing 不优于 scalar weighting。
  - Detect: same score scalar weighting vs hard routing。
  - Fallback: 将方法降级为 diagnostic，不做 routing method claim。
- Failure mode: boundary label post-hoc 分组没有受益。
  - Detect: true-label post-hoc core/boundary/high-disagreement bucket；训练不使用标签。
  - Fallback: 不再讲 boundary，只保留 alignment eligibility diagnostic。

## Novelty and Elegance Argument

CER-GCL 与 BCE-GCL 的关键区别是：不再声称 boundary-aware adaptive contrastive learning，而是把 reviewer 质疑变成方法的主检验。它只问一个狭窄问题：标准 GCL 的强 alignment 是否对所有节点同样合适？若不是，能否用无监督 eligibility signal 作出比 degree/uncertainty/weighting 更好的 routing 决策？

这使贡献更小，但更可证伪，也更符合当前项目纪律。

## Claim-Driven Validation Sketch

### Claim 1: Eligibility signal has diagnostic value beyond known proxies

- Minimal experiment: Cora/CiteSeer 3 seeds pilot，记录 eligibility 与 post-hoc alignment benefit / node classification correctness change 的相关性。
- Baselines / ablations: degree-only、uncertainty-only、augmentation-instability-only、feature-structure-only、random score。
- Metric: Spearman correlation、bucket-wise accuracy delta、alignment drift delta。
- Expected evidence: full eligibility 比单一 proxy 更能预测哪些节点在强 InfoNCE 下受损或受益。

### Claim 2: Objective routing is not just scalar reweighting

- Minimal experiment: vanilla GRACE vs same-score scalar weighting vs hard routing vs global weak pressure。
- Baselines / ablations: random routing、routing without score、weak-consistency only、downweight only。
- Metric: frozen Logistic Regression accuracy、bucket-wise diagnostic、embedding rank/variance collapse guard。
- Expected evidence: routing 在主要 bucket diagnostic 上优于 scalar weighting；若只提升 overall accuracy 但无 bucket evidence，claim 不成立。

## Experiment Handoff Inputs

- Must-prove claims:
  - C1: eligibility signal captures alignment suitability beyond known proxies。
  - C2: objective routing has independent value beyond scalar weighting。
- Must-run ablations:
  - degree-only、uncertainty-only、augmentation-instability-only、feature-structure-only、random。
  - same-score scalar weighting、global weak pressure、routing without score。
- Critical datasets / metrics:
  - Cora/CiteSeer pilot, stratified random `1:1:8`, seeds 0-2 first。
  - frozen encoder + Logistic Regression。
  - post-hoc bucket diagnostics; no training with test labels。
- Highest-risk assumptions:
  - eligibility score is not a proxy for degree/uncertainty。
  - weak/routed alignment helps the intended nodes rather than merely regularizing all nodes。

## Compute & Timeline Estimate

- Estimated GPU-hours: pilot 约 4-8 GPU-hours；3 seeds Cora/CiteSeer with GRACE variants。
- Data / annotation cost: none。
- Timeline: 1 day for implementation + smoke, 1-2 days for pilot matrix if scripts are stable。

