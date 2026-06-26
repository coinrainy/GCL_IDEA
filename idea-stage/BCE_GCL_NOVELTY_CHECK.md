# BCE-GCL 深度查新报告

生成时间：2026-06-26T08:34:42Z  
阶段：`/idea-discovery` Phase 3 — Deep Novelty Verification  
对象：BCE-GCL — Boundary-Conditioned Contrastive Eligibility for Graph Contrastive Learning  
状态：本地查新完成；fresh `gcl_scientific_reviewer` 独立审查完成。

## 一句话结论

BCE-GCL 不应原样进入 `/research-refine-pipeline`。本地查新给出 `REVISE`，fresh `gcl_scientific_reviewer` 给出更严格的 `PIVOT`，novelty score `4/10`。允许保留的问题不是 “Boundary-Conditioned GCL”，而是 pivot 为 **CER-GCL / Contrastive Eligibility Routing**：无监督 GCL 预训练中诊断哪些节点不适合强 positive alignment，并用轻量 objective routing 作为验证工具。

## Proposed Method

BCE-GCL 估计节点级 boundary / eligibility score，将节点路由为 core、boundary、unstable 三类：core 使用标准 InfoNCE，boundary 使用弱 boundary-preserving consistency，unstable 下调权重或只做 stability regularization。目标是在无监督 GCL 预训练阶段避免对 boundary / unstable nodes 做一刀切强 alignment，最终用 frozen encoder + Logistic Regression 做节点分类评估。

## Core Claims

| Claim | Novelty | Closest prior work | 判断 |
|---|---|---|---|
| 节点不是都适合强 contrastive alignment，需要 node-wise eligibility | MEDIUM | ASPECT, UGCL, HAR/SHARP, GRADE | node-wise policy / sample uncertainty / hard node weighting 已很接近；“eligibility” 作为问题命名还有空间，但必须证明不是 reweighting 换名。 |
| core / boundary / unstable 使用不同 objective route | MEDIUM | BES, UGCL, ASPECT-S | routing objective type 比单纯权重略有差异，但 boundary + adaptive contrastive learning 已被 BES 抢占一部分叙事。 |
| feature-structure disagreement + augmentation instability 可估计 boundary risk | LOW-MEDIUM | BES, semantic drift GCL, Bayesian uncertainty GCL | 各组成信号都有近邻；除非给出清晰无标签估计和解耦诊断，否则 novelty 偏弱。 |
| 该机制适合节点分类，可改善 boundary/high-disagreement 节点 | MEDIUM | BES, label non-uniformity, GRADE | 作为 diagnostic finding 可能有价值；作为主 method claim 需要 pilot 支撑。 |

## Closest Prior Work

| Paper | Year | Venue/source | Overlap | Key Difference / Delta |
|---|---:|---|---|---|
| Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement | 2026 | arXiv | 明确针对 class-boundary nodes，使用 adaptive contrastive learning shaping boundary embeddings。 | BCE-GCL 若保留，必须强调无监督 GCL pretraining 与 objective eligibility；BES 更像 GNN plug-in / supervised classification refinement。 |
| ASPECT: Node-Level Adaptive Spectral Fusion for Graph Contrastive Learning | 2026 | arXiv | node-wise policy、stability-aware extension、perturbation sensitivity。 | ASPECT route spectral/frequency channel；BCE-GCL route contrastive objective strength/type。 |
| Uncertainty-guided Graph Contrastive Learning from a Unified Perspective | 2025 | IJCAI | sample uncertainty 指导 augmentation 与 weighted GCL loss，处理 class ambiguity。 | BCE-GCL 需证明 boundary eligibility 不等同于 uncertainty，并且 objective routing 不是 weighted loss 变体。 |
| Mitigating Degree Bias Adaptively with Hard-to-Learn Nodes in GCL / SHARP | 2025 | arXiv | hard-to-learn nodes、adaptive reweight positives/negatives、degree-level fairness。 | BCE-GCL 不以 degree bias 为主，但 hard-node reweighting overlap 强。 |
| GRADE / Structural Fairness in GCL | 2022 | NeurIPS | low/high-degree nodes 不同策略、community boundary / degree bias 分析。 | BCE-GCL 的 boundary 不是 degree bucket，但必须做 degree-only baseline 证明差异。 |
| ProGCL / Node Similarity GCL | 2022 | ICML / arXiv | pair-level false negative handling、node similarity distribution。 | BCE-GCL 不是 pair negative reliability，而是 node eligibility；需要避免落回 false-negative weighting。 |
| Topology Reorganized GCL with Mitigating Semantic Drift | 2024 | arXiv | augmentation 语义漂移、prototype-based negative selection。 | BCE-GCL 将 semantic drift 信号用于节点 objective routing；但 semantic drift 本身不是新点。 |
| GCA | 2021 | WWW | adaptive augmentation，保护重要拓扑和属性语义。 | BCE-GCL 是 post-view objective routing，不是 view generator；但必须和 GCA/GCA-style adaptive augmentation 对照。 |

## Overall Novelty Assessment

- Local novelty score: `5.5/10`
- Reviewer novelty score: `4/10`
- Recommendation: `PIVOT`
- Key differentiator still possible: **无监督 GCL 预训练中的 node-wise contrastive eligibility + objective routing**，而不是泛化的 boundary-aware GNN 或 uncertainty-aware weighting。
- Main risk: reviewer 会认为这是 BES + UGCL + SHARP/GRADE 的组合式改名。
- 当前是否可进入实现：否。
- 当前是否可进入 `/research-refine-pipeline`：BCE-GCL 原样不可以；pivot 为 CER-GCL 后可以进入 refinement。

## Fresh Reviewer Verdict

- Reviewer route: fresh `gcl_scientific_reviewer`
- Trace path: `.aris/traces/novelty-check/2026-06-26_run01/`
- Verdict: `PIVOT`
- Novelty score: `4/10`
- Confidence: `0.72`

Reviewer 结论：

> 不允许以当前 BCE-GCL 原样进入 `/research-refine-pipeline`；允许先 PIVOT 成“无监督 contrastive eligibility diagnostic + lightweight routing”后，再进入 refine。

Reviewer 指出的 fatal weaknesses：

1. BES 已经非常接近 “boundary embedding shaping + adaptive contrastive learning + node classification”，继续使用 Boundary-Conditioned 叙事风险极高。
2. 从 “boundary nodes 受强 alignment 伤害” 到 “weak consistency/downweight 更好” 的机制链不闭合。
3. eligibility score 可能只是 degree、homophily、uncertainty、augmentation strength、feature norm 或 class imbalance 的代理变量。
4. 当前机制像 score + routing + reweighting + consistency 的堆叠，缺少不可替代的技术点。

## Suggested Positioning

建议改名和收缩贡献：

- 避免把题目写成 “Boundary-Conditioned Contrastive Learning”，因为 BES 已经非常接近。
- 推荐改为 **Eligibility-Routed Graph Contrastive Learning** 或 **Node-wise Contrastive Eligibility Routing**。
- contribution 从 “boundary-aware method” 改成 “a diagnostic and routing framework that decides whether a node should receive strong alignment, weak consistency, or stability-only regularization during unsupervised GCL pretraining”。
- boundary 只作为 eligibility 的一种来源，不作为全部贡献。

## 必须做的 Ablation / Diagnostic

1. `full eligibility` vs `degree-only` vs `uncertainty-only` vs `augmentation-instability-only` vs `feature-structure-disagreement-only`。
2. `objective routing` vs `same score but scalar loss weight`，证明不是普通 reweighting。
3. `weak consistency for boundary` vs `downweight boundary` vs `standard InfoNCE for all nodes`。
4. boundary/high-disagreement bucket 诊断：整体 accuracy 提升不够，必须证明目标节点群体受益或至少不受损。
5. 与 GRACE、GCA、BGRL/CCA-SSG、UGCL/SHARP 若可复现版本做 protocol-consistent comparison。
6. split / evaluator 必须遵循项目规则，不允许 public split 数字和 `1:1:8` 主表混用。

## 必须避免的 Claim

- 禁止声称 “first boundary-aware GCL”。
- 禁止声称 “robust” 或 “SOTA”，除非后续 formal 结果支持。
- 禁止把 pilot/development 结果写成 final claim。
- 禁止把 boundary score 解释为真实 class boundary，除非只用 train/val 诊断或 post-hoc 标签分析。
- 禁止用 test label 选择 score 阈值、Logistic Regression `C` 或任何 evaluator 参数。

## Phase Decision

当前阶段决策：`PIVOT`

下一步：

1. 将 BCE-GCL 原样叙事标记为 `PIVOT`，不进入实现。
2. 创建 pivot 版本：CER-GCL — Contrastive Eligibility Routing for Graph SSL。
3. 对 CER-GCL 进入 `/research-refine-pipeline`，但 refinement 必须以 diagnostic-first、falsification-first 为约束。
4. 若 CER-GCL refinement 仍无法形成清晰 problem anchor，则转向 ProtoGuard-GCL。
