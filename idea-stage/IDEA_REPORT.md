# Idea Discovery Report: CAST-GCL

**方向**：图对比学习假负样本方向，面向普通节点分类，继续避开 loss-only trick，追求更有 2026 投稿潜力的机制创新  
**日期**：2026-06-26  
**Pipeline**：research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline  
**阶段结论**：`REVISE_TO_CAST_REVISED_PRE_SMOKE`

## Executive Summary

上一轮 WILLOW-GCL 已经把方向从 loss-side 修补推进到 latent certificate driven hard positive views，但它仍存在一个硬伤：容易被 reviewer 解释为 **Graph-JEPA / Predict-Cluster-Refine + learned augmentation scorer**。这比 BOND/SIVA 更好，但还没有完全满足“足够创新”的目标。

本轮进一步把主线推进为 **CAST-GCL / Certificate-guided Semantic Transport for Graph Contrastive Learning**。CAST 不再只生成同一节点的 hard positive view，而是用 latent ego target-prediction certificate 构造 **跨节点语义传输图**：如果真实节点 `j` 可以通过低能量的 node-local intervention path 从 anchor `i` 到达，那么 `j` 就不应再作为 `i` 的 negative，而应进入 multi-positive / neutral relation closure。

核心思想：

```text
false negatives are missing semantic transport relations,
not merely over-weighted denominator terms.
```

fresh `gcl_scientific_reviewer` 给 CAST 的 verdict 是 `REVISE`，novelty `6.8/10`，confidence `0.72`。结论很明确：CAST 是当前最值得保留的候选，因为它比 WILLOW 更直接处理 false negatives；但它还不能 `GO`，必须证明 transport energy 不是 kNN/PPR/BMM positive mining 的复杂重写。

## Why CAST Replaces WILLOW As The Active Candidate

| Aspect | WILLOW | CAST |
|---|---|---|
| Main mechanism | same-anchor certified hard positive view | cross-node certified semantic transport |
| False-negative handling | indirect, by strengthening positive signal | direct, by identifying real nodes that should not be negatives |
| Main prior risk | Graph-JEPA + augmentation scorer | positive mining / transport heuristic |
| Required evidence | harder same-node positives help | transported nodes have high label agreement and reduce false-negative repulsion mass |

WILLOW remains useful as a module/control. CAST uses WILLOW's latent certificate, but adds the missing cross-node relation construction step.

## Prior Boundary

CAST must avoid the following collisions:

- **Graph-JEPA / Predict, Cluster, Refine**：latent context-target prediction already exists; CAST uses it only as a certificate for cross-node relation construction.
- **PMGCL / positive mining**：pair probability estimation already exists; CAST requires low-energy intervention paths, not only similarity/BMM probability.
- **BalanceGCL**：semantic-aware positives and hard negatives already exist at graph level; CAST is node-level, does not use class-aware counterfactual gradients, and does not generate hard negatives as the main contribution.
- **GCA/RGCL/AutoGCL**：learned augmentation already exists; CAST builds a certified transport relation graph between real nodes.

## Recommended Idea

### CAST-GCL: Certificate-guided Semantic Transport

- **Method**：train a latent ego target-prediction certificate; for each anchor, search candidate real nodes through short node-local intervention paths; score the path by certificate error, edit cost, and anchor drift; low-energy reachable nodes become semantic positives or neutral nodes in GCL.
- **Hypothesis**：false negatives can be reduced more directly by constructing a certified cross-node positive closure than by reweighting negatives or only making same-node positives harder.
- **Minimum experiment**：Cora seed=0 smoke; compare GRACE, WILLOW same-node certificate, kNN multi-positive, PMGCL-lite/BMM positive mining, PPR/diffusion positives, candidate-pool-only, similarity-only transport, edit-cost-only / anchor-drift-only, certificate-shuffled CAST, random budget-matched transport, CAST one-step transport, and CAST two-step transport.
- **Diagnostics**：transported positive count, offline label agreement, transport energy vs label agreement correlation, false-negative repulsion mass, LogReg accuracy, search budget.
- **Novelty status**：fresh reviewer `REVISE`，novelty `6.8/10`，confidence `0.72`。
- **Risk**：HIGH；transport may collapse into expensive positive mining or kNN multi-positive.
- **Pilot result**：SKIPPED；本轮未实现代码、未跑新实验。

## Mandatory Kill Rules

| Kill rule | 解释 |
|---|---|
| kNN multi-positive matches CAST | transport certificate unnecessary |
| PMGCL-lite/BMM matches CAST | CAST reduces to positive mining |
| certificate-shuffled CAST matches CAST | certificate has no semantic content |
| transport energy does not correlate with offline label agreement | kill CAST |
| CAST only works by selecting many more positives | invalid unless budget-matched variant also works |
| WILLOW same-node matches CAST | cross-node closure unnecessary; downgrade to WILLOW |
| candidate-pool-only / similarity-only matches CAST | transport certificate adds no value; kill or major pivot |

## Deprioritized / Control Ideas

| Idea | Current role |
|---|---|
| WILLOW-GCL | module/control; provides latent certificate but not enough cross-node false-negative handling |
| SIVA-GCL-positive-core | reconstruction-critic control |
| BOND-GCL | archived loss-side baseline |
| DSR-GCL | archived/pivoted due weak Cora seed=0 audit-smoke |

## Refined Proposal

- Candidate brief: `idea-stage/CAST_GCL_CANDIDATE_20260626_121500.md`
- Novelty check: `idea-stage/CAST_GCL_NOVELTY_CHECK.md`
- Mechanism spec: `refine-logs/CAST_MECHANISM_SPEC.md`
- Prior boundary: `literature/CAST_PRIOR_BOUNDARY.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Decision

`REVISE_TO_CAST_REVISED_PRE_SMOKE`.

当前只允许进入最小 Cora seed=0 smoke 设计/实现，不允许 formal，不允许写 SOTA/robust/comprehensive，不允许产生性能 claim。若 kNN/PPR/BMM/candidate-pool-only/shuffled-certificate controls 解释掉 CAST，应继续 pivot。
