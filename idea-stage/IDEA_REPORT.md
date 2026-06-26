# Idea Discovery Report: IRIS-GCL

**方向**：图对比学习假负样本方向，面向普通节点分类，寻找足够创新、适合 2026 投稿潜力的最佳 idea  
**日期**：2026-06-26  
**Pipeline**：research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline  
**阶段结论**：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`

## Executive Summary

经过 BOND → SIVA → WILLOW → CAST 的连续筛选，当前最好的 idea 切换为 **IRIS-GCL / Interventional Response Invariant Signatures for Graph Contrastive Learning**。

fresh `gcl_scientific_reviewer` 对 CAST 与 IRIS 做了直接比较，给出结论：

```text
SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE
```

IRIS 的 novelty 为 `7.2/10`，confidence `0.61`；CAST 降为 `6.5/10`。Reviewer 判断：CAST 更成熟但太接近 positive mining；IRIS 更高风险，但更像一个“足够创新”的 2026 论文主线，因为它用 **跨节点 interventional response-function equivalence** 来发现 false negatives，并通过 **anti-proximity** 强制避开普通 kNN/PPR/embedding positive mining。

当前没有 smoke、pilot、development 或 formal 结果。IRIS 是当前 best bet，不是已验证方法。

## Why IRIS Is Better Than CAST

| Aspect | CAST | IRIS |
|---|---|---|
| False-negative criterion | low-energy semantic transport path | invariant response-function sibling |
| Main risk | positive mining with certificate wrapper | response invariance may reflect role/degree not class semantics |
| How it avoids proximity mining | controls after candidate mining | anti-proximity built into sibling definition |
| Reviewer novelty | 6.5/10 | 7.2/10 |
| Current role | mandatory control | active best idea |

CAST 仍直接服务 false negatives，但它从 candidate pool 出发，再用 certificate 排序，容易被说成高级 positive mining。IRIS 的判据不同：两个节点即使 feature/graph/embedding 不近，只要在标准干预电池下呈现相似 response function，就不应被强行作为 negatives。

## Recommended Idea

### IRIS-GCL: Interventional Response Invariant Signatures

**Core thesis**：

```text
False negatives are cross-node response-invariant siblings under anti-proximity.
```

也就是说，节点语义相似不只体现在“近”，也可能体现在“对环境/结构/特征干预的响应函数相似”。IRIS 不直接用 pair similarity，也不重写 InfoNCE loss，而是先构造 response fingerprint，再发现 anti-proximity response siblings。

## Method Sketch

### 1. Fixed Intervention Battery

定义一组 label-free graph interventions：

- spectral high-frequency perturbation；
- low-pass / diffusion-scale perturbation；
- ego edge drop / keep；
- feature group mask；
- neighbor role swap；
- degree-bin preserving ego rewiring。

这些干预是固定环境扰动，不是 learned augmentation policy。

### 2. Minimal Response Fingerprint

先用 GRACE 或固定 SSL encoder warm up，然后冻结 encoder 构造 response signatures。

最小 IRIS 版必须先移除 reviewer 认为可能循环的 `positive-gradient proxy`，只保留：

```text
r_i(omega)
  = [
      delta z_i(omega),
      delta local smoothness_i(omega),
      delta ego prediction consistency_i(omega)
    ]
```

然后：

```text
R_i = concat_{omega in Omega} whiten(r_i(omega))
```

IRIS 使用 response fingerprint，而不是 raw embedding similarity。

### 3. Anti-proximity Response Siblings

节点 `j` 是 anchor `i` 的 response-invariant sibling 当且仅当：

```text
sim_response(R_i, R_j) >= tau_response
and sim_embedding(z_i, z_j) <= tau_embed_near
and graph_proximity(i, j) <= tau_graph_near
```

anti-proximity 不是装饰项；它是 IRIS 避开 kNN/PPR/embedding positive mining 的必要条件。必须报告被 anti-proximity 排除后的 coverage 与 label agreement。

### 4. Contrastive Relation Update

- same-node augmentations remain positives；
- response-invariant siblings become multi-positives；
- high response uncertainty nodes become neutral/excluded；
- all others remain ordinary negatives。

主贡献是 relation discovery，不是新 loss。

## Required Controls

| Control | Purpose |
|---|---|
| GRACE | base reference |
| kNN multi-positive | proximity mining |
| PPR/diffusion positives | graph proximity |
| PMGCL-lite/BMM | probabilistic positive mining |
| CAST one-step | strongest previous candidate/control |
| IRIS full | main candidate |
| IRIS response-shuffled | response semantic test |
| IRIS no anti-proximity | collapse-to-proximity test |
| IRIS structural-signature-only | role-equivalence test |
| IRIS no gradient-proxy | circularity test |

## Kill Rules

| Kill rule | Meaning |
|---|---|
| response-shuffled matches IRIS | response signature has no semantic content |
| structural-signature-only matches IRIS | IRIS is only role/degree equivalence |
| no anti-proximity beats full IRIS | IRIS collapses to ordinary proximity mining |
| response similarity has no partial correlation with label agreement after controlling feature/embedding/PPR/degree | kill IRIS |
| CAST matches IRIS on label agreement, false-negative repulsion mass, and accuracy with lower cost | do not switch; reconsider CAST or pivot |
| gains come only from more positives, compute, or broader candidate budget | invalid |

## Deprioritized / Control Ideas

| Idea | Current role |
|---|---|
| CAST-GCL | mandatory control; previous best but too close to positive mining |
| WILLOW-GCL | module/control; Graph-JEPA/PCR wrapper risk |
| SIVA-GCL | reconstruction-critic control |
| BOND-GCL | archived loss-side baseline |

## Refined Proposal

- IRIS challenger: `idea-stage/IRIS_GCL_CHALLENGER_20260626_130500.md`
- Prior boundary: `literature/IRIS_PRIOR_BOUNDARY.md`
- Reviewer decision: `idea-stage/IRIS_VS_CAST_REVIEW.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Decision

`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`.

IRIS 是当前“最好的 idea / best bet”，但仍是 high-risk pre-smoke。下一步只允许最小 Cora seed=0 smoke 或进一步机制压实；不允许 formal，不允许 SOTA/robust/comprehensive，不允许性能 claim。
