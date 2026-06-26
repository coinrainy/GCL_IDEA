# Best Idea Decision Audit

日期：2026-06-26  
目标：找到一个足够创新、不是 loss trick、适合作为 2026 GCL false-negative node-classification 主线的 best idea。  
最终选择：**IRIS-GCL / Interventional Response Invariant Signatures for Graph Contrastive Learning**  
决策标签：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`

## Decision Summary

当前 best idea 是 **IRIS-GCL**。

选择依据不是它已经被实验证明，而是它在所有已探索候选中最有机会避开 reviewer 最敏感的归约：

```text
loss trick
positive mining
Graph-JEPA/PCR wrapper
learned augmentation scorer
ordinary kNN/PPR/embedding proximity
```

IRIS 的核心是：

```text
false negatives = cross-node response-invariant siblings under anti-proximity
```

也就是说，两个节点即使在 feature / graph / embedding 空间里不近，只要它们在一组固定干预环境下呈现相似 response function，就不应被强行当作 negatives。

## Why Not Previous Candidates

| Candidate | Final role | Why not best |
|---|---|---|
| BOND-GCL | archived baseline | reviewer/user both judged it too close to loss-side correction |
| SIVA-GCL | reconstruction-critic control | stronger than loss trick, but GraphMAE/GCA/RGCL composition risk high |
| WILLOW-GCL | module/control | latent certificate idea useful, but easy to be read as Graph-JEPA/PCR + augmentation scorer |
| CAST-GCL | mandatory control | directly targets false negatives, but reviewer judged it too close to positive mining / candidate mining |
| IRIS-GCL | active best idea | highest novelty among reviewed candidates; response-function equivalence under anti-proximity is the cleanest non-loss mechanism |

## Fresh Reviewer Evidence

Fresh `gcl_scientific_reviewer` compared CAST and IRIS directly:

| Idea | Novelty | Confidence | Reviewer judgment |
|---|---:|---:|---|
| CAST-GCL | 6.5/10 | 0.74 | mature but too close to positive mining |
| IRIS-GCL | 7.2/10 | 0.61 | higher risk, but better bet for sufficient innovation |

Reviewer verdict:

```text
SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE
```

## Literature Boundary Checked

IRIS must not claim novelty for generic invariance or causal graph learning. Its specific boundary is narrower:

- vs GCIL / causal invariant GCL: GCIL learns invariant node representations under spectral intervention; IRIS compares cross-node response functions to identify false negatives.
- vs CIA-GCL / invariant augmentation: IRIS is not a graph-level or brain-graph augmentation method; it is node-level false-negative relation discovery.
- vs PMGCL / ProGCL / BMM mining: IRIS does not estimate pair probability from similarity; it requires anti-proximity and response-function congruence.
- vs CI-GCL / community-invariant augmentation: IRIS does not preserve community structure as augmentation rationale; it uses intervention response signatures to discover cross-node sibling relations.
- vs CAST: IRIS avoids transport/candidate-pool scoring and uses functional response equivalence instead.

## Minimal Irreplaceable Core

IRIS remains worth pursuing only if this core stays intact:

```text
fixed intervention battery
        ->
node response fingerprint
        ->
anti-proximity response-invariant sibling discovery
        ->
multi-positive / neutral relation closure
```

If it becomes pair similarity, kNN/PPR, BMM mining, learned augmentation, or denominator reweighting, it should be killed.

## Required Smoke Proof

IRIS can only move beyond smoke if Cora seed=0 diagnostics show:

1. response sibling label agreement beats kNN/PPR/PMGCL-lite/structural-role controls under matched budget；
2. response similarity has partial correlation with label agreement after controlling feature similarity, embedding similarity, PPR proximity, and degree；
3. response-shuffled is worse than IRIS full；
4. no-anti-proximity does not dominate IRIS full；
5. CAST one-step does not match IRIS with lower cost；
6. gains are not explained by more positives, more compute, or broader candidate budget。

## Current Integrity Status

- Code implemented：no。
- Smoke run：no。
- Pilot run：no。
- Formal run：no。
- Performance claim：none。
- Current allowed next action：Cora seed=0 smoke only。

## Final Decision

`IRIS-GCL` is the best current idea for the user's requirement.

This completes the idea-selection stage, not the experimental validation stage.
