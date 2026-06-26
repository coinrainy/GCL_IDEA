# CAST-GCL Novelty Check

日期：2026-06-26  
Reviewer：fresh `gcl_scientific_reviewer`  
Verdict：`REVISE`  
Novelty：`6.8/10`  
Confidence：`0.72`

## Reviewer Summary

CAST 比 WILLOW 更值得继续作为活跃候选，因为它从 same-anchor hard positive view 推进到 cross-node semantic positive closure，更直接触碰 node-level GCL false negatives。它也明确摆脱了单纯 loss trick。

但 CAST 还不能 `GO`。当前最大问题是：`cert_error + edit_cost + anchor_drift` 可能只是 similarity、PPR proximity、embedding smoothness 的复杂重写。审稿人很可能认为它是：

```text
Graph-JEPA-like certificate + heuristic candidate mining + multi-positive GCL
```

而不是一个真正不可替代的 semantic transport mechanism。

## Score

| Idea | Reviewer status | Novelty | Role |
|---|---|---:|---|
| CAST-GCL | REVISE | 6.8/10 | active revised pre-smoke candidate |
| WILLOW-GCL | REVISE | 7.0/10 | module/control; less direct for false negatives |
| SIVA-GCL | REVISE | 6.7/10 | reconstruction-critic control |
| BOND-GCL | REVISE / deprioritized | 5.8-6.0/10 | loss-side archived baseline |

WILLOW's score was slightly higher numerically, but reviewer recommendation keeps CAST active because CAST targets false negatives more directly. The score gap reflects that CAST is less mature and more exposed to positive-mining collision.

## Minimal Irreplaceable Core

CAST is only novel if the following remains true:

```text
latent ego target-prediction certificate
  certifies cross-real-node low-energy intervention paths
  and converts likely false negatives into multi-positive / neutral closure
```

If it becomes kNN positives, BMM positives, JEPA scoring, or denominator weighting, it should be killed or downgraded.

## Required Revisions

1. Define legal intervention operators exactly.
2. Define `cert_error(u -> v)` on mixed ego views.
3. Freeze the certificate before relation construction; precompute closure offline for smoke.
4. Decouple candidate pool from certificate evidence.
5. Pre-register `tau_transport`, max positives per anchor, and neutral rules.
6. Add similarity-only, edit-cost-only, anchor-drift-only, candidate-pool-only controls.
7. Fix compute/search budget.
8. Make the falsifiable claim: transport energy should predict offline label agreement better than kNN/BMM/PPR and reduce false-negative repulsion mass.

## Decision

`REVISE_TO_CAST_REVISED_PRE_SMOKE`。

No code, smoke, pilot, formal result, or performance claim exists.
