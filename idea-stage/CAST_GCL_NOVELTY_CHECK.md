# CAST-GCL Novelty Check

日期：2026-06-26  
Reviewer：fresh `gcl_scientific_reviewer`  
Verdict：`REVISE`  
Novelty：`6.8/10`  
Confidence：`0.72`

## Reviewer Summary

CAST 比 WILLOW 更值得继续作为活跃候选，因为它从 same-anchor hard positive view 推进到 cross-node semantic positive closure，更直接触碰 node-level GCL false negatives。它也明确摆脱了单纯 loss trick。

但 CAST 还不能 `GO`。当前最大问题是：`cert_error + edit_cost + anchor_drift` 可能只是 similarity、PPR proximity、embedding smoothness 的复杂重写。审稿人很可能认为它是 Graph-JEPA-like certificate + heuristic candidate mining + multi-positive GCL。

## Minimal Irreplaceable Core

```text
latent ego target-prediction certificate
  certifies cross-real-node low-energy intervention paths
  and converts likely false negatives into multi-positive / neutral closure
```

## Decision

`REVISE_TO_CAST_REVISED_PRE_SMOKE`。

No code, smoke, pilot, formal result, or performance claim exists.
