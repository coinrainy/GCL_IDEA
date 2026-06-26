# Best Idea Decision Audit

日期：2026-06-26  
最终选择：**IRIS-GCL / Interventional Response Invariant Signatures for Graph Contrastive Learning**  
决策标签：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`

## Why IRIS

IRIS is the current best idea because it is the least reducible to loss tricks or ordinary positive mining among all explored candidates.

Core:

```text
false negatives = cross-node response-invariant siblings under anti-proximity
```

Fresh reviewer compared CAST and IRIS:

| Idea | Novelty | Reviewer judgment |
|---|---:|---|
| CAST-GCL | 6.5/10 | mature but too close to positive mining |
| IRIS-GCL | 7.2/10 | more novel, high risk, better bet for sufficient innovation |

## Current Status

IRIS is the selected best idea, but not experimentally validated.

- Code：not implemented。
- Smoke：not run。
- Pilot/Formal：not run。
- Performance claim：none。

Next allowed action：Cora seed=0 smoke only, with controls in `refine-logs/EXPERIMENT_PLAN.md`.
