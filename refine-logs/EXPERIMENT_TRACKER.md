# Experiment Tracker: CAST-GCL

## Current Gate

- Status：`PLANNED_ONLY`
- Decision：`REVISE_TO_CAST_REVISED_PRE_SMOKE`
- Allowed next action：implement minimal Cora seed=0 smoke only, with full mining/proximity/certificate controls。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim。

## Smoke Matrix

| ID | Variant | Status | Result | Decision |
|---|---|---|---|---|
| C0 | GRACE | not_started | - | required |
| C1 | WILLOW same-node certificate | not_started | - | required |
| C2 | kNN multi-positive | not_started | - | required |
| C3 | PMGCL-lite/BMM positive mining | not_started | - | required |
| C4 | PPR/diffusion positives | not_started | - | required |
| C5 | candidate-pool-only positives | not_started | - | required |
| C6 | similarity-only transport | not_started | - | required |
| C7 | edit-cost-only / anchor-drift-only | not_started | - | required |
| C8 | certificate-shuffled CAST | not_started | - | required |
| C9 | random transport budget-matched | not_started | - | required |
| C10 | CAST one-step transport | not_started | - | main |
| C11 | CAST two-step transport | not_started | - | main |

## Evidence Status

- Fresh reviewer：completed for CAST, verdict `REVISE`, novelty `6.8/10`, confidence `0.72`。
- Code implemented：no。
- Smoke run：no。
- Pilot run：no。
- Formal run：no。
- Claim support：none。
