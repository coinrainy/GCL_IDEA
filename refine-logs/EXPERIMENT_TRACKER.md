# Experiment Tracker: IRIS-GCL

## Current Gate

- Status：`PLANNED_ONLY`
- Decision：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`
- Allowed next action：implement minimal Cora seed=0 smoke only。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim。

## Smoke Matrix

| ID | Variant | Status | Result | Decision |
|---|---|---|---|---|
| I0 | GRACE | not_started | - | required |
| I1 | kNN multi-positive | not_started | - | required |
| I2 | PPR/diffusion positives | not_started | - | required |
| I3 | PMGCL-lite/BMM | not_started | - | required |
| I4 | CAST one-step | not_started | - | required |
| I5 | IRIS full | not_started | - | main |
| I6 | IRIS response-shuffled | not_started | - | required |
| I7 | IRIS no anti-proximity | not_started | - | required |
| I8 | IRIS structural-signature-only | not_started | - | required |
| I9 | IRIS no gradient-proxy | not_started | - | required |

## Evidence Status

- Fresh reviewer：completed for IRIS vs CAST, decision `SWITCH_TO_IRIS`, IRIS novelty `7.2/10`, CAST novelty `6.5/10`。
- Code implemented：no。
- Smoke run：no。
- Pilot run：no。
- Formal run：no。
- Claim support：none。
