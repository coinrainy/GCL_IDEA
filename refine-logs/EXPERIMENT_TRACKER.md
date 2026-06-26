# Experiment Tracker: IRIS / R2-IRIS / CPR-IRIS

## Current Gate

- Status：`CPR_WEIGHT_SMOKE_COMPLETED_MIXED`
- Decision：`REVISE_TOWARD_CERTIFIED_CAST`
- Completed action：implemented response-certified proximity variants I14-I21 and ran Cora seed=0 smoke only。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim。

## Smoke Matrix

| ID | Variant | Status | Test@best-val | Label agreement | Decision |
|---|---|---|---:|---:|---|
| I0 | GRACE | completed | 84.73 | 0.0000 | reference |
| I1 | kNN multi-positive | completed | 85.29 | 0.8153 | strongest label agreement |
| I2 | PPR/diffusion positives | completed | 83.58 | 0.7272 | graph proximity control |
| I3 | PMGCL-lite/BMM | completed | 85.38 | 0.7698 | probabilistic control |
| I4 | CAST one-step proxy | completed | 85.65 | 0.7545 | strongest accuracy |
| I5 | IRIS hard anti-proximity | completed | 76.34 | 0.2426 | failed |
| I10 | R2 residualized response | completed | 84.13 | 0.7784 | rescue, not enough |
| I13 | residual + CAST hybrid | completed | 84.87 | 0.7971 | better label agreement, lower accuracy |
| I17 | CAST + residual cert w=0.30 | completed | 85.52 | 0.7953 | best CPR tradeoff |
| I18 | CAST + residual cert w=0.15 | completed | 85.47 | 0.7817 | close to CAST |
| I19 | CAST + residual cert w=0.45 | completed | 85.38 | 0.8020 | stronger label agreement |
| I20 | kNN + residual cert w=0.15 | completed | 85.15 | 0.8120 | near kNN agreement |
| I21 | kNN + residual cert w=0.45 | completed | 84.50 | 0.8018 | weaker accuracy |

## Evidence Status

- Code implemented：yes，`scripts/run_iris_smoke.py`。
- Config implemented：yes，`configs/iris_smoke.yaml`。
- Smoke run：completed, Cora seed=0 only。
- Pilot run：no。
- Formal run：no。
- Claim support：mixed smoke only；不支持性能提升 claim。
- Result summary：`results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.md`。
- Bridge results：`refine-logs/EXPERIMENT_RESULTS.md`。

## Decision Notes

CPR is the best direction so far because it improves CAST-style sibling quality without fully collapsing to the failed anti-proximity rule. It is still not enough for Pilot-A/B: the next change should make response residual act as a certificate for neutralization/closure over CAST candidates, not merely as a linear additive score.
