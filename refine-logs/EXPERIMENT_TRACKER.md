# Experiment Tracker: IRIS-GCL

## Current Gate

- Status：`SMOKE_COMPLETED_NEGATIVE`
- Decision：`PIVOT_REQUIRED`
- Completed action：implemented and ran minimal Cora seed=0 smoke only。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim、继续当前 IRIS anti-proximity 机制的 Pilot-A/B。

## Smoke Matrix

| ID | Variant | Status | Test@best-val | Label agreement | Decision |
|---|---|---|---:|---:|---|
| I0 | GRACE | completed | 84.78 | 0.0000 | reference |
| I1 | kNN multi-positive | completed | 85.24 | 0.8154 | stronger than I5 |
| I2 | PPR/diffusion positives | completed | 83.58 | 0.7272 | stronger than I5 |
| I3 | PMGCL-lite/BMM | completed | 85.38 | 0.7699 | stronger than I5 |
| I4 | CAST one-step proxy | completed | 85.65 | 0.7548 | dominates I5 |
| I5 | IRIS full | completed | 76.48 | 0.2418 | failed |
| I6 | IRIS response-shuffled | completed | 80.07 | 0.1027 | mixed diagnostic |
| I7 | IRIS no anti-proximity | completed | 84.59 | 0.7787 | kill rule triggered |
| I8 | IRIS structural-signature-only | completed | 77.68 | 0.1125 | weaker than I5 on label agreement |
| I9 | IRIS no gradient-proxy | completed | 76.48 | 0.2418 | equivalent to I5 by design |

## Evidence Status

- Code implemented：yes，`scripts/run_iris_smoke.py`。
- Config implemented：yes，`configs/iris_smoke.yaml`。
- Smoke run：completed, Cora seed=0 only。
- Pilot run：no。
- Formal run：no。
- Claim support：negative smoke only；不支持性能提升 claim。
- Result summary：`results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md`。
- Bridge results：`refine-logs/EXPERIMENT_RESULTS.md`。
- Bridge review：`refine-logs/EXPERIMENT_CODE_REVIEW.md`。

## Decision Notes

The planned smoke pass conditions are not met. I5 fails against proximity mining controls and is dominated by no-anti-proximity I7 and CAST proxy I4. The only positive diagnostic is response-similarity partial correlation with label agreement after controls (`0.3707`), but the current anti-proximity selection rule fails to exploit it.
