# Experiment Tracker: IRIS / R2-IRIS

## Current Gate

- Status：`R2_SMOKE_COMPLETED_MIXED`
- Decision：`REVISE_NOT_PILOT`
- Completed action：implemented residualized response variants I10-I13 and ran Cora seed=0 smoke only。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim。

## Smoke Matrix

| ID | Variant | Status | Test@best-val | Label agreement | Decision |
|---|---|---|---:|---:|---|
| I0 | GRACE | completed | 84.73 | 0.0000 | reference |
| I1 | kNN multi-positive | completed | 85.29 | 0.8151 | strongest label agreement |
| I2 | PPR/diffusion positives | completed | 83.58 | 0.7272 | graph proximity control |
| I3 | PMGCL-lite/BMM | completed | 85.52 | 0.7699 | stronger accuracy than R2 |
| I4 | CAST one-step proxy | completed | 85.65 | 0.7548 | strongest accuracy |
| I5 | IRIS hard anti-proximity | completed | 76.29 | 0.2419 | failed |
| I6 | IRIS response-shuffled | completed | 80.30 | 0.1029 | response sanity |
| I7 | IRIS no anti-proximity | completed | 84.64 | 0.7783 | hard-filter removal helps |
| I8 | IRIS structural-signature-only | completed | 77.35 | 0.1124 | weak |
| I9 | IRIS no gradient-proxy | completed | 76.29 | 0.2419 | equivalent to I5 by design |
| I10 | R2 residualized response | completed | 84.46 | 0.7783 | rescue, not enough |
| I11 | raw response no residual | completed | 84.64 | 0.7783 | matches I7 |
| I12 | residual + soft proximity penalty | completed | 84.18 | 0.7601 | worse than I10/I11 |
| I13 | residual + CAST hybrid | completed | 84.87 | 0.7968 | best new variant, still not pilot-ready |

## Evidence Status

- Code implemented：yes，`scripts/run_iris_smoke.py`。
- Config implemented：yes，`configs/iris_smoke.yaml`。
- Smoke run：completed, Cora seed=0 only。
- Pilot run：no。
- Formal run：no。
- Claim support：mixed/negative smoke only；不支持性能提升 claim。
- Result summary：`results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.md`。
- Bridge results：`refine-logs/EXPERIMENT_RESULTS.md`。

## Decision Notes

Residualization is a useful correction to hard anti-proximity, but the current R2-IRIS variants remain dominated or matched by simpler controls. Do not escalate until the response component becomes both diagnostically distinct and competitive with kNN/PMGCL/CAST controls.
