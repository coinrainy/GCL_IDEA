# Experiment Tracker: IRIS / R2-IRIS / CPR-IRIS

## Current Gate

- Status：`CERTIFIED_CLOSURE_SMOKE_COMPLETED_NEGATIVE`
- Decision：`KILL_CERTIFIED_CLOSURE_KEEP_CPR_SCORE`
- Completed action：implemented one certified CAST closure variant I22 and ran Cora seed=0 smoke only。
- Forbidden：multi-dataset pilot、formal 10-seed run、SOTA/performance claim、继续当前 I22 neutral-closure 机制。

## Smoke Matrix

| ID | Variant | Status | Test@best-val | Label agreement | Decision |
|---|---|---|---:|---:|---|
| I4 | CAST one-step proxy | completed | 85.70 | 0.7548 | strongest accuracy |
| I17 | CAST + residual cert w=0.30 | completed | 85.56 | 0.7953 | best surviving CPR score |
| I18 | CAST + residual cert w=0.15 | completed | 85.56 | 0.7817 | close to I17 |
| I19 | CAST + residual cert w=0.45 | completed | 85.42 | 0.8021 | higher agreement, lower accuracy |
| I20 | kNN + residual cert w=0.15 | completed | 85.15 | 0.8120 | near kNN agreement |
| I22 | certified CAST closure | completed | 84.96 | 0.7472 | killed |

## Evidence Status

- Code implemented：yes，`scripts/run_iris_smoke.py`。
- Config implemented：yes，`configs/iris_smoke.yaml`。
- Smoke run：completed, Cora seed=0 only。
- Pilot run：no。
- Formal run：no。
- Claim support：negative/mixed smoke only；不支持性能提升 claim。
- Result summary：`results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.md`。
- Bridge results：`refine-logs/EXPERIMENT_RESULTS.md`。

## Decision Notes

Certified closure is distinct from kNN/CAST but weaker. The only CPR signal worth preserving is additive response-certified CAST scoring. The overall IRIS/CPR line is near exhaustion; further work should either make a substantially new mechanism or pivot back to CAST/restart idea discovery.
