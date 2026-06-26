# Experiment Code Review: CPR-IRIS Weight Smoke

## Reviewed Files

- `configs/iris_smoke.yaml`
- `scripts/run_iris_smoke.py`
- `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.json`

## Integrity Notes

- The Cora seed=0 gate remains enforced.
- Split/evaluator protocol remains unchanged: project `1:1:8` split and frozen encoder + Logistic Regression evaluator.
- I14-I21 use residual response as a label-free certificate or reranker over kNN/CAST-style candidates.
- Labels are only used after relation construction for diagnostics.
- This remains relation-smoothed frozen embedding smoke, not a full method implementation.

## Verdict

`PASS_AS_SMOKE_IMPLEMENTATION_WITH_MIXED_RESULT`

The smoke evidence supports continued mechanism refinement toward certified CAST/proximity closure, but not pilot or formal scaling.

