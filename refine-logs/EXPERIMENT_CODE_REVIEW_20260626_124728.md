# Experiment Code Review: Certified CAST Closure Smoke

## Reviewed Files

- `configs/iris_smoke.yaml`
- `scripts/run_iris_smoke.py`
- `results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.json`

## Integrity Notes

- The Cora seed=0 gate remains enforced.
- Split/evaluator protocol remains unchanged: project `1:1:8` split and frozen encoder + Logistic Regression evaluator.
- I22 uses CAST candidates plus label-free residual-response certificate thresholds.
- Labels are only used after relation construction for diagnostics.
- I22 uses positive relations for smoothing and positive+neutral closure for false-negative mass diagnostics.

## Verdict

`PASS_AS_SMOKE_IMPLEMENTATION_WITH_NEGATIVE_RESULT`

The implementation is adequate for the allowed smoke. The mechanism result is negative and should not be escalated.

