# Experiment Code Review: R2-IRIS Smoke

## Reviewed Files

- `configs/iris_smoke.yaml`
- `scripts/run_iris_smoke.py`
- `refine-logs/EXPERIMENT_PLAN.md`
- `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.json`

## Local Integrity Review

- The code still enforces the Cora seed=0 gate.
- Split/evaluator path remains unchanged from the project GRACE helper: stratified random `1:1:8`, frozen encoder, Logistic Regression with `C` selected on train/validation.
- Residualized response selection does not use labels. Labels are used only for offline diagnostics after relations are selected.
- I10-I13 are smoke-stage diagnostic variants. They are not a full second-stage contrastive retraining implementation.
- The residualizer controls feature similarity, embedding similarity, diffusion/PPR-style graph proximity, and degree gap before ranking pair residuals.

## Risks

- I10 currently matches raw response/no-anti-proximity behavior, suggesting residualization does not yet create a meaningfully distinct selector.
- I13 improves label agreement over CAST proxy but remains below CAST/PMGCL on accuracy and below kNN on label agreement.
- The runner still uses relation-smoothed frozen embeddings; this is appropriate for smoke but not a final method implementation.
- `results/raw` and `results/summary` are ignored by Git per project policy, so raw local results must be preserved locally and referenced by paths.

## Verdict

`PASS_AS_SMOKE_IMPLEMENTATION_WITH_MIXED_RESULT`

The R2 implementation is valid for the allowed smoke scope, but the method decision is `REVISE_NOT_PILOT`.

