# Experiment Code Review: IRIS-GCL Smoke

## Reviewed Files

- `refine-logs/EXPERIMENT_PLAN.md`
- `configs/iris_smoke.yaml`
- `scripts/run_iris_smoke.py`
- `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.json`

## Local Integrity Review

- Protocol gate is enforced in code: `run_iris_smoke.py` raises an error unless dataset is `Cora` and seed is `0`.
- Split is created/loaded through the existing project helper `create_or_load_split`, which stores `stratified_random_1_1_8` JSON splits under `splits/`.
- Evaluator uses existing `logreg_val_eval`: train fit, validation selection over `C`, and test reporting only.
- The smoke runner reuses GRACE encoder, optimizer, augmentation helpers, dataset loading, split loading, and evaluator helpers instead of creating a separate evaluator path.
- Offline labels are used only for diagnostics such as label agreement and false-negative mass; these diagnostics are marked as offline-only and are not fed into relation selection.
- I3 and I4 are explicitly declared as smoke-stage proxy/diagnostic implementations, not full PMGCL or full CAST implementations.

## External Review Note

An initial external review response was discarded because it contained file-inconsistent claims. A second review with file-reading requirement returned `BLOCKED` with reason `iris_failed_smoke_test_multiple_kill_rules_triggered`, matching the local result interpretation.

## Risks

- This is a relation-smoothed frozen-GRACE diagnostic smoke, not a full contrastive retraining implementation of IRIS.
- I4 is a CAST one-step proxy, so CAST dominance is a strong warning but not a full CAST-vs-IRIS final verdict.
- I9 is identical to I5 because the positive-gradient proxy was removed during proposal refinement; therefore this run cannot test any gradient-proxy contribution.
- Single seed and single dataset prohibit all robustness/generalization claims.

## Verdict

`PASS_AS_SMOKE_IMPLEMENTATION_WITH_NEGATIVE_RESULT`

The code is adequate for the planned Cora seed=0 smoke and produced parseable JSON/Markdown outputs. The method result is negative and triggers `PIVOT_REQUIRED`; it should not be escalated to pilot or formal runs without a mechanism change.

