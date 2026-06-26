# Experiment Code Review: CAST Certificate Smoke

## Reviewed Files

- `configs/cast_certificate_smoke.yaml`
- `scripts/run_cast_certificate_smoke.py`
- `results/summary/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z_summary.json`

## Integrity Notes

- The script enforces Cora seed=0 for this smoke gate.
- It reuses project split and Logistic Regression evaluator helpers.
- The certificate is trained without labels: ego summaries predict frozen GRACE target embeddings of graph-neighbor targets.
- Relation construction uses certificate energy and CAST/candidate controls; labels are used only for offline diagnostics.
- Result JSON records `project_dirty=true` because the run creates untracked log files before writing results; code/config were committed first at `39c2707`.

## Verdict

`PASS_AS_SMOKE_IMPLEMENTATION_WITH_POSITIVE_SIGNAL`

This supports Pilot-A planning, not formal claims.

