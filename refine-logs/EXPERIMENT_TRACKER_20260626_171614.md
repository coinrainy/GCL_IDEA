# Experiment Tracker: ORBIT-GCL

- 时间：2026-06-26T17:16:14Z
- 当前阶段：`PLANNED_ONLY`
- Decision：`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`
- Claim status：无 ORBIT 实验结果；不得引用为性能证据。

## Gates

| Gate | Status | Requirement |
|---|---|---|
| OR-M0 implementation | pending | `configs/orbit_smoke.yaml` and `scripts/run_orbit_smoke.py` |
| OR-M0 dryrun | pending | Cora seed0 short epochs, output schema valid |
| OR-M0 full smoke | pending | Cora seed0 full smoke with O0-O14 controls |
| Pilot-A | blocked | only if OR-M0 passes all hard gates |
| Pilot-B | blocked | no path before Pilot-A positive |
| Formal | blocked | no path before fixed protocol + 10 seeds |

## Planned Runs

| ID | Dataset | Seed | Type | Status |
|---|---|---:|---|---|
| OR-M0-000 | Cora | 0 | dryrun | pending |
| OR-M0-001 | Cora | 0 | kill-smoke | pending |

## Hard Stop Conditions

- ORBIT full does not beat strongest matched control.
- Random/shuffled/shortcut response controls match ORBIT full.
- Held-out operator response prediction fails.
- Accuracy does not improve despite response diagnostics improving.
- Any evaluator leakage or test-set tuning is detected.

## Notes

Do not expand to Cora seeds 1-2, CiteSeer, PubMed, Amazon, or Wiki-CS before OR-M0-001 passes. Do not tune operator bank using test accuracy.
