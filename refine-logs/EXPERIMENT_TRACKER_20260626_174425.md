# Experiment Tracker: ORBIT-GCL

- 时间：2026-06-26T17:44:25Z
- 当前阶段：`SMOKE_NEGATIVE`
- Decision：`KILL_OR_PIVOT_REQUIRED`
- Claim status：Cora seed0 smoke negative；不得引用为性能提升证据。

## Gates

| Gate | Status | Requirement |
|---|---|---|
| OR-M0 implementation | done | `configs/orbit_smoke.yaml` and `scripts/run_orbit_smoke.py` |
| OR-M0 dryrun | done | 1-epoch Cora seed0 dryrun passed output schema |
| OR-M0 full smoke | done_negative | O14 ORBIT full `82.98`, below GRACE `84.78` and matched GRACE `84.96` |
| Pilot-A | blocked | OR-M0 failed hard gates |
| Pilot-B | blocked | no path after negative smoke |
| Formal | blocked | no path after negative smoke |

## Runs

| ID | Dataset | Seed | Type | Status |
|---|---|---:|---|---|
| OR-M0-000 | Cora | 0 | dryrun | done |
| OR-M0-001 | Cora | 0 | kill-smoke | done_negative |

## Hard Stop Conditions Triggered

- `KILL_ORBIT_AS_WEAKER_THAN_MATCHED_CONTROL`
- `KILL_ACCURACY_NOT_CONVERTED`
- `BLOCK_PROVENANCE_BEFORE_PILOT`

## Notes

OR-M0-001 did not pass. Do not expand to Cora seeds 1-2, CiteSeer, PubMed, Amazon, or Wiki-CS. Do not tune operator bank, basis rank, teacher source, or loss weights as a small repair; a new mechanism-level idea/refine step is required before any further ORBIT-like run.
