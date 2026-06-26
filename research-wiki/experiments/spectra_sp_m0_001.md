# SPECTRA SP-M0-001

- ID：`spectra_sp_m0_001`
- Idea：`spectra_gcl`
- 时间：2026-06-26T15:59:06Z
- 阶段：smoke
- Verdict：`no`
- Decision：`KILL_BOUNDARY_ENERGY_STORY`
- Dataset：Cora seed0
- Split：stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression train/val `C` selection
- Summary：`results/summary/sp_m0_001_Cora_seed0_20260626T155803Z_summary.md/json`
- Result report：`refine-logs/EXPERIMENT_RESULTS_20260626_155906.md`

## Result Table

| ID | Variant | Test@best-val | Val | Effective rank | Boundary retention | Uniformity |
|---|---|---:|---:|---:|---:|---:|
| S0 | GRACE | 84.78 | 85.19 | 64.57 | 0.9279 | -0.9995 |
| S1 | negative_free_rr | 83.58 | 86.30 | 74.16 | 0.9273 | -0.9687 |
| S2 | spectra_full | 82.47 | 84.07 | 76.56 | 0.9294 | -0.9622 |
| S3 | no_boundary_energy | 82.61 | 81.85 | 68.35 | 0.9256 | -0.8929 |
| S4 | no_rank_guard | 82.20 | 82.59 | 65.24 | 0.9261 | -0.9677 |
| S5 | random_spectrum_guard | 81.60 | 82.22 | 67.67 | 0.9261 | -1.0822 |
| S6 | low_band_only | 81.37 | 82.22 | 72.84 | 0.9259 | -1.1831 |
| S7 | matched_parameter_GRACE | 84.78 | 85.19 | 64.57 | 0.9279 | -0.9995 |

## Interpretation

S2 SPECTRA full improves effective rank and boundary-retention diagnostics over S1 negative-free RR, but is worse on downstream node-classification accuracy. This repeats the broader no-go pattern from CAST and BEACON: diagnostics that look closer to the intended mechanism do not necessarily improve Logistic Regression accuracy.

## Integrity

- No test-label thresholding.
- No kNN/PPR/CAST mining.
- No cross-node positive mining.
- No post-hoc relation smoothing.
- Dryrun caught and fixed YAML parsing and boundary-energy NaN issues before final smoke.

## Outcome

`no`

Blocked follow-up：SP-M1, Pilot-B, formal experiments, paper table, `/auto-review-loop`, and SPECTRA weight tuning.

