# SPECTRA-GCL SP-M0-001 Smoke Results

- 时间：2026-06-26T15:59:06Z
- Skill：`/experiment-bridge`
- 阶段：smoke，不是 pilot/formal。
- 数据集：Cora seed0，stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression train/val `C` selection
- 配置：`configs/spectra_smoke.yaml`
- 代码：`scripts/run_spectra_smoke.py`
- Summary：`results/summary/sp_m0_001_Cora_seed0_20260626T155803Z_summary.md/json`
- Raw：`results/raw/Cora/SPECTRA_GCL_SMOKE/sp_m0_001_Cora_seed0_20260626T155803Z/`
- Logs：`logs/spectra_smoke/sp_m0_001_Cora_seed0_20260626T155803Z/`

## Results

| ID | Variant | Test@best-val | Val | Eff rank | Boundary retention | Cov offdiag | Alignment | Uniformity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | GRACE | 84.78 | 85.19 | 64.57 | 0.9279 | 0.2465 | 0.0653 | -0.9995 |
| S1 | negative_free_rr | 83.58 | 86.30 | 74.16 | 0.9273 | 0.0902 | 0.1199 | -0.9687 |
| S2 | spectra_full | 82.47 | 84.07 | 76.56 | 0.9294 | 0.1008 | 0.1175 | -0.9622 |
| S3 | no_boundary_energy | 82.61 | 81.85 | 68.35 | 0.9256 | 0.0767 | 0.1026 | -0.8929 |
| S4 | no_rank_guard | 82.20 | 82.59 | 65.24 | 0.9261 | 0.0721 | 0.1130 | -0.9677 |
| S5 | random_spectrum_guard | 81.60 | 82.22 | 67.67 | 0.9261 | 0.0786 | 0.1310 | -1.0822 |
| S6 | low_band_only | 81.37 | 82.22 | 72.84 | 0.9259 | 0.0863 | 0.1467 | -1.1831 |
| S7 | matched_parameter_GRACE | 84.78 | 85.19 | 64.57 | 0.9279 | 0.2465 | 0.0653 | -0.9995 |

## Reading

- SPECTRA full does improve some intended diagnostics over the negative-free baseline: effective rank `76.56` vs `74.16`, boundary retention `0.9294` vs `0.9273`, and uniformity `-0.9622` vs `-0.9687`.
- That diagnostic improvement does not convert to accuracy. S2 `82.47` is below S1 `83.58` and GRACE/S7 `84.78`.
- S3 no-boundary-energy is slightly above S2 on accuracy (`82.61` vs `82.47`), so the boundary-energy term is not helping the downstream classifier in this smoke.
- S5 random-spectrum guard is weak on accuracy, but its boundary retention `0.9261` remains close enough that the current boundary-retention diagnostic is not a strong causal proof.
- This repeats the CAST/BEACON warning pattern: a more favorable diagnostic does not automatically imply better node classification.

## Integrity Notes

- S1-S6 use same-node two-view negative-free objectives only.
- No kNN/PPR/CAST mining, positive mining, score fusion, or relation smoothing is used.
- Test labels are not used for thresholding, training, early stopping, or evaluator hyperparameter selection.
- Label-only diagnostics are saved after training in JSON only for analysis.
- Two dryruns were intentionally failed/used to fix implementation integrity before the successful run:
  - YAML colon parsing in `description` was fixed.
  - `sqrt(0)` in boundary residual energy was stabilized with epsilon after NaN embeddings were caught by LogReg.

## Decision

`KILL_BOUNDARY_ENERGY_STORY`

Do not run SP-M1 Cora/CiteSeer/PubMed, Pilot-B, formal experiments, paper tables, or `/auto-review-loop` from this state. Do not continue by tuning SPECTRA thresholds/weights. The useful next action is another mechanism-level pivot, with the added evidence that negative-free boundary-energy diagnostics also failed to convert into accuracy on Cora seed0.

No performance claim is supported.

