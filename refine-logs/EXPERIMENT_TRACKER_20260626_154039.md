# Experiment Tracker: SPECTRA-GCL

- 更新时间：2026-06-26T15:40:39Z
- Active idea：SPECTRA-GCL
- Current decision：`GO_TO_SPECTRA_SMOKE_PLANNING`
- Current allowed scope：Cora seed0 smoke only

| ID | Stage | Task | Dataset/seeds | Status | Gate |
|---|---|---|---|---|---|
| SP-M0-001 | smoke | implement and run SPECTRA smoke variants S0-S7 | Cora seed0 | TODO | required before any pilot |
| SP-M0-AUDIT | audit | code/integrity review of runner and diagnostics | Cora seed0 outputs | TODO | required before interpreting smoke |
| SP-M1-CORA | pilot | Cora seeds 0-2 only if SP-M0 GO | Cora seeds 0-2 | BLOCKED | blocked until SP-M0 GO |
| SP-M1-CITESEER | pilot | CiteSeer seeds 0-2 only if Cora signal passes | CiteSeer seeds 0-2 | BLOCKED | blocked until SP-M1-CORA |
| SP-M1-PUBMED | pilot | PubMed seed0 sanity then seeds 0-2 | PubMed seeds 0-2 | BLOCKED | blocked until Cora/CiteSeer support |
| SP-FORMAL | formal | 10 seeds and paper table | all datasets | BLOCKED | not allowed from current evidence |

## Blocked Lines

- CAST C4/C5 continuation：blocked by CAST Pilot-A no-go。
- BEACON current gate tuning：blocked by BE-M0 no-go。
- Any pair mining / positive mining / relation smoothing variant：blocked by user hard constraint。
- Any SOTA/formal claim：blocked until formal protocol and result-to-claim review。

## Required SP-M0 Outputs

- `configs/spectra_smoke.yaml`
- `scripts/run_spectra_smoke.py`
- `results/summary/*spectra*summary.md/json`
- `refine-logs/EXPERIMENT_RESULTS_<timestamp>.md`
- `refine-logs/EXPERIMENT_CODE_REVIEW_<timestamp>.md`

