# Experiment Tracker: SPECTRA-GCL

- 更新时间：2026-06-26T15:59:06Z
- Active idea：SPECTRA-GCL
- Current decision：`KILL_BOUNDARY_ENERGY_STORY`
- Current allowed scope：no further SPECTRA runs unless idea is explicitly reactivated with a new mechanism.

| ID | Stage | Task | Dataset/seeds | Status | Gate |
|---|---|---|---|---|---|
| SP-M0-001 | smoke | implement and run SPECTRA smoke variants S0-S7 | Cora seed0 | DONE_NEGATIVE | `KILL_BOUNDARY_ENERGY_STORY` |
| SP-M0-AUDIT | audit | code/integrity review of runner and diagnostics | Cora seed0 outputs | DONE | local-only review passed, result negative |
| SP-M1-CORA | pilot | Cora seeds 0-2 only if SP-M0 GO | Cora seeds 0-2 | BLOCKED | SP-M0 no-go |
| SP-M1-CITESEER | pilot | CiteSeer seeds 0-2 only if Cora signal passes | CiteSeer seeds 0-2 | BLOCKED | SP-M0 no-go |
| SP-M1-PUBMED | pilot | PubMed seed0 sanity then seeds 0-2 | PubMed seeds 0-2 | BLOCKED | SP-M0 no-go |
| SP-FORMAL | formal | 10 seeds and paper table | all datasets | BLOCKED | no formal claim allowed |

## Result Pointers

- Summary：`results/summary/sp_m0_001_Cora_seed0_20260626T155803Z_summary.md/json`
- Raw：`results/raw/Cora/SPECTRA_GCL_SMOKE/sp_m0_001_Cora_seed0_20260626T155803Z/`
- Logs：`logs/spectra_smoke/sp_m0_001_Cora_seed0_20260626T155803Z/`
- Result report：`refine-logs/EXPERIMENT_RESULTS_20260626_155906.md`
- Code review：`refine-logs/EXPERIMENT_CODE_REVIEW_20260626_155906.md`

## Blocked Lines

- SPECTRA weight/threshold tuning：blocked by diagnostic-improves-but-accuracy-drops pattern。
- SP-M1 multi-seed pilot：blocked。
- Formal experiment and paper table：blocked。

