# Experiment Tracker: PACER-GCL

- 更新时间：2026-06-26T16:06:34Z
- Active idea：PACER-GCL
- Current decision：`GO_TO_PACER_SMOKE_PLANNING_WITH_CAUTION`
- Current allowed scope：Cora seed0 smoke only
- Fresh auditor note：`WARN`; no pilot if exact run command is missing or `project_dirty=true`.

| ID | Stage | Task | Dataset/seeds | Status | Gate |
|---|---|---|---|---|---|
| PA-CER-M0-001 | smoke | implement and run PACER variants P0-P8 | Cora seed0 | TODO | required before any pilot |
| PA-CER-M0-AUDIT | audit | code/integrity review and leakage check | Cora seed0 outputs | TODO | required before interpreting smoke |
| PA-CER-M1-CORA | pilot | Cora seeds 0-2 | Cora seeds 0-2 | BLOCKED | blocked until M0 GO |
| PA-CER-M1-CITESEER | pilot | CiteSeer seeds 0-2 | CiteSeer seeds 0-2 | BLOCKED | blocked until Cora signal |
| PA-CER-FORMAL | formal | 10 seeds paper table | all datasets | BLOCKED | not allowed |

## Blocked Lines

- CAST C4/C5 continuation：blocked。
- BEACON gate tuning：blocked。
- SPECTRA boundary/rank tuning：blocked。
- Pair mining / positive mining / score fusion / relation smoothing：blocked。
- Claiming unsupervised GCL：blocked; PACER is semi-supervised / train-label-calibrated。
