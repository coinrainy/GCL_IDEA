# Experiment Tracker: MEGA-GCL

- 更新时间：2026-06-26T16:40:00Z
- Active idea：MEGA-GCL
- Current decision：`REVISE_TO_MEGA_CORA_KILL_SMOKE`
- Current allowed scope：Cora seed0 kill-smoke only

| ID | Stage | Task | Dataset/seeds | Status | Gate |
|---|---|---|---|---|---|
| ME-M0-001 | kill-smoke | implement and run MEGA variants M0-M13 | Cora seed0 | TODO | required before any pilot |
| ME-M0-AUDIT | audit | code/integrity review and leakage check | Cora seed0 outputs | TODO | required before interpreting smoke |
| ME-M1-CORA | pilot | Cora seeds 0-2 | Cora seeds 0-2 | BLOCKED | blocked until M0 GO |
| ME-M1-CITESEER | pilot | CiteSeer seeds 0-2 | CiteSeer seeds 0-2 | BLOCKED | blocked until Cora signal |
| ME-FORMAL | formal | 10 seeds paper table | all datasets | BLOCKED | not allowed |

## Blocked Lines

- CAST C4/C5 continuation：blocked。
- BEACON gate tuning：blocked。
- SPECTRA boundary/rank tuning：blocked。
- PACER probe routing/threshold/weight tuning：blocked。
- kNN/PPR/CAST mining、positive mining、score fusion、post-hoc smoothing：blocked。
- Performance claim：blocked until formal results exist。
