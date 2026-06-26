# Experiment Tracker: PACER-GCL

- 更新时间：2026-06-26T16:32:36Z
- Active idea：PACER-GCL
- Current decision：`KILL_OR_PIVOT_REQUIRED`
- Current allowed scope：stop PACER; no PA-CER-M1 / pilot / formal
- Method status：train-label-calibrated / semi-supervised GCL, not unsupervised

| ID | Stage | Task | Dataset/seeds | Status | Gate |
|---|---|---|---|---|---|
| PA-CER-M0-001 | smoke | implement and run PACER variants P0-P8 | Cora seed0 | DONE_NEGATIVE | blocks any pilot |
| PA-CER-M0-AUDIT | audit | code/integrity review and leakage check | Cora seed0 outputs | DONE | no code-level blocker after fixes |
| PA-CER-M1-CORA | pilot | Cora seeds 0-2 | Cora seeds 0-2 | BLOCKED | blocked by M0 no-go |
| PA-CER-M1-CITESEER | pilot | CiteSeer seeds 0-2 | CiteSeer seeds 0-2 | BLOCKED | blocked by M0 no-go |
| PA-CER-FORMAL | formal | 10 seeds paper table | all datasets | BLOCKED | not allowed |

## M0 Result

- P0 GRACE：`84.69`
- P3 PACER full：`83.63`
- P4 shuffled-label probe：`83.63`
- P5 random route：`84.92`
- P6 scalar reweight：`83.76`
- P8 SupCon control：`85.15`

Triggered rules：`KILL_PACER`、`KILL_LABEL_CALIBRATION_STORY`、`KILL_ROUTING_STORY`、`KILL_OBJECTIVE_ROUTING_STORY`、`KILL_PACER_AS_UNNECESSARY`。

## Blocked Lines

- PACER threshold/weight/routing small tuning：blocked。
- PA-CER-M1/Pilot/formal：blocked。
- Claiming unsupervised GCL：blocked; PACER is semi-supervised / train-label-calibrated。
- Performance improvement claim：blocked; smoke result is negative。
