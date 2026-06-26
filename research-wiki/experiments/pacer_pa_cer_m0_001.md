# PACER PA-CER-M0-001

- 日期：2026-06-26T16:32:36Z
- 阶段：smoke
- Verdict：`no`
- Decision：`KILL_OR_PIVOT_REQUIRED`
- Dataset / seed：Cora / 0
- Split：stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression
- Method status：train-label-calibrated / semi-supervised GCL
- Raw JSON：`results/raw/Cora/PACER_GCL_SMOKE/pacer_smoke_Cora_seed0_20260626T163236Z/pacer_smoke_Cora_seed0_20260626T163236Z.json`
- Summary：`results/summary/pacer_smoke_Cora_seed0_20260626T163236Z_summary.md`

## Result

| ID | System | Test@best-val |
|---|---|---:|
| P0 | GRACE | 84.69 |
| P1 | probe-only diagnostic | 83.44 |
| P2 | mask-only consistency | 74.45 |
| P3 | PACER full | 83.63 |
| P4 | shuffled-label probe | 83.63 |
| P5 | random-probe route | 84.92 |
| P6 | scalar reweight control | 83.76 |
| P7 | train-only CV probe | 83.90 |
| P8 | supervised-contrastive small-label control | 85.15 |

P3 improves train/val probe margin diagnostics but does not improve accuracy. It fails against GRACE, random routing, scalar reweighting, and SupCon control; shuffled-label probe matches P3.

## Triggered Rules

- `KILL_PACER`
- `KILL_LABEL_CALIBRATION_STORY`
- `KILL_ROUTING_STORY`
- `KILL_OBJECTIVE_ROUTING_STORY`
- `KILL_PACER_AS_UNNECESSARY`

## Claim Status

No performance claim. No Pilot-A/B. No formal experiment. PACER should not be continued by threshold/weight/routing small tuning.
