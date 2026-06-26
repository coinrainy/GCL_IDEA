# PACER-GCL PA-CER-M0-001 Smoke Results

- 时间：2026-06-26T16:32:36Z
- Skill：`/experiment-bridge`
- 状态：smoke only
- Dataset / seed：Cora / 0
- Split：stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression，`C` 由 train/val 选择，test 仅在选择后报告
- Method status：PACER 是 train-label-calibrated / semi-supervised GCL，不是无监督 GCL
- Result JSON：`results/raw/Cora/PACER_GCL_SMOKE/pacer_smoke_Cora_seed0_20260626T163236Z/pacer_smoke_Cora_seed0_20260626T163236Z.json`
- Summary：`results/summary/pacer_smoke_Cora_seed0_20260626T163236Z_summary.md`
- Logs：`logs/pacer_smoke/pacer_smoke_Cora_seed0_20260626T163236Z/`
- Exact command：`/root/miniconda3/bin/python scripts/run_pacer_smoke.py --config configs/pacer_smoke.yaml --dataset Cora --seed 0 --run-tag pa_cer_m0_001 --log-every 25`
- Commit：`81a3fc34fc6d13c5aada3fb571197097ea2c579f`
- Dirty at start：`False`

## Results

| ID | System | Test@best-val | Val@best | C | Train margin delta | Val margin delta | Fragile fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | GRACE | 84.69 | 85.56 | 128.0 | 0.4163 | 0.3275 | 0.0000 |
| P1 | probe-only diagnostic | 83.44 | 84.44 | 512.0 | 0.0000 | 0.0000 | 0.0000 |
| P2 | mask-only consistency | 74.45 | 75.93 | 512.0 | -1.2427 | -1.1885 | 0.0000 |
| P3 | PACER full | 83.63 | 85.19 | 256.0 | 1.3903 | 1.3004 | 0.3955 |
| P4 | shuffled-label probe | 83.63 | 85.56 | 256.0 | 0.2755 | 0.2483 | 0.4317 |
| P5 | random-probe route | 84.92 | 85.56 | 512.0 | 1.4529 | 1.5783 | 0.3955 |
| P6 | scalar reweight control | 83.76 | 85.56 | 512.0 | 0.4637 | 0.3998 | 0.3955 |
| P7 | train-only CV probe | 83.90 | 85.19 | 512.0 | 0.2521 | 0.3813 | 0.4302 |
| P8 | supervised-contrastive small-label control | 85.15 | 86.30 | 4.0 | 1.9664 | 1.3914 | 0.0000 |

## Control Gaps

- `p3_minus_p0_grace`: -1.0609
- `p3_minus_p2_mask_only`: +9.1790
- `shuffled_control_gap_p3_minus_p4`: 0.0000
- `random_route_control_gap_p3_minus_p5`: -1.2915
- `scalar_control_gap_p3_minus_p6`: -0.1384
- `train_cv_control_gap_p3_minus_p7`: -0.2768
- `supcon_control_gap_p3_minus_p8`: -1.5221

## Diagnostics

P3 的 train / val probe margin delta 明显为正，说明 probe-aligned routing 确实改变了读出方向；但 accuracy 未超过 GRACE、random-route、scalar-reweight 和 SupCon control。该结果说明本轮 train-label calibrated routing 的代理改善没有转化为节点分类收益。

## Decision

`KILL_OR_PIVOT_REQUIRED`

触发规则：

- `KILL_PACER`
- `KILL_LABEL_CALIBRATION_STORY`
- `KILL_ROUTING_STORY`
- `KILL_OBJECTIVE_ROUTING_STORY`
- `KILL_PACER_AS_UNNECESSARY`

不进入 PA-CER-M1、Pilot、formal 或 `/auto-review-loop`。该 smoke 不支持任何性能提升 claim。
