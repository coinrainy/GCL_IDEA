# ORBIT-GCL OR-M0-001 Experiment Results

- 时间：2026-06-26T17:44:25Z
- Plan：`refine-logs/EXPERIMENT_PLAN.md`
- Run：`orbit_smoke_Cora_seed0_20260626T174314Z`
- Dataset/seed：Cora / 0
- Split：stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression，train/val select `C`，final test report
- Command：`/root/miniconda3/bin/python scripts/run_orbit_smoke.py --config configs/orbit_smoke.yaml --dataset Cora --seed 0 --run-tag or_m0_001 --log-every 25`
- Commit：`e563f5325e4b68138268c8d521ee555b6d8041b1`
- Dirty at start：`True`（未跟踪 logs 目录）
- Claim status：Cora seed0 smoke only；不支持 Pilot/Formal/SOTA claim。

## Summary

ORBIT full minimal 未通过 kill-smoke。O14 ORBIT full 为 `82.98`，低于 O0 GRACE `84.78` 和 O1 matched-parameter GRACE `84.96`。虽然 ORBIT 超过 GraphMAE2-lite、GCMAE-style、Graph-JEPA、D-SLA、same-node response、raw response 等多个 controls，但预注册硬规则要求超过最强 matched control；该条件未满足。

Decision：`KILL_OR_PIVOT_REQUIRED`

Triggered rules：

- `BLOCK_PROVENANCE_BEFORE_PILOT`
- `KILL_ORBIT_AS_WEAKER_THAN_MATCHED_CONTROL`
- `KILL_ACCURACY_NOT_CONVERTED`

## Main Table

| ID | System | Test@best-val | Val@best | Heldout response MSE |
|---|---|---:|---:|---:|
| O0 | GRACE | 84.78 | 85.19 | 0.9946 |
| O1 | matched-parameter GRACE | 84.96 | 85.56 | 1.0546 |
| O2 | GraphMAE2-lite latent target | 66.84 | 66.30 | 423.6406 |
| O3 | GCMAE-style MAE+CL hybrid | 81.96 | 81.48 | 1.1339 |
| O4 | Graph-JEPA latent target | 81.87 | 82.22 | 1.4236 |
| O5 | TTER-style operator-id prediction | 33.72 | 35.93 | 1.6405 |
| O6 | D-SLA-style discrepancy regression | 70.76 | 78.52 | 0.8441 |
| O7 | same-node response control | 81.92 | 81.85 | 1.6741 |
| O8 | ORBIT response-magnitude-only | 75.18 | 76.67 | 0.8466 |
| O9 | ORBIT degree/feature-only response | 64.81 | 65.93 | 0.7710 |
| O10 | random operator labels | 69.60 | 71.85 | 0.9951 |
| O11 | shuffled response basis | 69.28 | 71.85 | 0.8019 |
| O12 | random teacher response basis | 80.49 | 81.11 | 0.6403 |
| O13 | no-SVD raw response | 53.27 | 54.81 | 287.0751 |
| O14 | ORBIT full minimal | 82.98 | 85.19 | 0.6069 |

## Control Gaps

- O14 - O0 GRACE：`-1.7989`
- O14 - O1 matched GRACE：`-1.9834`
- O14 - strongest matched control：`-1.9834`
- O14 - shortcut best：`2.4908`
- O14 - GCMAE-style：`1.0148`
- O14 - Graph-JEPA：`1.1070`
- O14 - same-node response control：`1.0609`

## Interpretation

ORBIT 的 response-basis target 有一定信号：它超过 GCMAE-style、Graph-JEPA 和 same-node response controls，并且 held-out response MSE 较低。但该信号不足以超过 GRACE/matched GRACE，因此不能支持继续 Pilot-A。按项目纪律，当前 ORBIT 不能通过调 operator bank、basis rank、loss 权重或 teacher 继续小修；若要继续，必须重新定义机制后再进入新的 idea/refine 阶段。

## Paths

- Raw JSON：`results/raw/Cora/ORBIT_GCL_SMOKE/orbit_smoke_Cora_seed0_20260626T174314Z/orbit_smoke_Cora_seed0_20260626T174314Z.json`
- Summary MD：`results/summary/orbit_smoke_Cora_seed0_20260626T174314Z_summary.md`
- Summary JSON：`results/summary/orbit_smoke_Cora_seed0_20260626T174314Z_summary.json`
- Logs：`logs/orbit_smoke/orbit_smoke_Cora_seed0_20260626T174314Z/`

## Ready for Auto Review

`NO`

当前是 negative smoke，不应进入 `/auto-review-loop`、Pilot-A/B 或 formal。
