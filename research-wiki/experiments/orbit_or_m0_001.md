# ORBIT-GCL OR-M0-001

- ID：`orbit_or_m0_001`
- 方法：`ORBIT-GCL / Operator-Response Basis Induction`
- 阶段：`smoke`
- Verdict：`no`
- Confidence：`medium`
- 时间：2026-06-26T17:44:25Z
- Dataset/seed：Cora / 0
- Split：stratified random `1:1:8`
- Evaluator：frozen encoder + Logistic Regression
- Claim status：smoke negative only；不支持性能提升 claim。

## Command

```bash
/root/miniconda3/bin/python scripts/run_orbit_smoke.py --config configs/orbit_smoke.yaml --dataset Cora --seed 0 --run-tag or_m0_001 --log-every 25
```

## Provenance

- Commit：`e563f5325e4b68138268c8d521ee555b6d8041b1`
- Dirty at start：`True`，原因是未跟踪 logs 目录。
- Raw JSON：`results/raw/Cora/ORBIT_GCL_SMOKE/orbit_smoke_Cora_seed0_20260626T174314Z/orbit_smoke_Cora_seed0_20260626T174314Z.json`
- Summary：`results/summary/orbit_smoke_Cora_seed0_20260626T174314Z_summary.md`
- Logs：`logs/orbit_smoke/orbit_smoke_Cora_seed0_20260626T174314Z/`

## Result

| ID | System | Test@best-val |
|---|---|---:|
| O0 | GRACE | 84.78 |
| O1 | matched-parameter GRACE | 84.96 |
| O3 | GCMAE-style MAE+CL hybrid | 81.96 |
| O4 | Graph-JEPA latent target | 81.87 |
| O7 | same-node response control | 81.92 |
| O12 | random teacher response basis | 80.49 |
| O14 | ORBIT full minimal | 82.98 |

## Decision

`KILL_OR_PIVOT_REQUIRED`

ORBIT full did not beat GRACE or matched GRACE:

- O14 - O0：`-1.7989`
- O14 - O1：`-1.9834`
- O14 - strongest matched control：`-1.9834`

Do not proceed to Pilot-A/B/formal. Do not continue with small hyperparameter or operator-bank tuning.
