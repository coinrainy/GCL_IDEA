# BEACON-GCL Smoke Code Review

- 时间：2026-06-26T15:33:38Z
- 范围：`configs/beacon_smoke.yaml`、`scripts/run_beacon_smoke.py`
- 审查类型：executor-local checklist；本轮未启用 fresh subagent reviewer。
- 阶段：BE-M0 implementation dryrun；不支持 formal、SOTA 或 robust claim。

## Implementation Checklist

- 新增独立 runner，没有修改 CAST runner 主线。
- 仅允许 `Cora seed=0`，符合 BE-M0 gate。
- 复用项目 split：`stratified_random_1_1_8`。
- 复用 frozen GRACE encoder + Logistic Regression evaluator。
- C4 certificate 训练不使用标签；label agreement 仅作为 offline diagnostic。
- BEACON gate 使用 train/val 校准的 Logistic Regression probe margin，不使用 test label 选阈值。
- 输出 B0-B8：GRACE、kNN、CAST proxy、C4、all-candidate、BEACON full、shuffled、no-boundary、no-geometry。
- 输出 accepted overlap、probe-margin delta、effective-rank delta、uniformity delta、degree-bucket concentration、utility accepted/rejected。

## Dryrun

命令：

```bash
python scripts/run_beacon_smoke.py --dataset Cora --seed 0 --epochs 1 --cert-epochs 1 --run-tag be_m0_dryrun --status smoke --log-every 1
```

结果：

- Summary JSON: `results/summary/be_m0_dryrun_Cora_seed0_20260626T153303Z_summary.json`
- Summary MD: `results/summary/be_m0_dryrun_Cora_seed0_20260626T153303Z_summary.md`
- Logs: `logs/beacon_smoke/be_m0_dryrun_Cora_seed0_20260626T153303Z/`
- `gate_uses_test_labels = false`
- B0-B8 均成功产生 evaluation 与 diagnostics。

## Risks

- BEACON-lite 当前使用 post-hoc relation smoothing 近似 neutralization，不等于最终训练期 denominator edit。
- Boundary probe 使用 train/val 标签校准，应在叙事中标明为 train/val-calibrated smoke，不可伪装成纯无监督方法。
- Dryrun 只验证链路，不能用于任何方法结论。

## Decision

`GO_TO_BE_M0_FULL_SMOKE`
