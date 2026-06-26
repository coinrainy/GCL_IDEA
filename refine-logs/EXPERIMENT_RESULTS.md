# PACER-GCL PA-CER-M0-001 Smoke Results

- 时间：2026-06-26T16:32:36Z
- 状态：smoke only
- Dataset / seed：Cora / 0
- Result JSON：`results/raw/Cora/PACER_GCL_SMOKE/pacer_smoke_Cora_seed0_20260626T163236Z/pacer_smoke_Cora_seed0_20260626T163236Z.json`
- Summary：`results/summary/pacer_smoke_Cora_seed0_20260626T163236Z_summary.md`
- Decision：`KILL_OR_PIVOT_REQUIRED`

P3 PACER full 为 `83.63`，低于 P0 GRACE `84.69`、P5 random route `84.92`、P8 SupCon control `85.15`，且与 P4 shuffled-label probe 持平。虽然 P3 的 train/val probe margin delta 为正，但未转化为节点分类 accuracy。

Triggered rules：`KILL_PACER`、`KILL_LABEL_CALIBRATION_STORY`、`KILL_ROUTING_STORY`、`KILL_OBJECTIVE_ROUTING_STORY`、`KILL_PACER_AS_UNNECESSARY`。

不进入 PA-CER-M1、Pilot、formal 或 `/auto-review-loop`；不支持性能提升 claim。
