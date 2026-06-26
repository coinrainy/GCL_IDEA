# PACER-GCL Experiment Code Review

- 时间：2026-06-26T16:32:36Z
- Reviewer：fresh `gcl_experiment_auditor`
- Scope：`configs/pacer_smoke.yaml`、`scripts/run_pacer_smoke.py`、`scripts/run_grace_1_1_8.py`

## Initial Blocking Findings

1. 共享 `logreg_val_eval` 在遍历 `C` 时提前计算 test accuracy。虽然没有用 test 选择 `C`，但不满足本计划“选择完成后才读取 test”的严格规则。
2. dryrun 结果是 dirty run，不能作为 PA-CER-M0-001 evidence。
3. tracker 尚未同步 dryrun/raw 状态。

## Fixes Applied

- `scripts/run_grace_1_1_8.py`：Logistic Regression evaluator 改成先只 fit train、看 val 选 `C`，选完 best classifier 后才单次读取 test labels 报告 test accuracy。
- `scripts/run_pacer_smoke.py`：P7 改为 train-fold ensemble probe route，并用 CV probe target logits，不再混用 full-train probe target。
- `scripts/run_pacer_smoke.py`：JSON 增加 `project_dirty` 主字段，并保留 `project_dirty_at_start` / `project_dirty_at_write`、`config_path`、`split_path`、exact command、commit。
- 正式 smoke 在 clean commit `81a3fc34fc6d13c5aada3fb571197097ea2c579f` 后启动，`project_dirty_at_start=False`。

## Final Review Status

复审结论：无新的代码级 BLOCKING。剩余 non-blocking 说明：

- `Y` 和 `test_idx` 仍在 evaluator 函数开头物化，但 test labels 未在 `C` 选择循环中索引。
- P7 route 是 train-fold ensemble margin，不是纯 OOF-only route；可接受，但报告中应标注其为 train-only CV diagnostic。
- `lambda_consistency` 当前未使用，不影响运行，但后续若不继续 PACER 可不必修。

## Bridge Decision

代码和 integrity 足以完成 Cora seed0 smoke；结果触发 no-go，因此不进入更多实验。
