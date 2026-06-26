# ORBIT-GCL Experiment Code Review

- 时间：2026-06-26T17:44:25Z
- 范围：`configs/orbit_smoke.yaml`、`scripts/run_orbit_smoke.py`
- Review type：local-only checklist review
- 原因：本轮按 `/experiment-bridge` 执行，但未调用外部 reviewer；本地按 skill checklist 审查。

## Verdict

`WARN_NON_BLOCKING`

代码可运行并已通过 1-epoch dryrun 与 OR-M0-001 full smoke。未发现会改变本轮结论的 blocking leakage 或 evaluator 错误。

## Checks

| Check | Status | Notes |
|---|---|---|
| split protocol | pass | 使用 `create_or_load_split`，Cora seed0，stratified random `1:1:8` |
| evaluator | pass | 使用 `logreg_val_eval`，train fit / val select `C` / final test report |
| test-label leakage | pass | operator targets、thresholds、routing 均不使用 label；test label 只进入最终 evaluator |
| systems O0-O14 | pass | 已实现计划中的 15 个 systems |
| provenance | warn | full smoke 从 clean code commit `e563f53` 运行，但 `project_dirty_at_start=True`，因为工作区已有未跟踪 logs 目录 |
| result persistence | pass | 写出 raw JSON、summary JSON/MD、JSONL logs |
| scope control | pass | gate 限制为 Cora seed0；未运行 Pilot-A/B/formal |

## Non-Blocking Issues

1. `random_operator_labels` 当前实现为随机连续 target，而不是 categorical operator id 的随机标签。它仍可作为 artifact/shortcut control，但命名可以在后续若继续该线时更精确。
2. `degree_preserving_edge_rewire` 是 source-degree preserving approximation，并非严格双端 degree-preserving edge swap。由于 ORBIT 已触发 no-go，这一点不影响继续扩展决策。
3. `project_dirty_at_start=True` 来自未跟踪日志目录，不影响 raw result 的 commit hash 记录，但正式实验前应清理或 ignore logs。

## Conclusion

本轮结果为 negative/no-go；不建议基于当前 ORBIT runner 扩展 Pilot-A。若未来重开 ORBIT，应先重写 operator bank 与 target diagnostics，再重跑 clean dryrun。
