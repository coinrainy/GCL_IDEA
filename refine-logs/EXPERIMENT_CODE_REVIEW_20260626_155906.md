# SPECTRA-GCL SP-M0-001 Code Review

- 时间：2026-06-26T15:59:06Z
- Review mode：local-only checklist；未调用 secondary reviewer。
- Files reviewed：
  - `configs/spectra_smoke.yaml`
  - `scripts/run_spectra_smoke.py`

## Scope Check

- Dataset gate：只允许 Cora seed0。
- Evaluator：复用 `run_grace_1_1_8.logreg_val_eval`，train fit / val select / test report。
- Split：复用 `create_or_load_split`，stratified random `1:1:8`。
- Result format：每个 variant 单独 JSON，另有 parseable summary JSON 和 Markdown summary。
- Metrics：accuracy、effective rank、participation ratio、boundary energy retention、covariance redundancy、alignment、uniformity。

## Blocking Issues Found and Fixed

1. YAML `description` 中未加引号的冒号导致 `yaml.scanner.ScannerError`。
   - Fix：将 `S2` 描述加引号。

2. boundary residual energy 使用 `sqrt(sum(x^2))`，在零残差点可能产生不稳定梯度，dryrun 中导致 NaN embedding，Logistic Regression 拒绝输入。
   - Fix：改为 `sqrt(sum(x^2) + 1e-12)`。
   - Added guard：SPECTRA 专用学习率 `0.0001` 与 `max_grad_norm=5.0`。

## Non-blocking Issues

- S7 matched-parameter GRACE 复用 S0 embedding，因为当前 SPECTRA 与 GRACE 的 encoder/projection hidden dimensions 均为 `128`。这对 smoke 足够透明，但后续如果 SPECTRA 改 projection budget，S7 应单独训练。
- `boundary_energy_retention` 是 profile-based diagnostic，不是可直接解释为分类边界保真率的理论量。结果显示它不足以支持机制 claim。
- `rank_guard_loss` 使用固定 `gamma=1.0`，当前 smoke 不允许继续调参；若未来重启必须先重新论证机制，不应直接扫权重。

## Integrity Verdict

`PASS_WITH_NEGATIVE_RESULT`

代码完成 SP-M0-001 的执行要求，且 dryrun 捕捉并修复了数值问题。最终结果为 negative smoke：SPECTRA full 的 diagnostics 改善没有转化为 accuracy，应停止当前 boundary-energy story。

