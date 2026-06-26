# EXPERIMENT_CODE_REVIEW: DSR-GCL Experiment Bridge

时间：2026-06-26T10:42:30Z  
类型：`[local-only]` code review fallback  
原因：当前会话未显式要求子代理；按工具约束不 spawn reviewer。  
范围：`scripts/run_dsr_smoke.py`、`configs/dsr_smoke.yaml`、`refine-logs/DSR_MECHANISM_SPEC.md` 与已完成的 Cora seed=0 fix-audit-smoke。

## Review Checklist

1. 方法实现是否匹配 proposal/spec：
   - 已修复为 raw projected `p_sem` 上的 VICReg-style loss。
   - InfoNCE 在 loss 内部 normalize。
   - DSR 同时保存 encoder-level `h_*` 和 projected normalized `z_*` 表示。
   - 主 DSR audit 评估使用 `h_concat`，与 GRACE encoder-output evaluator 更一致。

2. 超参数是否可配置：
   - `num_epochs`, `low_pass_k`, `alpha_res`, `beta_orth`, `lambda_var`, `lambda_cov`, `variance_gamma`, `firewall_threshold`, `make_undirected_after_dropout` 均在 `configs/dsr_smoke.yaml` 中记录。
   - variants A0/A2/A3/A4/A4b/A5/A5a/A5b/A5c/A9 均显式配置。

3. 数据 split / seed / evaluation 是否合规：
   - 使用项目已有 stratified random `1:1:8` split：`splits/Cora/split_seed_0.json`。
   - 使用 frozen encoder + Logistic Regression evaluator。
   - Logistic Regression 的 `C` 由 train/val 选择；未用 test 选择 embedding。

4. 是否存在 metric / leakage / output 问题：
   - raw JSON 与 JSONL logs 均已保存。
   - 每个 variant 保存 result path、log path、parameter count、rank、branch correlation、negative-gradient leakage。
   - Evaluation 对 dataset ground-truth labels 执行，不是对其他模型输出执行。

## Blocking Issues

无新的 blocking code issue。当前阻断来自实验结论，而不是脚本崩溃：

- residual-only `A3` 仍极低；
- fixed DSR-full `A9` 仍低于 no-firewall controls 与 single-head controls；
- 当前结果不满足进入 formal 或多数据集 pilot 的 gate。

## Non-Blocking Issues

- `refine-logs/EXPERIMENT_PLAN.md` 仍保留较旧的 pilot schedule；`EXPERIMENT_TRACKER.md` 是当前真实状态源。
- 如未来继续，应先写新的 residual/low-pass focused plan，而不是沿用旧 Pilot-A/B。

## Verdict

`PASS_FOR_SANITY_COLLECTION_ONLY`。  
代码可用于记录 Cora seed=0 fix-audit-smoke 结果，但当前 evidence gate 阻断 full experiment bridge deployment。不得进入 formal，不得生成性能 claim。
