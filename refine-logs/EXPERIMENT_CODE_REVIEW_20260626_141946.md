# CAST Certificate Pilot-A Diagnostic Code Review

- 时间：2026-06-26T14:19:46Z
- 范围：`configs/cast_certificate_smoke.yaml`、`scripts/run_cast_certificate_smoke.py`
- 审查类型：executor-local integrity review，不是 fresh external reviewer verdict。
- 实验阶段：smoke/sanity instrumentation；不支持 formal、SOTA 或 robust claim。

## 变更检查

本轮把 CAST certificate runner 从 C0-C5 扩展为 Pilot-A 诊断版本：

- 新增 C6 `ppr_diffusion_positive`，作为 graph diffusion / PPR-style positive mining control。
- 输出 `overlap_with_ppr_C6`，与已有 kNN/CAST overlap 一起用于证明 C4/C5 不只是 kNN/PPR/CAST mining。
- 新增 `sampled_partial_corr`，在 certificate finite candidate pairs 上采样，估计 certificate score 在控制 feature similarity、embedding similarity、graph diffusion similarity 与 degree gap 后对同标签关系的剩余相关。
- 保留 Pilot-A gate：仅允许 Cora/CiteSeer/PubMed seeds 0-2。

## 完整性结论

- `python -m py_compile scripts/run_cast_certificate_smoke.py` 通过。
- PA-M0-001b sanity 通过：`results/summary/pa_m0_001_sanity_v3_CiteSeer_seed0_20260626T141904Z_summary.json`。
- 该 sanity 只使用 1 GRACE epoch + 1 certificate epoch，目的仅为确认多数据集链路、C6、PPR overlap 和 sampled partial-correlation 都能落盘。
- 当前 partial-correlation 是 offline label diagnostic；训练、candidate construction 和 relation smoothing 没有读取标签。

## 风险与限制

- `graph_sim`、`emb_sim`、`feat_sim`、`cast_score` 仍是 dense pair matrix；PubMed Pilot-A 可能有较高 CPU 内存/时间开销，但已避免全量 `triu_indices` pair 展开。
- C6 是 local multi-hop diffusion proxy，不是精确 personalized PageRank；Pilot-A 报告中应称为 PPR-style / graph diffusion control。
- PA-M0-001b sanity 不应进入结果表，只用于证明 runner instrumentation 可执行。

## 决策

`GO_TO_PA_M1_WITH_RESOURCE_CAUTION`

下一步先顺序运行 PA-M1-CORA seeds 0-2，再运行 PA-M1-CITE，最后视资源运行 PA-M1-PUB。
