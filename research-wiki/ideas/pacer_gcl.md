# PACER-GCL

- ID：`pacer_gcl`
- 全称：Probe-Aligned Calibration for Evaluator-Ready Graph Contrastive Learning
- 阶段：`proposed`
- 当前决策：`GO_TO_PACER_SMOKE_PLANNING_WITH_CAUTION`
- 创建时间：2026-06-26T16:06:34Z
- 类型：train-label-calibrated / semi-supervised GCL for ordinary node classification
- Claim status：proposal only；未实现、未跑 smoke/pilot/formal；不支持性能 claim。
- Fresh review status：scientific reviewer `PIVOT`; experiment auditor `WARN`。

## Origin

PACER-GCL 是 CAST / BEACON / SPECTRA 三连 no-go 后的机制级 pivot。

- CAST：低-overlap certificate pairs 没有稳定提升 accuracy。
- BEACON：更高 label agreement / margin-rank diagnostics 没有提升 accuracy。
- SPECTRA：effective-rank / boundary-retention diagnostics 改善没有提升 accuracy。

Fresh `gcl_scientific_reviewer` verdict：`PIVOT`。推荐从 pair purity 和 representation diagnostics 转向少标签 Logistic Regression downstream alignment。

Fresh `gcl_experiment_auditor` verdict：`WARN`。下一轮 smoke 必须预注册 controls/kill rules，保存 exact run command，并避免 dirty-run provenance 风险；Cora seed0 未过不得扩大。

## Core Hypothesis

False-negative GCL 的核心瓶颈不是 pair 识别，而是 pretraining objective 与最终 frozen Logistic Regression evaluator 错配。PACER 用 train-label logistic probe 估计 anchor 对下游读出的 fragility，并据此切换 objective type。

## Method Sketch

1. GRACE warmup。
2. Train-label Logistic Regression probe calibration。
3. Anchor-level objective routing：
   - stable anchors：GRACE instance discrimination；
   - fragile anchors：masked logit-preserving consistency。

PACER 不做 cross-node positive mining，不做 pair score，不做 relation smoothing。

## Risks

- 不是无监督 GCL，必须诚实标注为 train-label-calibrated / semi-supervised。
- 可能与 SupCon、semi-supervised GCL、probe loss、meta-learning readout 接近。
- 有 validation leakage 和 train-probe overfit 风险。

## Smoke Plan

`PA-CER-M0-001`：

- Cora seed0 only。
- P0-P7 controls：GRACE、probe-only、mask-only、PACER full、shuffled-label probe、random-probe route、scalar reweight、train-only CV probe。
- 额外 P8 supervised-contrastive small-label control。
- 必须输出 accuracy、train/val probe margin delta、fragile-anchor fraction、route distribution、control gaps、exact run command、commit 与 dirty flag。

## Decision

`GO_TO_PACER_SMOKE_PLANNING_WITH_CAUTION`

Next allowed action：实现 Cora seed0 smoke。不得进入 Pilot-A/B 或 formal，直到 P3 PACER full 明确超过 GRACE、mask-only、shuffled/random/scalar controls。
