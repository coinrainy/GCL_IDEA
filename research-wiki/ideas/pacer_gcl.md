# PACER-GCL

- ID：`pacer_gcl`
- 全称：Probe-Aligned Calibration for Evaluator-Ready Graph Contrastive Learning
- 阶段：`smoke_negative`
- 当前决策：`KILL_OR_PIVOT_REQUIRED`
- 创建时间：2026-06-26T16:06:34Z
- 类型：train-label-calibrated / semi-supervised GCL for ordinary node classification
- Claim status：Cora seed0 smoke negative；不支持性能 claim。
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

## Smoke Result

`PA-CER-M0-001` 已完成，Cora seed0 only。

| ID | System | Test@best-val |
|---|---|---:|
| P0 | GRACE | 84.69 |
| P3 | PACER full | 83.63 |
| P4 | shuffled-label probe | 83.63 |
| P5 | random-probe route | 84.92 |
| P6 | scalar reweight control | 83.76 |
| P8 | supervised-contrastive small-label control | 85.15 |

P3 的 train/val probe margin delta 为正，但没有转化为 accuracy；P4 持平、P5/P8 反超，说明当前 train-label calibrated routing 叙事不成立。

## Decision

`KILL_OR_PIVOT_REQUIRED`

Triggered rules：`KILL_PACER`、`KILL_LABEL_CALIBRATION_STORY`、`KILL_ROUTING_STORY`、`KILL_OBJECTIVE_ROUTING_STORY`、`KILL_PACER_AS_UNNECESSARY`。

Next allowed action：不要继续 PACER threshold/weight/routing 小调；若继续 false-negative GCL，应重新做 idea discovery，并显式吸收 PACER 失败证据。
