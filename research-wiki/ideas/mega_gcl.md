# MEGA-GCL

- ID：`mega_gcl`
- 全称：Masked Evidence Group Assignment for Graph Contrastive Learning
- 阶段：`proposed`
- 当前决策：`REVISE_TO_MEGA_CORA_KILL_SMOKE`
- 创建时间：2026-06-26T16:40:00Z
- 类型：label-free masked discriminative graph SSL for ordinary node classification
- Claim status：proposal only；未实现、未跑 smoke/pilot/formal；不支持性能 claim。

## Origin

MEGA-GCL 是 CAST / BEACON / SPECTRA / PACER 四连 no-go 后的 pivot。

- CAST：低 overlap certificate positives 没有分类收益。
- BEACON：更高 label agreement / margin-rank diagnostics 没有收益。
- SPECTRA：rank / boundary retention 改善没有收益。
- PACER：train/val probe margin delta 改善也没有收益，random route / SupCon 更强。

## Core Hypothesis

问题不是还没找对 false negatives，而是 node-node contrastive denominator 与少标签节点分类任务错配。MEGA 将 contrastive target 换成 masked ego evidence groups，绕开真实节点负样本，同时训练表示保留 class-readable evidence。

## Method Sketch

1. 无标签 evidence tokenizer：feature evidence、structural evidence、residual evidence。
2. Mask anchor ego evidence。
3. Encoder 从剩余上下文预测 masked evidence group。
4. Frozen encoder + Logistic Regression evaluator。

## Non-Goals

- 不做 kNN/PPR/CAST mining。
- 不做 cross-node positive mining。
- 不做 pair score fusion。
- 不做 post-hoc relation smoothing。
- 不继续 PACER-style train-label probe routing。

## Fresh Reviewer

Fresh `gcl_scientific_reviewer` verdict：`REVISE`，confidence `0.74`。MEGA 是 Top1，但必须证明不是 GraphMAE/GraphMAE2/GCMAE/VQ/SwAV/prototype/token-corruption 变体。

## Minimum Kill-Smoke

`ME-M0-001`：Cora seed0，M0-M13 controls。若 MEGA full 不能超过 GRACE、GraphMAE-style、GCMAE-style、VQ/prototype、MINT token-corruption、random/permuted evidence controls，则 kill。

## Decision

`REVISE_TO_MEGA_CORA_KILL_SMOKE`
