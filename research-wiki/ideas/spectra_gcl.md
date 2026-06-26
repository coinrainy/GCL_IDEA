# SPECTRA-GCL

- ID：`spectra_gcl`
- 全称：Spectral Boundary-Preserving Redundancy Reduction for Graph Contrastive Learning
- 阶段：`proposed`
- 当前决策：`GO_TO_SPECTRA_SMOKE_PLANNING`
- 创建时间：2026-06-26T15:40:39Z
- 任务类型：ordinary node classification GCL
- Claim status：idea/proposal only；无 smoke/pilot/formal 结果；不支持性能 claim。

## Origin

SPECTRA-GCL 来自 CAST Certificate Pilot-A no-go 与 BEACON-GCL BE-M0 no-go 之后的机制级 pivot。

- CAST：C4 latent certificate 与 kNN/PPR/CAST overlap 低，但低-overlap certificate positives 没有稳定提升分类。
- BEACON：B5 full gate 的 label agreement 和 margin-rank diagnostics 更好，但 accuracy 低于多个简单 controls。

结论：继续寻找更好的 cross-node pair 不是当前最有价值路径。需要从 objective-level / representation-level 重新定义 false-negative GCL。

## Core Hypothesis

pair-level false-negative correction 在节点分类上失败，是因为它只修补局部 relation，而没有直接保护下游 Logistic Regression 需要的表示结构：有效秩、局部边界残差、中高频 class-boundary cues，以及不过度 smoothing 的多粒度簇结构。

SPECTRA-GCL 用 negative-free redundancy reduction 避开 cross-node false negatives，并用 label-free spectral boundary energy preservation 防止 negative-free objective 抹平边界。

## Method

目标：

```text
L_SPECTRA = L_rr + lambda_b * L_boundary + lambda_r * L_rank
```

- `L_rr`：same-node two-view redundancy reduction。
- `L_boundary`：label-free graph residual / high-pass boundary energy preservation。
- `L_rank`：effective-rank / variance guard。

## Non-contributions

- 不做 kNN/PPR/CAST mining。
- 不做 cross-node positive mining。
- 不做 score fusion。
- 不做 loss-only reweighting。
- 不做 post-hoc relation smoothing。
- 不继续 CAST C4/C5 或 BEACON gate。

## Smoke Plan

`SP-M0-001`：

- Cora seed0 only。
- 对照 S0-S7：GRACE、negative-free RR、SPECTRA full、no-boundary-energy、no-rank-guard、random-spectrum guard、low-band-only、matched-parameter GRACE。
- 输出 accuracy、effective rank、boundary energy retention、covariance redundancy、alignment/uniformity。

## Risks

- 与 Graph Barlow Twins / CCA-SSG / BGRL / VICReg 接近。
- 与 spectral SSL / DSR spectral 分支存在叙事碰撞。
- 如果 diagnostics 改善但 accuracy 不改善，将重复 BEACON 失败模式，必须停止而不是调权重。

## Decision

`GO_TO_SPECTRA_SMOKE_PLANNING`

Next allowed action：通过 `/experiment-bridge` 实现并运行 `SP-M0-001` Cora seed0 smoke。

