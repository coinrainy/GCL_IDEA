# Operator-Response GCL Candidate Set

- 时间：2026-06-26T17:16:14Z
- 方向：重新进行 idea-discovery，要求可行性、创新性、完整论文骨架，不做小型创新
- 输入约束：吸收 CAST / IRIS / BEACON / SPECTRA / PACER / MEGA 等 no-go；避开 pair mining、loss trick、diagnostic regularizer、pseudo-task stacking
- Claim status：候选生成与审查；未实现、未跑实验。

## 候选排名

| Rank | Idea | Verdict | 为什么保留/淘汰 |
|---:|---|---|---|
| 1 | ORBIT-GCL / Operator-Response Basis Induction | `REVISE` | 最像机制级 pivot：学习节点在多种无标签图干预算子下的 response field，而不是节点相似性或输入重建。需要压实成低秩 response-basis prediction，证明不是 Graph-JEPA/TTER/D-SLA 换名。 |
| 2 | MORPH-GCL / Morphological Ego Program Learning | `PIVOT` | 用 ego graph edit program 作自监督目标，有一定创新性；但离普通节点分类 GCL 过远，实现和强 baseline 风险大，容易撞 Graph Edit Networks / topology transformation SSL。 |
| 3 | SCALE-GCL / Multi-scale Evidence Conservation | `KILL/weak PIVOT` | 容易重复 SPECTRA 的失败：multi-scale/rank/boundary diagnostics 改善但 accuracy 不转化。 |
| 4 | ATLAS-GCL / Self-Supervised Label-Function Atlas | `KILL` | 太像多 pretext / pseudo-label stacking；难形成强论文主线。 |

## Top1: ORBIT-GCL

### 核心问题

现有 GCL 通常学习同一节点两视图 agreement 或不同节点 separation；masked SSL 学输入/latent 重建；近来的 latent predictive SSL 学 context-target latent。它们共同假设“稳定表征来自视图不变或输入恢复”。但普通少标签节点分类真正需要的是：节点对不同证据扰动的响应模式是否能暴露 class-readable boundary。

### 核心方法

ORBIT 固定一组 label-free graph intervention operators：

- feature-frequency mask
- edge-degree preserving perturbation
- diffusion-temperature shift
- ego-radius crop
- positional perturbation

对每个节点，EMA teacher 在 clean graph 与 operator-perturbed graphs 上生成 response field：

```text
R_i = [h_T(O_1(G), i) - h_T(G, i), ..., h_T(O_M(G), i) - h_T(G, i)]
```

将 `R_i` 投影成低秩 response basis target `b_i`。Student encoder 只从 clean / weak view 输入中预测 `b_i`，并通过 held-out operator response prediction 检验是否学到可泛化响应函数。

### 为什么不是小创新

- 不是调 loss：核心 supervision object 从 node-node / reconstruction 变成 operator-valued response function。
- 不是 masked autoencoder：目标不是恢复 feature/edge/position。
- 不是 Graph-JEPA：目标不是单个 target latent，而是跨 operator 的 response basis。
- 不是 TTER/D-SLA：目标不是预测 transformation id 或全局 discrepancy，而是 node-level response field。
- 不是 IRIS/WILLOW：不做跨节点 response positive search，不构造 positive/neutral pairs。

## Fresh Reviewer Summary

Fresh `gcl_scientific_reviewer` verdict：`REVISE`，confidence `0.73`。

Reviewer 排名：ORBIT > MORPH > SCALE > ATLAS。

Reviewer 的主要担忧：

1. ORBIT 当前可能被解释为 GraphCL augmentation + Graph-JEPA latent prediction + TTER/D-SLA transformation prediction + Barlow guard 的组合。
2. 必须证明 response basis 与下游 class-readable evidence 有关系，而不是 degree/feature/diffusion shortcut。
3. 组件必须收缩，避免提升来自 EMA teacher、regularization、capacity 或 augmentation budget。

## Decision

`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`

只允许先写 ORBIT minimal spec 与 Cora seed0 kill-smoke。不得直接 Pilot-A/B，不得 formal，不得 claim 性能。
