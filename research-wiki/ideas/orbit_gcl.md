# ORBIT-GCL

- ID：`orbit_gcl`
- 全称：Operator-Response Basis Induction for Graph Contrastive Learning
- 阶段：`proposed`
- 当前决策：`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`
- 创建时间：2026-06-26T17:16:14Z
- 任务类型：ordinary node classification GCL
- Claim status：proposal only；未实现、未跑 ORBIT smoke/pilot/formal；不支持性能 claim。

## Origin

ORBIT-GCL 是 MEGA-GCL kill-smoke no-go 后的新一轮 idea-discovery pivot。此前 CAST/IRIS/BEACON/SPECTRA/PACER/MEGA 的共同问题是：pair-level purity、boundary/rank/probe diagnostics、masked evidence prediction 没有稳定转化为 frozen Logistic Regression 节点分类 accuracy。

## Core Hypothesis

节点分类所需的 class-readable evidence 不是简单节点相似性或输入重建，而是节点对一组图干预算子的响应模式。若一个 encoder 能从 clean / weak view 预测该节点在 feature、structure、diffusion、ego radius、position perturbations 下的 response basis，则它可能比 node-node contrastive 或 masked reconstruction 更贴近下游 class boundary。

## Method Sketch

1. 固定 label-free operator bank `Omega`。
2. 用 EMA teacher 在 clean graph 与每个 operator graph 上产生 node response field。
3. 对 response field 做 low-rank basis projection。
4. Student encoder 从 clean / weak view 预测 response basis。
5. 用 held-out operator response prediction 和 frozen Logistic Regression accuracy 共同判定。

## Fresh Reviewer

Fresh `gcl_scientific_reviewer` verdict：`REVISE`，confidence `0.73`。

Reviewer 认为 ORBIT 是当前候选中最值得保留的方向，但必须证明它不是 Graph-JEPA、TTER/D-SLA、GraphMAE2、GCIL/AS-GCL 或 regularization/capacity 的换名。

## Required Controls

- GRACE / matched GRACE
- GraphMAE2-lite / GCMAE-style
- Graph-JEPA latent target
- TTER-style operator-id prediction
- D-SLA-style discrepancy regression
- IRIS/WILLOW response control
- random operator labels
- shuffled response basis
- degree/feature-only response
- random teacher
- no-SVD raw response

## Decision

`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`

Next allowed action：implement and run only Cora seed0 OR-M0 kill-smoke. No Pilot-A/B/formal and no performance claim before passing hard gates.
