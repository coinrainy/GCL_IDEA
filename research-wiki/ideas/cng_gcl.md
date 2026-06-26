# CNG-GCL: Conflict-Negative Graph Selection for GCL

status: proposed_candidate  
decision: GO_FOR_REFINEMENT  
created_utc: 2026-06-26T09:21:55Z  
source: `/idea-creator` false-negative GCL generation beyond IGT-GCL  
task: node classification  
method family: graph contrastive learning

## Summary

CNG-GCL 将 hard-negative mining 从 pair scoring 改写为 anchor-local conflict graph selection。对每个 anchor，在候选 negatives 之间构造 conflict graph：若候选节点之间结构、特征、EMA 表示或邻域高度耦合，则它们可能来自同一语义区域，不应同时构成有效 denominator。训练只使用小规模、低冲突、高覆盖的 selected negatives。

## Why Beyond IGT

IGT-GCL 仍需要为 anchor-candidate pair 赋 interval 或 action。CNG-GCL 不输出 pair posterior、interval 或连续权重，而是选择一个组合上更少 false-negative 风险的 negative set。

## Closest Prior Risk

E2Neg 已提出少量高质量、非拓扑耦合 negatives 的方向；B2-Sampling、ProGCL 和 cluster-refined negative sampling 也相关。CNG 必须证明 anchor-local conflict graph 的组合结构不是 random small negatives 或 E2Neg representative negatives 的再包装。

## Required Ablations

- vanilla GRACE / GCA
- random `m` negatives
- hardest `m` negatives
- E2Neg-style representative negatives
- ProGCL-style true-negative weighting
- no conflict edges, diversity only
- conflict score single-signal vs multi-signal

## Kill Rule

若 random small negatives 或 E2Neg-style representative selection 匹配 full CNG，或 selected negatives 的 label-only offline false-negative 诊断不优于 hard negatives，则 kill。

## Evidence Status

当前只有 idea generation 文档，无 smoke、pilot、development 或 formal 实验，无性能 claim。
