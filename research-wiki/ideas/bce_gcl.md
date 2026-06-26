---
node_id: idea_bce_gcl
type: idea
title: "BCE-GCL: Boundary-Conditioned Contrastive Eligibility for Graph Contrastive Learning"
status: pivoted
outcome: pivot
created_at: 2026-06-26T08:34:42Z
updated_at: 2026-06-26T08:44:00Z
---

# BCE-GCL: Boundary-Conditioned Contrastive Eligibility

## 阶段状态

- 当前阶段：`pivoted`
- 当前决策：`PIVOT`
- 是否已实现：否
- 是否已运行 pilot：否
- 是否允许性能 claim：否

## 核心问题

标准图对比学习通常默认所有节点都适合相同强度的 cross-view alignment。BCE-GCL 的问题定义是：节点分类中 core nodes、boundary nodes、unstable nodes 对 contrastive alignment 的需求不同；强行对 boundary / unstable nodes 做强 alignment 可能削弱类别边界。

## 方法草案

1. 用 feature-structure disagreement、augmentation instability、local semantic conflict 估计 node-wise boundary / eligibility score。
2. 将节点路由为 core、boundary、unstable 三组。
3. core nodes 使用标准 InfoNCE。
4. boundary nodes 使用弱 boundary-preserving consistency。
5. unstable nodes 下调权重或只做 stability regularization。
6. 下游评估遵循项目协议：frozen encoder + Logistic Regression。

## 主要风险

- 可能与 hard-node weighting / uncertainty-aware GCL / degree-aware reweighting 重叠。
- 2026-06-18 的 BES 已经明确提出 boundary nodes + adaptive contrastive learning，叙事风险很高。
- 如果 boundary score 不能与 degree、uncertainty、augmentation strength 解耦，方法贡献会塌缩为普通 reweighting。
- 当前没有 pilot，不支持任何性能或 robustness claim。

## Novelty Check 结论

fresh `gcl_scientific_reviewer` 给出 `PIVOT`，novelty score `4/10`。BCE-GCL 原样不允许进入 `/research-refine-pipeline`。

主要原因：

- BES 已经非常接近 boundary nodes + adaptive contrastive learning + node classification。
- BCE 的 eligibility score 容易被解释为 degree / uncertainty / hard-node weighting 的代理。
- “boundary nodes 需要弱 alignment” 尚未被机制或 pilot 证明。

允许保留的 pivot 版本：

- `CER-GCL: Contrastive Eligibility Routing for Graph SSL`
- 叙事从 boundary-aware method 改为 unsupervised contrastive eligibility diagnostic + lightweight routing。
- boundary 只作为 post-hoc stratum，不作为主贡献词。

## 必须完成的下一步

- [x] 独立 novelty-check。
- [x] fresh `gcl_scientific_reviewer` 方法审查。
- [x] 将 BCE-GCL 原样叙事标记为 `PIVOT`。
- [ ] 为 CER-GCL 运行 refinement / experiment planning。
- [ ] 若 CER-GCL 仍失败，优先考虑 ProtoGuard-GCL。

## 关联文件

- `idea-stage/IDEA_REPORT.md`
- `idea-stage/BCE_GCL_NOVELTY_BRIEF_20260626_083442.md`
- `idea-stage/BCE_GCL_NOVELTY_CHECK_20260626_083442.md`

