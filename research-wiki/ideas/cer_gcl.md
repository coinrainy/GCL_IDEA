---
node_id: idea_cer_gcl
type: idea
title: "CER-GCL: Contrastive Eligibility Routing for Graph SSL"
status: archived
outcome: abandoned_by_user
created_at: 2026-06-26T08:44:00Z
updated_at: 2026-06-26T08:50:34Z
---

# CER-GCL: Contrastive Eligibility Routing for Graph SSL

## 来源

CER-GCL 是从 BCE-GCL novelty-check 后 pivot 得到的收缩版本。fresh `gcl_scientific_reviewer` 判断 BCE-GCL 原样 `PIVOT`，但允许保留 “无监督 contrastive eligibility diagnostic + lightweight objective routing” 这一方向。

## Problem Anchor 草案

在无监督图对比学习中，并非所有节点都同样适合强 positive alignment。核心问题不是 “boundary nodes 是否需要特殊处理”，而是：能否用无标签信号预测哪些节点从强 alignment 中受益、哪些节点会因增强不稳定或结构-特征冲突而受损，并据此选择最小的 objective route。

## 方法草案

- 估计无监督 eligibility signal：augmentation-instability、feature-structure disagreement、contrastive loss stability。
- 不把 signal 直接解释为真实 class boundary。
- 将节点路由为：
  - strong alignment：标准 InfoNCE。
  - weak consistency：降低 positive alignment 压力。
  - stability-only / downweight：只保留稳定性约束或降低权重。
- routing 作为验证工具，核心 claim 优先是 diagnostic predictive value，而不是 SOTA 性能。

## 必须证伪

- 若 eligibility score 与 degree / uncertainty / augmentation strength 高度相关，且控制这些 baseline 后无增益，则停止。
- 若 objective routing 不优于同分数 scalar weighting，则停止。
- 若收益只来自更弱 contrastive pressure 或温度/权重正则化，则停止。

## 阶段决策

当前决策：`ARCHIVED_BY_USER`

2026-06-26T08:50:34Z 用户明确要求放弃之前的 idea 工作，并重新开始一个新的方向。因此 CER-GCL 不再作为当前主线推进；不得继续实现、跑 smoke/pilot，或把此前 refinement 产物当作 active proposal 使用。

归档前状态：`READY_FOR_PILOT_PLANNING`

`/research-refine-pipeline` 已完成。fresh `gcl_scientific_reviewer` Round 1 给出 `REVISE` / `6.7`，修订后二次复评给出 `READY` for minimum pilot planning / `7.7`。

关键收缩：

- 三路 routing 改为二路 routing：eligible anchors 使用 full InfoNCE，ineligible anchors 使用 positive-only stop-gradient consistency。
- `s_loss` 不再作为 gate 输入，只作 diagnostic。
- 首轮 gate 固定为 `q_i = rank_norm(drift_i) + rank_norm(fs_residual_i)`，`rho=0.8`，不做 pilot threshold search。
- same-score scalar weighting 是第一 kill baseline。

下一步：只允许进入 smoke/pilot implementation，不允许 formal claim。

关联文件：

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `refine-logs/PIPELINE_SUMMARY.md`
