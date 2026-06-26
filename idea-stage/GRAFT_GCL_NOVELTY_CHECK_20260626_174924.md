# GRAFT-GCL Novelty and Risk Audit

- 时间：2026-06-26T17:49:24Z
- 审查类型：executor-local novelty/risk audit
- Fresh reviewer：未在本轮调用；实现前建议补 fresh `gcl_scientific_reviewer`
- Claim status：pre-review only；不支持实验或性能 claim。

## Verdict

`REVISE_TO_PRE_REVIEW_AND_M0_SMOKE_PLANNING`

预估 novelty：`7.0 / 10`

信心：`0.66`

## Why Not KILL Immediately

GRAFT 的机制不再围绕 ORBIT 的 operator response、MEGA 的 masked evidence、PACER 的 train-label probe、SPECTRA 的 diagnostic regularizer 或 CAST/BEACON/IRIS 的 pair mining。它把 GCL 的监督对象从“样本/视图/响应”提升到“无标签任务族”，这是足够大的机制级 pivot。

## Main Novelty Hook

> Instead of constructing better positives/negatives, reconstructing masked graph evidence, or predicting operator responses, fit a graph encoder to a distribution of plausible node-classification tasks generated without labels on the current graph.

该 hook 与 GraphPFN 的关系是“借鉴 prior-task fitting 的思想，但不做 foundation model”；与 AutoSSL/AGSSL 的关系是“不是组合 pretexts，而是生成任务分布”；与 pseudo-label SSL 的关系是“不是预测真实标签，而是采样先验任务”。

## Biggest Rejection Risks

1. **Pseudo-labeling accusation**：如果 synthetic tasks 来自 clustering / label propagation，reviewer 会认为只是 pseudo-label SupCon。
2. **AutoSSL accusation**：如果只堆 feature/structure/diffusion tasks，reviewer 会认为是 multi-pretext stacking。
3. **GraphPFN adjacency**：若论文写成“synthetic task prior”，必须清楚区分同图 frozen encoder GCL 与跨图 foundation/in-context training。
4. **No downstream conversion**：如果 task prior meta-risk 改善但 frozen LogReg 不提升，应直接 kill。
5. **Hidden tuning budget**：task generator 的 family/temperature/class-count 可能形成巨大搜索空间，必须预注册并限制。

## Required Tightening

- 明确 `task prior` 的生成公式，不允许用真实标签或 test feedback 调先验。
- 明确 GRAFT full 与 clustering SupCon、random tasks、feature-only tasks、topology-only tasks、label-propagation tasks 的差异。
- 必须加入 `true-label SupCon upper bound`，只作诊断，不作为方法或 claim。
- 必须报告每个 synthetic task 与真实 train label 的 NMI/ARI 诊断，但不得用该诊断调参。
- 必须证明 GRAFT 不是一个 task 数量更多、epoch 更多、参数更多的 unfair control。

## Decision

`REVISE_TO_PRE_REVIEW_AND_M0_SMOKE_PLANNING`

下一步只能写最小 smoke 计划；实现前不允许宣称 GRAFT 优于任何方法。

