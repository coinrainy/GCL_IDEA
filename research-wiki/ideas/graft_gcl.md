# GRAFT-GCL

- ID：`graft_gcl`
- 全称：Graph Prior-Task Fitted Contrastive Learning
- 阶段：`proposed`
- 当前决策：`REVISE_TO_PRE_REVIEW_AND_M0_SMOKE_PLANNING`
- 创建时间：2026-06-26T17:49:24Z
- 任务类型：ordinary node classification GCL
- Claim status：proposal only；无 smoke/pilot/formal 结果。

## Origin

GRAFT-GCL 是 ORBIT-GCL `OR-M0-001` negative smoke 后的机制级 pivot。ORBIT full minimal `82.98` 低于 GRACE `84.78` 和 matched GRACE `84.96`，触发 `KILL_ORBIT_AS_WEAKER_THAN_MATCHED_CONTROL` 与 `KILL_ACCURACY_NOT_CONVERTED`。因此不得继续 ORBIT operator bank / basis rank / teacher response / loss weight 小调。

## Core Hypothesis

Graph SSL 与节点分类之间的 mismatch 不只是 false negatives 或 target quality 问题，而是 pretraining objective 没有模拟 downstream node-classification task family。GRAFT 通过无标签任务先验在当前图上生成许多 plausible class partitions，并用 task-conditioned supervised contrastive meta-risk 训练 encoder。

## Mechanism

1. 构造 label-free task prior `p(tau | G)`。
2. 每个 task 产生 synthetic labels `y_i^tau`。
3. Encoder 用 task-conditioned SupCon 学习 representations。
4. 训练后冻结 encoder，用真实 train labels 训练 Logistic Regression，并在 validation 选择 `C` 后报告 test。

## Task Families

- Feature mixture tasks。
- Topology role tasks。
- Homophily-controlled MRF tasks。
- Feature-topology interaction tasks。

## Required Controls

- GRACE / matched GRACE。
- clustering SupCon pseudo-label。
- random task prior。
- feature-only / topology-only priors。
- label-propagation/PPR prior。
- AutoSSL-style multi-pretext。
- task-shuffled GRAFT。
- true-label SupCon upper bound diagnostic。

## Decision

`REVISE_TO_PRE_REVIEW_AND_M0_SMOKE_PLANNING`

下一步只能是 fresh review 或 Cora seed0 `GRAFT-M0-001` kill-smoke planning/implementation。禁止 Pilot-A/B/formal 与性能 claim。

