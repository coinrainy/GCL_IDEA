# Research Proposal: WILLOW-GCL

## Problem Anchor

- **Bottom-line problem**：普通节点分类 GCL 同时受两个信号问题限制：随机增强 positives 经 message passing 后可能过度预对齐，真实节点 negatives 在 hard region 又容易包含 false negatives。
- **Must-solve bottleneck**：不要继续改 denominator 或 pair weights；必须改变 positive view 的生成机制，并证明该机制不是普通 learned augmentation。
- **Core mechanism**：latent ego target-prediction certificate + certified intervention positive search。
- **Non-goals**：不主打 virtual hard negatives；不重构 raw feature；不向原图加入 synthetic nodes/edges；不使用 test label；不把 smoke 写成性能 claim。
- **Constraints**：首轮只做 Cora seed=0 smoke，`1:1:8` split，frozen encoder + Logistic Regression evaluator。

## Method Thesis

**One-sentence thesis**：WILLOW-GCL uses a latent ego target-prediction certificate to search for hard but semantically certified positive views, making graph contrastive positives informative without relying on loss-side false-negative correction.

## Minimal Irreplaceable Core

WILLOW 不能被拆成任意单组件：

- latent world prediction alone = Graph-JEPA / Predict-Cluster-Refine risk；
- intervention search alone = SIVA / GCA / RGCL / ACGA risk；
- WILLOW = 用 latent ego target-prediction error 约束 intervention search，生成更远但语义稳定的 GCL positive view。

## Proposed Method

### 1. Latent Ego Target-Prediction Certificate

对 anchor `i` 采样 context ego-subgraph `c_i` 与多个 masked target ego-subgraphs `t_i^1...t_i^m`。

```text
h_c      = E_online(c_i)
h_t^k    = E_target(t_i^k)       # EMA target encoder
pred_t^k = P(h_c, pos_k)
L_cert   = sum_k || pred_t^k - stopgrad(h_t^k) ||_2^2
```

实现约束：

- `E_target` 使用 EMA，不通过 predictor 反传。
- 不使用 raw feature decoder。
- collapse prevention 可先采用 variance/covariance regularizer 或 target embedding normalization。
- `epsilon` 必须由 train/val diagnostic 选择，禁止看 test。

### 2. Certified Intervention Positive Search

在固定 search budget 下，对每个 anchor 搜索 intervention `a`：

```text
a_i^+ = argmax_a dist(view_i(a), view_i(original))
        subject to cert_error(i, a) <= epsilon
```

候选 intervention 只包括 node-local 操作：feature mask、edge drop/keep、neighbor context mask、小范围 ego-context substitution。首版使用 greedy candidate pool，不训练大 generator。

### 3. Contrastive Encoder Training

用 certified positive view 替代普通 random positive：

```text
positive pair = (ordinary view of ego_i, certified intervention view_i(a_i^+))
loss = GRACE/GCA-style contrastive objective
```

真实节点 negatives 保持 easy/random subset 或诊断项；主线不加入 virtual negatives。

## Required Controls

| System | Purpose |
|---|---|
| GRACE | base GCL reference |
| Graph-JEPA-only | 检查 latent prediction 本身是否已解释效果 |
| SIVA reconstruction-critic positive | 检查 WILLOW 是否真的强于 GraphMAE-like critic |
| edit-distance matched random hard positive | 检查 certificate 是否必要 |
| certificate-shuffled WILLOW | 检查 certificate 是否包含 anchor-level 语义 |
| WILLOW-certified positive | main candidate |

## Metrics

- Frozen Logistic Regression accuracy under project protocol。
- `cert_error` distribution。
- view edit distance / representation distance。
- post-GNN positive similarity。
- positive gradient norm。
- offline label-agreement diagnostic only for auditing, never for training。
- false-negative repulsion mass diagnostic。

## Kill Rules

- Graph-JEPA-only matches WILLOW：kill。
- SIVA reconstruction critic matches WILLOW：downgrade to SIVA/control。
- matched random positive matches WILLOW：certificate invalid。
- certificate-shuffled matches WILLOW：certificate not semantic。
- `cert_error` is uncorrelated with offline semantic stability / label agreement：kill certificate。
- gains come only from larger search or compute budget：invalid mechanism evidence。

## Current Verdict

`REVISE_TO_WILLOW_SMOKE_PLANNING`。  
这是 proposal evidence，不是实验结果。下一步只能做最小 smoke，不允许 formal 或性能 claim。
