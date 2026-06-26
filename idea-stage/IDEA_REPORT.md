# IDEA_REPORT: False-Negative GCL Beyond IGT

生成时间：2026-06-26T09:24:55Z  
工作流：`/idea-discovery`  
方向：图对比学习中的假负样本问题，用于节点分类  
目标：寻找比 IGT-GCL 新颖性更强、机制更清晰、可在当前仓库落地、具有 2026 顶会/顶刊投稿潜力的方法 idea。  
最终 decision：**REVISE_IDEA**

## 1. Executive Summary

本轮不继续推进 IGT-GCL。IGT-GCL 的 novelty score 为 `5.5/10`，低新颖性风险来自 point posterior / interval / abstention / sample reweighting 邻域。

新的主候选是：

**DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning**  
核心机制：**Spectral Gradient Firewall**

DSR-GCL 不再判断哪个 pair 是 false negative，而是将表示分成 semantic channel 与 residual channel：semantic channel 不接收 node-node negative repulsion；residual channel 承担 InfoNCE/rank-style uniformity 和边界压力。fresh `gcl_scientific_reviewer` verdict 为 **REVISE_IDEA**，novelty score **6.2/10**。它小幅强于 IGT-GCL，但还不能进入 `GO_TO_EXPERIMENT_BRIDGE`。

本轮未实现代码、未跑 smoke/pilot/formal 实验、未产生任何性能 claim。

## 2. Protocol Constraints

- 同配图：stratified random `1:1:8` split。
- 异配图：官方或公认 fixed split。
- GCL：frozen encoder + Logistic Regression evaluator。
- Logistic Regression 超参只能由 train/val 或 train-only CV 决定。
- formal 默认 10 seeds，但本阶段没有 formal。
- smoke/pilot/development 不支持 SOTA、robust、comprehensive claim。
- 不混 split，不用 pilot/development 结果做论文 claim。

## 3. Literature Landscape

本轮基于已有 `literature/` 和补充检索，得到以下边界：

| Prior line | Boundary |
|---|---|
| ProGCL / AUHNM / NML-GCL | false negative / hard negative 分离、true-negative probability、uncertainty negative mining 已拥挤。 |
| Node Similarity / IFL-GCL | point posterior、node similarity distribution、PU-GCL 已覆盖大量 pair scoring 空间。 |
| IGT-GCL | interval / lower-upper target 仍被 reviewer 判为类似双阈值 reweighting。 |
| BGRL / CCA-SSG / Graph Barlow Twins | negative-free 目标已能绕开 negatives，DSR 必须证明 residual contrast 有额外价值。 |
| ReGCL | 已从 InfoNCE gradient conflict 角度分析 GCL，DSR 必须区别于 gradient-weighted InfoNCE。 |
| SPGCL 2026 | Dirichlet energy + positive pre-alignment 是强近邻，DSR 必须强调 negative repulsion leakage。 |
| ASPECT 2026 | spectral low/high fusion 是强近邻，DSR 必须强调 loss/gradient routing 而非 spectral view fusion。 |
| GraphRank | margin/rank loss 已缓解 excessive negative separation，DSR 不能只改成 rank loss。 |

核心 gap 从“识别 false-negative pair”转移为：

> 如何阻止 false-negative repulsive gradients 破坏节点分类需要的语义子空间，同时保留 contrastive objective 的 uniformity / boundary 好处？

## 4. Ranked Ideas

| Rank | Idea | Status | Why |
|---|---|---|---|
| 1 | **DSR-GCL / Spectral Gradient Firewall** | **REVISE_IDEA, recommended for refinement** | 跳出 IGT 的 pair scoring/interval 路线，从表示子空间和梯度权限处理 false-negative damage。 |
| 2 | CNG-GCL / Conflict-Negative Graph Selection | Backup | 把 negative mining 改成 anchor-local conflict graph 的组合选择，不做 pair posterior；但 E2Neg 近邻风险较高。 |
| 3 | MPC-GCL / Masked Positive Completion | Backup | 用 masked context completion 增加非平凡 positives，缓解 positive scarcity；但 GraphMAE/GCMAE 近邻强。 |
| 4 | ROT-GCL / Regularized OT Assignment | Backup | 用 balanced prototype assignment 替代 instance-level all-vs-all repulsion；但 SwAV/PCL/LR-GCL 覆盖风险高。 |
| 5 | GO-GCL / Gradient Orthogonalization | High-risk backup | 更直接的梯度正交化版本，工程复杂、稳定性风险高。 |

已排除为主 idea：

- point posterior GCL；
- interval / ambiguity-set GCL；
- abstention / sample reweighting 变体；
- simple positive expansion；
- ordinary false-negative filtering；
- rank loss alone；
- spectral fusion alone；
- GraphMAE + GRACE 简单加权 hybrid。

## 5. Recommended Idea: DSR-GCL

### Problem Anchor

在节点级 GCL 中，false negatives 可能仍会存在于 denominator。与其尝试识别每个 false-negative pair，不如限制它们能破坏的表示方向：不允许 negative repulsion 更新 semantic channel。

### Mechanism

DSR-GCL 使用参数隔离双通道：

```text
h_sem = GNN_sem(X_sem, A)
h_res = GNN_res(X_res, A)
z_sem = P_sem(h_sem)
z_res = P_res(h_res)
z_eval = concat(z_sem, stopgrad(z_res))
```

- `L_sem`: negative-free alignment + variance/covariance guard。
- `L_res`: InfoNCE / rank-style residual contrast。
- `L_orth`: 降低 `z_sem` 与 `z_res` 的冗余。
- negative loss 只更新 residual channel，不更新 semantic channel。

这比 shared encoder + two heads 更严格，直接回应 reviewer 的主要弱点。

### Dominant Contribution

**Repulsive-gradient containment**：把 false-negative problem 从 sample decision 转为 gradient permission / representation channel design。

## 6. Novelty Review

fresh `gcl_scientific_reviewer` 结论：

- Verdict：**REVISE_IDEA**
- Novelty score：**6.2/10**
- 相对 IGT-GCL：小幅强于 `5.5/10`，但不安全强于。
- Confidence：**0.72**

Reviewer 认为 DSR 的优点是跳出 pair posterior / interval / reweighting；主要问题是若使用 shared encoder，negative gradient 仍会污染 semantic branch。因此最终 proposal 改为参数隔离双通道，并把 `||Proj_sem(g_neg)||` 写成必须诊断。

## 7. Required Ablations

| Variant | Purpose |
|---|---|
| GRACE / GCA | contrastive baseline |
| BGRL / CCA-SSG / Graph Barlow Twins | negative-free baseline |
| same-parameter single-head | 排除参数量和双 head 优势 |
| pure semantic-only | 检查 residual 是否必要 |
| residual-only InfoNCE | 检查 semantic 是否必要 |
| semantic branch receives negatives | 检查 firewall 是否必要 |
| no orthogonality | 检查 branch leakage |
| random branch split | 检查 spectral/residual split 是否有效 |
| GraphRank residual loss | 检查是否只是 rank loss |
| SPGCL/ASPECT-style spectral fusion control | 检查是否只是 spectral fusion |

## 8. Experiment Plan Summary

详见：

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`

计划只定义未来 smoke/pilot，不启动实验。核心 gate：

- leakage diagnostic 必须能记录；
- pure negative-free baseline 不能匹配 full DSR；
- semantic branch receives negatives 必须显著伤害机制诊断；
- same-parameter single-head 不能匹配 full DSR；
- random split 不能匹配 spectral/residual split。

## 9. Research Wiki Updates

本轮同步：

- `research-wiki/ideas/dsr_gcl.md`
- `research-wiki/ideas/cng_gcl.md`
- `research-wiki/log.md`
- `AGENTS.md`
- `MANIFEST.md`

## 10. Final Decision

**Final decision: REVISE_IDEA.**

DSR-GCL 比 IGT-GCL 更值得继续 refinement，但当前还不能进入 `GO_TO_EXPERIMENT_BRIDGE`。主要原因：novelty 只有 `6.2/10`，强近邻包括 BGRL/CCA/Barlow、ReGCL、SPGCL、ASPECT、GraphRank 和 low-rank/spectral GCL。下一步应先进一步打磨 gradient firewall 的理论/实现定义，或在明确获得用户批准后做极小 smoke/pilot；不得直接实现 full method、不得跑 formal、不得产生性能 claim。
