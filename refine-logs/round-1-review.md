# Round 1 Method Review

Reviewer: fresh `gcl_scientific_reviewer`  
Trace: `.aris/traces/research-refine/2026-06-26_run01/`  
Verdict: `REVISE`  
Overall score: `6.7/10`

## Score Table

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.0 |
| Method Specificity | 6.5 |
| Contribution Quality | 5.5 |
| Frontier Leverage | 7.0 |
| Feasibility | 8.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 5.5 |

## Main Required Changes

1. Delete three-way routing; use two-way routing only: full InfoNCE vs positive-only stop-gradient consistency.
2. Define the ineligible objective mathematically so it is not reducible to scalar loss weighting.
3. Remove `s_loss` from the gate to avoid hard-node / uncertainty weighting collision; keep it only as diagnostic.
4. Reframe eligibility as an alignment treatment-risk proxy.
5. Fixed quantile threshold in pilot; no threshold search.

## Raw Reviewer Response

<details>
<summary>Round 1 Review</summary>

**Verdict**
REVISE。项目决策映射也是 `REVISE`，不是 `GO/READY`。CER-GCL 已经比 BCE-GCL 聚焦很多，问题锚点基本正确，但当前还不能直接进入 experiment planning；需要先把核心机制从“heuristic score + routing + loss reweighting”收紧成一个可区分、可实现、可被证伪的 objective-routing 机制。

**Scores**
| 维度 | 分数 |
|---|---:|
| Problem Fidelity | 8.0 |
| Method Specificity | 6.5 |
| Contribution Quality | 5.5 |
| Frontier Leverage | 7.0 |
| Feasibility | 8.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 5.5 |

Overall score: **6.7 / 10**

Drift Warning: **NONE**。当前 proposal 保住了“无监督 contrastive eligibility diagnostic + minimal objective routing”的问题锚点，没有退回原始 boundary-aware 叙事。但 `s_loss/hardness` 若成为主信号，会重新滑向 hard-node reweighting / uncertainty weighting。

Fatal Weaknesses:

1. `eligibility` 还不是一个清晰的机制量。当前 `s_instability + s_fs + s_loss` 更像 proxy 拼接，而不是“强 positive alignment 是否有害”的可操作估计。
2. `weak consistency` 和 `stability-only` 没有精确定义。若只是降低 InfoNCE 权重或温度，routing 就会被 reviewer 判成 scalar reweighting。
3. 技术 delta 仍然薄。UGCL 已有 sample uncertainty + augmentation + weighted GCL loss，HAR/SHARP 已有 hard-to-learn node reweighting，ASPECT 已有 node-wise policy/stability-aware routing，BES 又压住了 boundary 叙事。
4. 当前贡献更像“diagnostic framing + ablation discipline”，还不像强 ML/graph venue 的独立方法贡献。

Simplification Opportunities:

1. 删除三路 routing，先做二路：`InfoNCE` vs `positive-only consistency`。
2. 删除 `s_loss` 作为 gate 输入，避免 hard-node weighting 碰撞；保留为 diagnostic。
3. 固定 quantile，不在 pilot 中搜索阈值。

一句话结论：不允许按当前版本直接进入 experiment planning；先 REVISE 方法定义，尤其把 objective routing 与 scalar reweighting 在公式上切开，再进入最小 pilot 规划。

</details>

