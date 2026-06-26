# DSR-GCL Novelty Brief

生成时间：2026-06-26T09:24:55Z  
阶段：fresh `/idea-discovery` beyond IGT-GCL  
对象：**DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning**  
核心机制别名：**Spectral Gradient Firewall**

## 1. Motivation

IGT-GCL 的 novelty score 只有 `5.5/10`，问题在于仍然围绕 pair posterior / interval / abstention / sample reweighting。DSR-GCL 明确避开这条低新颖性路线。

本方法的核心判断是：节点级 GCL 的 false negative 问题不仅在于“哪些 pair 被误当负样本”，还在于 **InfoNCE 的 repulsive gradient 被允许作用于整条表示向量**。当同类或同语义节点被当作 negative 时，repulsion 会撕裂节点分类所需的 class-semantic / low-frequency manifold。

## 2. Proposed Method

DSR-GCL 将表示拆成两个互补分支：

```text
z = [z_sem || z_res]
```

- `z_sem`: semantic / class-manifold branch。承载节点分类中常见的低频、平滑、类语义信息。该分支不接收 node-node negative repulsion。
- `z_res`: residual / boundary branch。承载实例去冗余、局部边界、高频残差信息。InfoNCE 或 rank-style negative pressure 只在该分支上生效。

训练目标：

1. **Semantic branch loss**  
   `z_sem` 使用 negative-free cross-view consistency，并配合 VICReg/Barlow-style variance-covariance guard 或 lightweight masked-completion guard，避免 collapse。

2. **Residual branch loss**  
   `z_res` 使用 GRACE-style InfoNCE、rank loss 或 small-negative objective 维持 uniformity 和局部边界区分。

3. **Spectral Gradient Firewall**  
   用 graph Laplacian / Dirichlet energy 或 learnable orthogonal projection 约束：来自 negative denominator 的 repulsive gradient 不进入 `z_sem`，只进入 `z_res`。实现上可以是 two-head stop-gradient routing，也可以显式记录 `||Proj_sem(g_neg)||` 并最小化。

4. **Downstream representation**  
   主 evaluator 优先使用 `z_sem`；`concat(z_sem, stopgrad(z_res))` 作为诊断/备选。具体选择必须在 train/val 上固定，不能用 test 调整。

## 3. Why This Is Not IGT-GCL

DSR-GCL 不做：

- point posterior；
- lower/upper interval；
- ambiguous abstention；
- false-negative filtering；
- negative downweight；
- positive expansion；
- pair action set。

它不试图判断某个 negative 是否为假负，而是改变 **false-negative repulsion 可以破坏的表示子空间**：

```text
false negatives may still exist,
but they cannot tear apart the class-semantic branch.
```

## 4. Closest Prior Work and Required Delta

| Prior work | Overlap | Required delta for DSR-GCL |
|---|---|---|
| BGRL / CCA-SSG / Graph Barlow Twins / VICReg | negative-free branch and collapse prevention | DSR must show negative-free alone is insufficient; residual branch provides uniformity while semantic branch is protected from false-negative repulsion. |
| SPGCL, 2026 | Dirichlet energy, message-passing pre-alignment, high/low energy feature handling | DSR is not positive sampling or energy-aware propagation alone; it routes objectives/gradients by semantic-residual subspace to contain false-negative damage. |
| ASPECT, 2026 | low/high frequency views and node-wise spectral fusion | DSR is not spectral view fusion; it assigns different losses/gradient permissions to semantic vs residual channels. |
| ReGCL, AAAI 2024 | InfoNCE gradient conflict in GNN message passing | DSR is not gradient-weighted InfoNCE or structure learning; it firewall-protects the semantic branch from denominator repulsion. |
| GraphRank | margin/rank objective reduces excessive negative separation | DSR can use rank loss only inside residual branch; main novelty is branch-level repulsion containment. |
| LR-GCL / low-rank GCL | low-rank/prototype representation for node classification | DSR is not just low-rank regularization; it must diagnose and reduce negative-gradient leakage into the semantic manifold. |
| Zero-Mean Spectral CL | relaxes negative-pair orthogonality in spectral contrastive loss | DSR is graph/node-classification-specific loss-channel routing, not global zero-mean regularization. |

## 5. Core Claims to Review

1. **False-negative damage containment**：即使 false negatives 仍存在，repulsion 被限制在 residual branch，不再撕裂 class-semantic branch。
2. **Objective alignment**：semantic branch 更接近节点分类所需 class manifold；residual branch 单独承担 instance uniformity。
3. **Mechanism beyond negative-free**：相比纯 BGRL/CCA/VICReg，DSR 保留 residual contrast，因此不牺牲 uniformity / discriminative boundary。
4. **Mechanism beyond spectral fusion**：相比 SPGCL/ASPECT，DSR 的贡献是 loss/gradient routing，而不是频谱 view 选择。

## 6. Mandatory Ablations

- vanilla GRACE / GCA；
- BGRL / CCA-SSG / Graph Barlow Twins style negative-free baseline；
- residual-only InfoNCE；
- same-parameter single-head control；
- semantic branch receives negatives；
- residual branch removed；
- no spectral/orthogonal constraint；
- random branch split；
- `z_sem` vs `z_res` vs `concat(z_sem, stopgrad(z_res))` evaluator；
- gradient leakage diagnostic: negative-gradient norm projected onto semantic branch。

## 7. Minimum Experiments Before Any Claim

只允许 smoke/pilot/development，不支持 SOTA claim：

- Cora/CiteSeer/PubMed under stratified random `1:1:8`；
- Amazon-Computers/Photo under same `1:1:8`；
- optional heterophily fixed split only if already supported；
- GCL frozen encoder + Logistic Regression evaluator；
- labels only for downstream evaluator and offline diagnostics。

Key diagnostics：

- `z_sem` effective rank / variance / covariance；
- class-wise and degree-wise Logistic Regression accuracy；
- alignment/uniformity for `z_sem` and `z_res` separately；
- negative-gradient leakage into `z_sem`；
- whether adding negatives to `z_sem` hurts same-label compactness；
- whether residual branch improves uniformity compared with pure negative-free baseline。

## 8. Stop Rules

- If pure negative-free baseline matches DSR, DSR novelty collapses.
- If same-parameter single-head matches DSR, branch decoupling is not useful.
- If `z_sem` collapses or has low effective rank, kill or redesign collapse guard.
- If negative-gradient leakage diagnostic does not differ from GRACE, Spectral Gradient Firewall is not working.
- If gains come only from larger projector or extra loss weights, no mechanism claim.
- If SPGCL/ASPECT-style spectral fusion matches DSR, pivot to spectral fusion instead of claiming DSR.

## 9. Desired Reviewer Output

Please review DSR-GCL as a fresh `gcl_scientific_reviewer`. Output:

1. Verdict: GO_TO_EXPERIMENT_BRIDGE / REVISE_IDEA / PIVOT_REQUIRED / KILL.
2. Novelty score: 0-10, and whether it is clearly stronger than IGT-GCL's 5.5/10.
3. Closest 6 prior works with overlap and remaining delta.
4. Fatal weaknesses and strongest alternative explanations.
5. Required refinement before writing `FINAL_PROPOSAL.md`.
6. Mandatory ablations and forbidden claims.
7. One-sentence final recommendation.
