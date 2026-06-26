# FINAL_PROPOSAL: DSR-GCL

标题：**DSR-GCL: Decoupled Semantic-Residual Graph Contrastive Learning**  
核心机制：**Spectral Gradient Firewall**  
任务：图对比学习用于节点分类  
状态：reviewer-informed revised idea  
decision：**REVISE_IDEA**  
证据等级：idea / literature / review only；无实验结果，无性能 claim。

## 1. Problem Anchor

节点级 GCL 的 false-negative 问题不只是“哪些其他节点被误当成负样本”。更深的错位是：

> InfoNCE 的 negative repulsion 会作用在整条节点表示上，因此当同类或同语义节点被当作负样本时，repulsive gradient 会撕裂节点分类所需的 class-semantic manifold。

IGT-GCL 尝试对 pair 做 interval / action set，但 reviewer 认为其新颖性只有 `5.5/10`，仍接近 point posterior、negative downweight 和 abstention。DSR-GCL 改为从 **表示子空间和梯度权限** 处理 false-negative damage，而不是继续判断 pair 语义。

## 2. Method Thesis

DSR-GCL 将表示学习拆成两个参数隔离的通道：

```text
z_sem = E_sem(view)
z_res = E_res(view)
z_final = concat(z_sem, stopgrad_or_scaled(z_res))
```

- `E_sem`：semantic channel，用 negative-free objective 学稳定、非塌缩的节点分类候选流形。
- `E_res`：residual channel，用 InfoNCE / rank-style objective 学局部边界、实例去冗余和 uniformity。
- **Spectral Gradient Firewall**：negative loss 只允许更新 `E_res`，不反传到 `E_sem`；两个通道的参数在 branch split 后隔离，并通过 orthogonality / covariance penalty 降低信息泄漏。

核心不是“这个 pair 是不是 false negative”，而是：

```text
false negatives may remain in the denominator,
but their repulsive gradient cannot update the semantic channel.
```

## 3. Concrete Architecture

### 3.1 Branch Inputs

给定图 `G=(X,A)`，构造两个轻量输入通道：

- `X_sem`: low-pass / smooth input candidate，例如 `P_L X`、diffusion-smoothed features，或原始 `X` 加低频正则。
- `X_res`: residual / high-frequency input candidate，例如 `X - P_L X`、edge-local residual，或 learned residual adapter。

`P_L` 可用少阶 Chebyshev / normalized adjacency diffusion 近似，避免显式特征分解。

### 3.2 Parameter Isolation

为避免 reviewer 指出的 shared-encoder leakage，默认实现使用两个小宽度 GNN/adapters：

```text
h_sem = GNN_sem(X_sem, A_sem)
h_res = GNN_res(X_res, A_res)
z_sem = P_sem(h_sem)
z_res = P_res(h_res)
```

约束：

- `L_sem` 只更新 `GNN_sem, P_sem`。
- `L_res` 只更新 `GNN_res, P_res`。
- 不共享 message-passing weights。
- 若未来实现 shared trunk，必须显式做 gradient projection；首版不采用 shared trunk。
- 总 hidden dimension 与 baseline 对齐，例如 `dim_sem + dim_res = dim_GRACE`，防止参数量优势。

### 3.3 Semantic Loss

semantic channel 不使用 node-node negatives。默认：

```text
L_sem = L_align(z_sem^1, z_sem^2)
      + lambda_var * L_variance(z_sem^1, z_sem^2)
      + lambda_cov * L_covariance(z_sem^1, z_sem^2)
```

可选加入 lightweight masked-completion guard，但首版应优先使用 VICReg/Barlow-style guard，降低工程面。

### 3.4 Residual Loss

residual channel 承担局部边界和 uniformity：

```text
L_res = InfoNCE(z_res^1, z_res^2)
```

或 GraphRank-style margin loss作为消融。`L_res` 只能更新 residual channel。

### 3.5 Orthogonality and Leakage Diagnostics

训练加一个轻量正交约束：

```text
L_orth = || Corr(z_sem, z_res) ||_F^2
```

总 loss：

```text
L = L_sem + alpha * L_res + beta * L_orth
```

必须记录：

```text
leakage = || Proj_{z_sem}(grad_negative) ||
```

在参数隔离首版中，`leakage` 对 `E_sem` 应接近 0；若非 0，说明实现违反 firewall。

## 4. Downstream Evaluation

主评估嵌入必须预注册：

```text
z_eval = concat(z_sem, stopgrad(z_res))
```

同时报告诊断：

- `z_sem` only；
- `z_res` only；
- concat。

但主表口径不能由 test 选择。若后续想让 train/val 选择 `z_eval` 形式，必须在 protocol 中预注册并对所有 baselines/variants一致处理。

GCL evaluator 仍是 frozen encoder + Logistic Regression，`C` 等超参只由 train/val 或 train-only CV 选择。

## 5. Novelty Positioning

相对 IGT-GCL：DSR 不构造 pair interval / posterior / abstention，而是限制 negative gradient 的作用子空间。

相对 BGRL / CCA-SSG / Graph Barlow Twins：DSR 不只做 negative-free；它保留 residual contrast 来维持 uniformity 和边界信号，同时防止 residual negative loss 更新 semantic channel。

相对 ReGCL：DSR 不做 gradient weighting 或结构学习；它做参数/子空间级 gradient permission，并直接诊断 negative-gradient leakage。

相对 SPGCL / ASPECT：DSR 不主张频谱 view fusion 或 positive sampling，而是将不同 loss 分配到语义/残差通道，核心是 false-negative damage containment。

相对 GraphRank：DSR 可把 rank loss作为 residual channel 消融，但主贡献不是 margin loss，而是 semantic branch 的 repulsion firewall。

## 6. Required Ablations

| Ablation | Purpose |
|---|---|
| GRACE / GCA | contrastive baseline |
| BGRL / CCA-SSG / Barlow-style | negative-free baseline |
| same-parameter single-head | 排除参数量和双 head 优势 |
| pure `L_sem` only | 检查 residual branch 是否必要 |
| pure `L_res` only | 检查 semantic branch 是否必要 |
| semantic branch receives negatives | 检查 firewall 是否核心 |
| no residual branch | 检查 residual uniformity |
| no orthogonality | 检查 branch leakage |
| random branch split | 检查 spectral/semantic split 是否有效 |
| GraphRank residual loss | 检查是否只是 margin loss |
| SPGCL/ASPECT-style spectral fusion control | 检查是否只是 spectral fusion |
| `z_sem` / `z_res` / fixed concat evaluator | 诊断分支作用 |

## 7. Stop Rules

- pure negative-free baseline 匹配 full DSR：`PIVOT_REQUIRED`。
- same-parameter single-head 匹配 full DSR：`KILL` or `PIVOT_REQUIRED`。
- semantic branch receives negatives 不变差：firewall claim 不成立。
- `leakage` diagnostic 无法接近 0：实现机制失败。
- `z_sem` collapse 或 effective rank 很低：`KILL` or redesign collapse guard。
- 收益只在强同配图出现，Amazon/heterophily 明显下降：只能保留窄 claim 或 pivot。

## 8. Current Decision

Reviewer verdict：**REVISE_IDEA**。  
Novelty score：**6.2/10**，小幅强于 IGT-GCL 的 `5.5/10`，但不足以进入 `GO_TO_EXPERIMENT_BRIDGE`。

当前最合理动作是继续 refinement 或做极小 smoke 设计审查；不要直接实现 full pipeline，不要跑 formal，不要产生性能 claim。
