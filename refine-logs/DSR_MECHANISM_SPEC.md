# DSR-GCL Mechanism Spec

生成时间：2026-06-26T09:40:55Z  
对象：DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning  
核心机制：Spectral Gradient Firewall  
阶段：mechanism refinement + fix-audit-smoke  
状态：已完成 Cora seed=0 fix-audit-smoke；不进入 formal，不产生性能 claim。

## 1. Fixed Problem Anchor

节点级 GCL 的 false-negative 问题不只是“哪些其他节点被误当负样本”，而是 InfoNCE 的 negative repulsion 会作用在整条节点表示上。当同类或同语义节点被当作 negative 时，repulsive gradient 会撕裂节点分类所需的 semantic manifold。

DSR-GCL 的目标不是识别 false-negative pair，而是限制 negative repulsion 的作用通道：

```text
negative repulsion may shape residual/boundary representation,
but must not update semantic representation.
```

## 2. Design Decision: Parameter-Isolated Firewall

本轮 refinement 固定首版机制为 **parameter-isolated two-channel firewall**，不用 shared trunk。

原因：reviewer 指出 shared encoder + two heads 不是 firewall，因为 residual InfoNCE 仍会通过 shared encoder 间接污染 semantic branch。

因此首版实现必须满足：

```text
Params(E_sem) ∩ Params(E_res) = ∅
Params(P_sem) ∩ Params(P_res) = ∅
L_res.backward() updates only E_res, P_res
L_sem.backward() updates only E_sem, P_sem
L_orth updates projection heads only, or uses stop-gradient on one side
```

如果以后尝试 shared trunk，必须单独作为 variant `shared_trunk_projected_grad`，不得作为默认 DSR。

## 3. Notation

Graph:

```text
G = (X, A), X ∈ R^{N×F}
```

Two graph augmentations:

```text
(X^1, A^1), (X^2, A^2) = aug(G)
```

Low-pass operator:

```text
S = D^{-1/2}(A + I)D^{-1/2}
P_L(X, A; K) = S^K X
```

Implementation audit note:

- PyG edge dropout can remove only one direction of an originally undirected edge.
- If the post-dropout `edge_index` is not symmetric, the spectral interpretation
  of `S` as an undirected normalized adjacency is not strict.
- The smoke runner therefore exposes `make_undirected_after_dropout` for the
  DSR low-pass/residual input construction. When enabled, dropped edges are
  symmetrized before self-loops and normalization.
- The current implementation aggregates messages from source `row` to target
  `col` and computes degrees on `col`, matching the source-to-target GCN
  convention. With a symmetrized graph, row/col degree ambiguity disappears.

Residual input:

```text
P_R(X, A; K) = X - stopgrad(P_L(X, A; K))
```

The stop-gradient in `P_R` prevents residual loss from updating low-pass preprocessing if it is implemented as a learnable adapter later.

Channels:

```text
h_sem^v = E_sem(P_L(X^v, A^v; K), A^v)
h_res^v = E_res(P_R(X^v, A^v; K), A^v)
p_sem^v = P_sem(h_sem^v)
p_res^v = P_res(h_res^v)
z_sem^v = normalize(p_sem^v)
z_res^v = normalize(p_res^v)
```

where `v ∈ {1,2}`.

Default dimensions:

```text
d_sem + d_res = d_base
d_sem = d_res = d_base / 2
```

If `d_base` is odd, set `d_sem = ceil(d_base/2)` and `d_res = floor(d_base/2)`. Parameter count must be logged and matched against same-parameter controls.

## 4. Loss Definitions

### 4.1 Semantic Branch Loss

Semantic branch uses negative-free invariance plus collapse guard.

Important implementation rule:

```text
VICReg-style alignment / variance / covariance are computed on raw projected
features p_sem, not on L2-normalized unit vectors z_sem.
```

Reason: if `L_var` is applied to unit-normalized vectors, `γ=1.0` is almost
unreachable for high-dimensional features and the variance regularizer becomes
miscalibrated. Normalization is reserved for InfoNCE and auxiliary projected
diagnostics.

Symmetric alignment:

```text
L_align = (1/N) Σ_i || p_sem,i^1 - p_sem,i^2 ||_2^2
```

Variance regularizer, VICReg-style:

```text
std_j(Z) = sqrt(Var_i(Z_{ij}) + eps)
L_var(Z) = (1/d) Σ_j max(0, γ - std_j(Z))
L_var = L_var(p_sem^1) + L_var(p_sem^2)
```

Covariance regularizer:

```text
C(Z) = (Z - mean(Z))^T (Z - mean(Z)) / (N - 1)
L_cov(Z) = (1/d) Σ_{p≠q} C(Z)_{pq}^2
L_cov = L_cov(p_sem^1) + L_cov(p_sem^2)
```

Semantic loss:

```text
L_sem = L_align + λ_var L_var + λ_cov L_cov
```

Default:

```text
λ_var = 1.0
λ_cov = 1.0
γ = 1.0
eps = 1e-4
```

These defaults may be tuned only on train/val during pilot; test cannot influence them.

### 4.2 Residual Branch Loss

Residual branch uses InfoNCE on residual projected features. The implementation
passes raw `p_res` to InfoNCE and InfoNCE normalizes internally:

For anchor `i` in view 1:

```text
ℓ_i^{1→2} = -log exp(sim(normalize(p_res,i^1), normalize(p_res,i^2))/τ)
             / Σ_{j=1}^N exp(sim(normalize(p_res,i^1), normalize(p_res,j^2))/τ)
```

Symmetric:

```text
L_res = (1/2N) Σ_i (ℓ_i^{1→2} + ℓ_i^{2→1})
```

Default:

```text
τ = baseline GRACE temperature unless variant overrides it.
```

Important firewall rule:

```text
∂L_res/∂θ_sem = 0
∂L_res/∂θ_res ≠ 0
```

### 4.3 Branch Orthogonality / Redundancy Loss

Use centered cross-correlation between branches:

```text
Q = Corr(stopgrad(z_sem), z_res)
L_orth = ||Q||_F^2
```

Default direction:

```text
L_orth updates residual branch only.
```

Reason: semantic channel should remain the protected channel. Updating `E_sem` with `L_orth` could create a secondary path for residual pressure. A variant may update both sides, but default does not.

### 4.4 Total Loss

```text
L_total = L_sem + α L_res + β L_orth
```

Default:

```text
α = 1.0
β = 0.01
```

`α` and `β` are not claims; they are pilot defaults and must be logged.

## 5. Gradient Firewall Definition

Let:

```text
θ_sem = Params(E_sem, P_sem)
θ_res = Params(E_res, P_res)
g_neg = ∇_{θ}(L_res)
```

The firewall is satisfied iff:

```text
|| ∇_{θ_sem}(L_res) ||_2 = 0
```

up to numerical tolerance.

Implementation-level diagnostic:

```text
firewall_leak_param =
  || grad(θ_sem; L_res) ||_2 / (|| grad(θ_res; L_res) ||_2 + eps)
```

Expected default:

```text
firewall_leak_param < 1e-8
```

Representation-level diagnostic:

```text
firewall_leak_repr =
  || J_sem^T ∇_{h_shared}(L_res) ||
```

For parameter-isolated default, there is no shared representation, so this is `not_applicable`. For shared-trunk variants, it is required.

## 6. Forward / Training Pseudocode

```python
def train_step(data):
    x, edge_index = data.x, data.edge_index

    # 1. Two standard graph augmentations, matched to GRACE/GCA budget.
    x1, edge1 = augment(x, edge_index)
    x2, edge2 = augment(x, edge_index)

    # 2. Low-pass and residual inputs.
    x1_sem = low_pass(x1, edge1, K)
    x2_sem = low_pass(x2, edge2, K)
    x1_res = x1 - stopgrad(x1_sem)
    x2_res = x2 - stopgrad(x2_sem)

    # 3. Parameter-isolated encoders.
    h1_sem = E_sem(x1_sem, edge1)
    h2_sem = E_sem(x2_sem, edge2)
    p1_sem = P_sem(h1_sem)
    p2_sem = P_sem(h2_sem)
    z1_sem = normalize(p1_sem)
    z2_sem = normalize(p2_sem)

    h1_res = E_res(x1_res, edge1)
    h2_res = E_res(x2_res, edge2)
    p1_res = P_res(h1_res)
    p2_res = P_res(h2_res)
    z1_res = normalize(p1_res)
    z2_res = normalize(p2_res)

    # 4. Semantic negative-free objective.
    loss_align = mse(p1_sem, p2_sem)
    loss_var = vicreg_variance(p1_sem) + vicreg_variance(p2_sem)
    loss_cov = vicreg_covariance(p1_sem) + vicreg_covariance(p2_sem)
    loss_sem = loss_align + lambda_var * loss_var + lambda_cov * loss_cov

    # 5. Residual contrast. This graph must not touch theta_sem.
    loss_res = info_nce(p1_res, p2_res, tau)  # normalizes internally

    # 6. Branch orthogonality, default updates residual only.
    loss_orth = corr_frobenius(stopgrad(z1_sem), z1_res)
    loss_orth += corr_frobenius(stopgrad(z2_sem), z2_res)

    # 7. Total loss.
    loss = loss_sem + alpha * loss_res + beta * loss_orth

    optimizer.zero_grad()
    loss.backward()

    # 8. Diagnostics before optimizer step.
    diag = {}
    diag["firewall_leak_param"] = grad_norm(theta_sem, from_loss="loss_res") / (
        grad_norm(theta_res, from_loss="loss_res") + eps
    )
    diag["rank_sem"] = effective_rank(cat(z1_sem, z2_sem))
    diag["rank_res"] = effective_rank(cat(z1_res, z2_res))
    diag["corr_sem_res"] = corr_frobenius(z1_sem.detach(), z1_res.detach())

    optimizer.step()
    return loss.item(), diag
```

Practical note: to compute `firewall_leak_param`, run an auxiliary backward or `torch.autograd.grad(loss_res, params, retain_graph=True, allow_unused=True)` before the main backward. Missing gradients for `θ_sem` should be recorded as zero.

## 7. Inference / Evaluation Definition

Default DSR audit evaluator embedding:

```text
h_eval = concat(h_sem, stopgrad(h_res))
```

Rationale:

- GRACE baseline evaluates the encoder output, not the projection head.
- Therefore DSR audit-smoke must report encoder-level `h_*` embeddings as the
  primary fairness view.
- Projected normalized `z_*` embeddings are still logged as auxiliary
  diagnostics, but they are not the default main table setting.
- `stopgrad` is irrelevant after pretraining but documents that evaluator must not fine-tune.

Required diagnostics:

```text
h_eval_sem = h_sem
h_eval_res = h_res
h_eval_concat = concat(h_sem, h_res)
z_eval_sem = normalize(p_sem)
z_eval_res = normalize(p_res)
z_eval_concat = concat(z_sem, z_res)
```

Main pilot may report all three, but the main method setting must be pre-registered before test evaluation. Recommended pilot default is concat; if concat underperforms but `z_sem` wins, treat that as a refinement signal rather than silently switching the method.

## 8. Minimal Implementation Switches

Use explicit config switches:

```yaml
method: dsr_gcl
base_dim: 256
sem_dim: 128
res_dim: 128
low_pass_k: 1
semantic_loss: vicreg
residual_loss: infonce
orth_updates: residual_only
eval_embedding: h_concat
alpha_res: 1.0
beta_orth: 0.01
firewall: parameter_isolated
```

Variant switches:

```yaml
variant:
  semantic_receives_negatives: false
  no_residual_branch: false
  no_orthogonality: false
  random_branch_split: false
  same_param_single_head: false
  residual_loss_type: infonce
```

## 9. Ablation Matrix

| ID | Variant | Exact change | Tests which claim | Expected diagnostic if DSR is real | Kill/Pivot signal |
|---|---|---|---|---|---|
| A0 | GRACE | Original InfoNCE on one representation | Baseline false-negative exposure | Higher semantic negative-gradient damage | DSR no better in any diagnostic |
| A1 | GCA | Adaptive augmentation baseline | Stronger GCL baseline | DSR should remain competitive in diagnostics | DSR only beats weak GRACE |
| A2 | Negative-free | `L_sem` only with same dim/params | Residual necessity | Lower uniformity / weaker boundary signal than full | Matches full DSR |
| A3 | Residual-only | `L_res` only on residual branch | Semantic necessity | Worse class-smooth diagnostics | Matches full DSR |
| A4 | Same-param single-head | One encoder/projector, same params, mixed loss | Branch decoupling | Higher leakage / less stable semantic rank | Matches full DSR |
| A5 | No firewall | Apply InfoNCE to `z_sem` too | Firewall necessity | `z_sem` compactness/rank worsens | No difference from full |
| A6 | No orthogonality | Set `β=0` | Leakage/redundancy control | Higher sem-res correlation | Matches or beats full |
| A7 | Random branch split | Random feature split instead of low-pass/residual | Spectral split utility | Weaker diagnostics than low/res split | Matches full |
| A8 | Shared trunk | Shared GNN + two heads | Reviewer leakage risk | Non-zero `firewall_leak_repr` | Matches full, then default can simplify |
| A9 | GraphRank residual | Replace residual InfoNCE with rank loss | Not just InfoNCE | Similar firewall, different residual pressure | Rank alone explains gains |
| A10 | Spectral fusion control | ASPECT/SPGCL-style fuse low/high views, no firewall | Not just spectral fusion | DSR has lower leakage | Fusion matches full |
| A11 | Larger projector control | Increase GRACE projector to match params | Parameter count | No mechanism diagnostics | Matches full DSR |

Minimum smoke variants:

```text
A0, A2, A5, A9
```

Minimum pilot variants:

```text
A0, A1, A2, A3, A4, A5, A6, A7, A9, A11
```

## 10. Implementation Readiness Checklist

Before coding:

- [ ] Decide `base_dim`, `sem_dim`, `res_dim` from existing GRACE config.
- [ ] Decide low-pass operator `P_L`; default `S^1 X`.
- [ ] Decide whether `E_sem` and `E_res` each use half hidden dim or half projection dim.
- [ ] Write config variants A0-A11 before running.
- [ ] Ensure `firewall_leak_param` can be computed with `autograd.grad`.
- [ ] Ensure raw JSON logs include all diagnostics.
- [ ] Pre-register `eval_embedding = concat`.

## 11. Current Decision

**REVISE_IMPLEMENTATION_BEFORE_PIVOT**。

上一轮 `PIVOT_REQUIRED` 暂时降级，因为发现并修复了：

- VICReg 作用在 L2-normalized unit vectors 上；
- DSR 主评估使用 projector-level `z_concat`，而 GRACE 使用 encoder output；
- 公式写了 stop-gradient alignment，但实现没有 predictor/target stop-gradient 结构。

修复后的 Cora seed=0 audit-smoke 仍不支持 formal 或性能 claim；如果继续，只能先做更小范围的 residual / low-pass 实现审计。
