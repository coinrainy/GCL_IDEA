# Research Proposal: CAST-GCL

## Problem Anchor

- **Bottom-line problem**：节点级 GCL 把大量真实节点作为 negatives，但其中一部分可能与 anchor 共享下游类别语义，形成 false negatives。
- **Must-solve bottleneck**：不能只改 loss 权重，也不能只生成同一 anchor 的 harder positive view；需要构造可验证的跨节点 semantic relation。
- **Core mechanism**：frozen latent ego target-prediction certificate + mixed-ego cross-node semantic transport + positive / neutral closure。
- **Non-goals**：不使用 test labels；不使用 class-aware counterfactual gradients；不生成 hard negatives 作为主贡献；不把 smoke 写成性能 claim。
- **Constraints**：首轮只做 Cora seed=0 smoke，`1:1:8` split，frozen encoder + Logistic Regression evaluator。

## Method Thesis

**One-sentence thesis**：CAST-GCL treats false negatives as missing semantic transport relations and constructs a certified cross-node positive closure before contrastive training.

## Proposed Method

### 1. Latent Ego Target-Prediction Certificate

Train a WILLOW-style certificate, then freeze it before relation construction:

```text
h_c      = E_online(context_ego_i)
h_t^k    = E_target(masked_target_ego_i^k)
pred_t^k = P(h_c, pos_k)
cert_error = sum_k || pred_t^k - stopgrad(h_t^k) ||_2^2
```

The certificate is only used to score semantic stability of interventions and transport paths. During smoke, closure is precomputed offline and not updated with the contrastive encoder.

### 2. Candidate Pool

For each anchor `i`, create a small candidate pool from label-free signals:

- feature kNN；
- diffusion / personalized PageRank neighborhood；
- current frozen-pretrain embedding kNN；
- random budget-matched candidates。

No label or test-set information is used. Candidate pools provide candidates only; they cannot decide positives without certificate scoring.

### 3. Legal Intervention Operators

All operators act inside a 1-hop or 2-hop ego context under a fixed budget:

| Operator | Definition | Cost |
|---|---|---:|
| `feature_replace(i -> j, S)` | replace anchor-ego feature dimensions `S` with candidate-ego feature statistics | `|S| / d` |
| `edge_keep(i -> j, R)` | keep anchor edges whose endpoint role also appears in candidate ego role set `R` | dropped edge ratio |
| `neighbor_substitute(i -> j, q)` | substitute at most `q` anchor neighbors with candidate neighbors matched by degree bin and feature cosine bin | `q / deg(i)` |
| `ego_mix(i, j, alpha)` | convex mix of normalized ego feature summaries for certificate query only | `alpha` |

### 4. Semantic Transport Energy

For candidate `j`, search a short path of legal local interventions:

```text
i = u_0 -> u_1 -> ... -> u_L = j
```

For a step `u -> v`, compute certificate error on a mixed ego query:

```text
cert_error(u -> v, a)
  = mean_k || P(E_context(ego_mix(u, v, a)), pos_k)
              - stopgrad(E_target(target_ego(v)^k)) ||_2^2
```

Transport energy:

```text
E_transport(i, j)
  = min_path sum_l [
      cert_error(u_l -> u_{l+1})
      + lambda_edit * edit_cost(u_l, u_{l+1})
      + lambda_anchor * anchor_drift(u_l, i)
    ]
```

Smoke uses `L=1` and `L=2` greedy paths only, candidate pool size `K=20`, and max transported positives per anchor `B=5`.

### 5. Positive Closure

Define:

```text
P_i = {j: E_transport(i, j) <= tau_transport}
```

Training relation:

- same-node augmented views remain positives；
- `j in P_i` becomes cross-node positive；
- candidates within `tau_transport * [1.0, 1.15]` become neutral diagnostics；
- remaining nodes are standard negatives。

The main contribution is the relation constructor, not a new contrastive loss.

## Required Controls

| System | Purpose |
|---|---|
| GRACE | base GCL reference |
| WILLOW same-node | certificate without cross-node closure |
| kNN multi-positive | simple positive mining control |
| PMGCL-lite/BMM | probabilistic positive mining control |
| PPR/diffusion positives | graph-proximity positive control |
| candidate-pool-only | candidate source without certificate |
| similarity-only energy | removes certificate and edit path |
| edit-cost-only / anchor-drift-only | energy component controls |
| certificate-shuffled CAST | semantic certificate control |
| random transport budget-matched | search budget control |
| CAST one-step / two-step | main variants |

## Metrics

- Frozen Logistic Regression accuracy。
- transported positives per anchor。
- offline label agreement of transported positives。
- transport energy vs label agreement correlation。
- false-negative repulsion mass before/after closure。
- positive gradient norm。
- search time and candidate budget。

## Kill Rules

- kNN multi-positive matches CAST：transport certificate unnecessary。
- PMGCL-lite/BMM matches CAST：CAST reduces to ordinary positive mining。
- PPR/diffusion positives match CAST：CAST reduces to graph-proximity positives。
- candidate-pool-only or similarity-only matches CAST：certificate/path adds no value。
- certificate-shuffled CAST matches CAST：certificate has no semantic content。
- random transport matches CAST：path search is not meaningful。
- transport energy is uncorrelated with offline label agreement：kill。
- WILLOW same-node matches CAST：cross-node closure unnecessary。

## Current Verdict

`REVISE_TO_CAST_REVISED_PRE_SMOKE`。  
这是 proposal evidence，不是实验结果。下一步只允许最小 smoke；不允许 formal 或性能 claim。
