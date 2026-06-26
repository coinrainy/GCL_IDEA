# Research Proposal: R2-IRIS / Residualized Response Invariant Signatures

## Problem Anchor

- **Bottom-line problem**：节点级 GCL 会把潜在同类节点当 negatives；仅靠相似度、近邻或 loss weighting 容易把 false-negative 发现退化为 positive mining。
- **Must-solve bottleneck**：需要一个不依赖 pair proximity 的 semantic relation criterion。
- **Core mechanism**：interventional response fingerprint + residualized response-invariant sibling discovery + multi-positive / neutral closure。
- **Non-goals**：不改 InfoNCE denominator 作为主贡献；不使用 test labels；不使用 learned augmentation policy；不把 smoke 写成性能 claim。
- **Constraints**：首轮只做 Cora seed=0 smoke，`1:1:8` split，frozen encoder + Logistic Regression evaluator。

## Method Thesis

**One-sentence thesis**：R2-IRIS discovers false negatives by identifying nodes whose interventional response similarity remains unusually high after controlling feature similarity, embedding similarity, graph proximity, and degree.

## Proposed Method

### 1. Fixed Intervention Battery

Define a label-free intervention set `Omega` before training:

- spectral high-frequency perturbation；
- low-pass / diffusion-scale perturbation；
- ego edge drop / keep；
- feature group mask；
- neighbor role swap；
- degree-bin preserving ego rewiring。

These interventions simulate environmental shifts. They are fixed, not learned.

### 2. Frozen Warm-up Encoder

Warm up a base encoder `E` with GRACE or another fixed GCL baseline. Freeze it for response signature construction.

### 3. Minimal Response Fingerprint

For node `i` and intervention `omega`:

```text
r_i(omega)
  = [
      delta z_i(omega),
      delta local_smoothness_i(omega),
      delta ego_prediction_consistency_i(omega)
    ]
```

The reviewer explicitly required removing `positive-gradient proxy` from the minimal fingerprint to avoid circular loss-dynamics mining.

Construct:

```text
R_i = concat_{omega in Omega} whiten(r_i(omega))
```

### 4. Residualized Response Siblings

The first IRIS smoke showed that hard anti-proximity filtering destroys sibling quality. The revised rule replaces hard exclusion with pair-level residualization:

```text
sim_response(R_i, R_j)
  = beta_0
  + beta_1 sim_feature(x_i, x_j)
  + beta_2 sim_embedding(z_i, z_j)
  + beta_3 graph_proximity(i, j)
  + beta_4 |degree_i - degree_j|
  + residual_response(i, j)
```

Node `j` becomes an R2-IRIS sibling for anchor `i` if `residual_response(i, j)` is among the top relation candidates for anchor `i`.

This is not allowed to use labels. Labels are used only after mining for offline diagnostics.

### 5. Contrastive Relation Closure

- same-node augmentations remain positives；
- IRIS siblings become cross-node positives；
- high response uncertainty nodes become neutral/excluded；
- all others remain ordinary negatives。

The contribution is relation discovery, not a new contrastive objective.

## Required Controls

| System | Purpose |
|---|---|
| GRACE | base reference |
| kNN multi-positive | proximity mining |
| PPR/diffusion positives | graph proximity mining |
| PMGCL-lite/BMM | probabilistic positive mining |
| CAST one-step | strongest previous challenger/control |
| IRIS full | main candidate |
| IRIS response-shuffled | response semantic test |
| IRIS no anti-proximity | collapse-to-proximity test |
| IRIS structural-signature-only | role-equivalence test |
| IRIS no gradient-proxy | circularity test |
| R2 residualized response | revised main candidate |
| raw response no residual | tests whether residualization changes anything |
| residual + soft proximity penalty | tests soft replacement for hard anti-proximity |
| residual + CAST hybrid | tests whether response residuals help a stronger transport-like control |

## Diagnostics

- response sibling count per anchor。
- anti-proximity retained coverage。
- offline label agreement of response siblings。
- partial correlation between response similarity and label agreement after controlling feature similarity, embedding similarity, PPR proximity, and degree。
- false-negative repulsion mass。
- LogReg accuracy。
- runtime and candidate budget。

## Kill Rules

- response-shuffled matches IRIS：kill。
- structural-signature-only matches IRIS：kill or major pivot。
- no anti-proximity beats full IRIS：IRIS collapses to proximity mining。
- response similarity has no partial correlation with label agreement after controlling feature/embedding/PPR/degree：kill。
- CAST matches IRIS on label agreement, false-negative repulsion mass, and accuracy with lower cost：do not switch blindly; reconsider CAST or pivot。
- gains only come from more positives, more compute, or broader candidate budget：invalid。

## Current Verdict

`REVISE_NOT_PILOT`。  
Hard anti-proximity IRIS failed the first smoke. R2 residualization rescues the failure mode but still does not beat the strongest controls in Cora seed=0 smoke. No Pilot-A/B or formal experiment is currently supported.
