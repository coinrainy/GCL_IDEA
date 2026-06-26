# Research Proposal: CPR-IRIS / Response-Certified Proximity

## Problem Anchor

- **Bottom-line problem**：节点级 GCL 会把潜在同类节点当 negatives；仅靠相似度、近邻或 loss weighting 容易把 false-negative 发现退化为 positive mining。
- **Must-solve bottleneck**：需要让 proximity/CAST 候选获得语义证书，而不是退化成裸 kNN/PPR mining。
- **Core mechanism**：interventional response fingerprint + response-certified CAST/proximity candidate closure。
- **Non-goals**：不改 InfoNCE denominator 作为主贡献；不使用 test labels；不使用 learned augmentation policy；不把 smoke 写成性能 claim。
- **Constraints**：首轮只做 Cora seed=0 smoke，`1:1:8` split，frozen encoder + Logistic Regression evaluator。

## Method Thesis

**One-sentence thesis**：CPR-IRIS discovers false negatives by letting strong proximity/CAST candidate relations enter contrastive closure only when their interventional response residuals certify semantic consistency beyond ordinary feature, embedding, graph-proximity, and degree effects.

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

### 4. Response-Certified Candidate Closure

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

The latest smoke indicates that residual response alone is not enough. CPR therefore uses strong label-free candidate generators first, then applies residual response as a certificate:

```text
score_cpr(i, j)
  = score_candidate(i, j)
  + lambda * residual_response(i, j)
```

where `score_candidate` is CAST-style or kNN-style and `lambda` is fixed before a smoke run.

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
| response-certified CAST | current best CPR direction |
| response-certified kNN | collapse-to-kNN control |

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

`GO_TO_PILOT_PLANNING_WITH_CAUTION`。  
Hard anti-proximity IRIS failed, but the real CAST latent target-prediction certificate produced a positive Cora seed=0 smoke. Continue with narrow Pilot-A planning for CAST certificate only; do not revive IRIS/CPR incremental variants.
