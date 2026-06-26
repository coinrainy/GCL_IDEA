# Research Proposal: IRIS-GCL

## Problem Anchor

- **Bottom-line problem**：节点级 GCL 会把潜在同类节点当 negatives；仅靠相似度、近邻或 loss weighting 容易把 false-negative 发现退化为 positive mining。
- **Must-solve bottleneck**：需要一个不依赖 pair proximity 的 semantic relation criterion。
- **Core mechanism**：interventional response fingerprint + anti-proximity response-invariant sibling discovery + multi-positive / neutral closure。
- **Non-goals**：不改 InfoNCE denominator 作为主贡献；不使用 test labels；不使用 learned augmentation policy；不把 smoke 写成性能 claim。
- **Constraints**：首轮只做 Cora seed=0 smoke，`1:1:8` split，frozen encoder + Logistic Regression evaluator。

## Method Thesis

**One-sentence thesis**：IRIS-GCL discovers false negatives by identifying nodes with invariant response functions under a fixed intervention battery, while enforcing anti-proximity to avoid ordinary kNN/PPR/embedding positive mining.

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

### 4. Anti-proximity Response Siblings

Node `j` becomes an IRIS sibling for anchor `i` if:

```text
sim_response(R_i, R_j) >= tau_response
and sim_embedding(z_i, z_j) <= tau_embed_near
and graph_proximity(i, j) <= tau_graph_near
```

The anti-proximity constraints are mandatory. They force IRIS to search beyond ordinary feature/embedding/PPR neighbors.

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

`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`。  
IRIS is the current best idea / best bet for sufficient innovation, but it has no smoke or formal evidence. Next step is minimal Cora seed=0 smoke only.
