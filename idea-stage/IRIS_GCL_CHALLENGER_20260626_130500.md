# IRIS-GCL Challenger Brief

生成时间：2026-06-26T13:05:00Z  
方向：图对比学习假负样本，普通节点分类，非 loss-only 机制  
状态：challenger candidate；未实现代码、未跑实验、未产生性能 claim

## Motivation

CAST-GCL 是当前最好候选，但 fresh reviewer 指出它仍可能被归约为：

```text
Graph-JEPA-like certificate + candidate mining + multi-positive GCL
```

核心风险是 transport energy 可能只是 kNN/PPR/BMM positive mining 的复杂重写。为了继续寻找“足够创新”的 best idea，本候选尝试完全避开 proximity / transport / pair posterior。

## Top Challenger: IRIS-GCL

**名称**：IRIS-GCL / Interventional Response Invariant Signatures for Graph Contrastive Learning  
**目标**：用节点的 label-free interventional response function 识别 false negatives，而不是用 pair similarity、transport energy 或 loss reweighting。

## Core Thesis

Two nodes should not be contrasted as negatives if they exhibit the same response function under a standardized battery of graph interventions, even when they are far apart in feature, graph, or embedding space.

In node classification, semantic equivalence is not only about proximity. It can appear as **functional equivalence under perturbation**:

```text
same class-semantics => similar response to causal/environmental interventions
```

IRIS constructs false-negative candidates by comparing response signatures, not representations.

## Mechanism

### 1. Intervention Battery

Define a fixed set of label-free graph interventions:

- spectral high-frequency perturbation；
- low-pass / diffusion-scale perturbation；
- ego edge drop / keep；
- feature group mask；
- neighbor role swap；
- degree-bin preserving ego rewiring。

These interventions simulate environment shifts. They are not learned augmentations and are fixed before training.

### 2. Response Fingerprint

Warm up a base encoder `E` with GRACE or another fixed SSL baseline, then freeze it for fingerprint construction.

For node `i` under intervention `omega`, compute:

```text
r_i(omega)
  = [
      delta z_i(omega),
      delta local smoothness_i(omega),
      delta ego prediction consistency_i(omega),
      delta positive-gradient proxy_i(omega)
    ]
```

The response fingerprint is:

```text
R_i = concat_{omega in Omega} whiten(r_i(omega))
```

Important: relation construction uses response fingerprints after removing raw embedding similarity components. This is meant to avoid collapsing into kNN.

### 3. Response-Invariant False-Negative Discovery

For anchor `i`, node `j` is a response-invariant sibling if:

```text
sim_response(R_i, R_j) >= tau_response
and sim_embedding(z_i, z_j) <= tau_near
and graph_proximity(i, j) <= tau_graph_near
```

The anti-proximity constraints intentionally search for nontrivial false negatives: nodes that are not simply close neighbors but behave similarly under interventions.

### 4. Contrastive Relation Update

- same-node augmentations remain positives；
- response-invariant siblings become multi-positive；
- high response uncertainty becomes neutral / excluded；
- all others remain ordinary negatives。

The contribution is a relation-discovery mechanism, not a new loss.

## Why IRIS Might Beat CAST

| Aspect | CAST | IRIS |
|---|---|---|
| Relation evidence | low-energy intervention path between nodes | invariant response function under many interventions |
| Main collision risk | kNN/PPR/BMM positive mining | causal invariant GCL / response signature heuristic |
| How it avoids proximity | tries to beat candidate-pool controls | explicitly requires non-near embedding/graph pairs |
| False-negative story | nodes reachable by semantic transport should not be negatives | nodes with same intervention response should not be negatives |
| Reviewer hook | semantic transport closure | functional equivalence / response invariance |

IRIS may be more novel than CAST if response signatures predict same-label pairs beyond proximity baselines.

## Prior Boundary

- vs GCIL / causal invariant GCL：existing work encourages each node representation to be invariant across causal interventions. IRIS compares **cross-node response functions** to discover false negatives.
- vs CIA-GCL：CIA-GCL is brain graph / graph-level invariant subgraph augmentation; IRIS targets ordinary node classification and false-negative discovery.
- vs PMGCL / positive mining：IRIS does not estimate true-positive probability from pair similarity; it uses response congruence under intervention battery with anti-proximity constraints.
- vs CAST：IRIS does not search transport paths or rely on candidate-pool proximity.
- vs Graph-JEPA/PCR：IRIS does not need latent target prediction as the main certificate.

## Minimal Smoke

- Dataset：Cora seed=0 only。
- Protocol：project `1:1:8` split，frozen encoder + Logistic Regression。
- Systems：
  - GRACE；
  - kNN multi-positive；
  - PPR/diffusion positives；
  - PMGCL-lite/BMM；
  - CAST revised one-step；
  - IRIS response-invariant siblings；
  - IRIS response-shuffled；
  - IRIS no anti-proximity；
  - IRIS embedding-response unwhitened。
- Diagnostics：
  - response sibling count per anchor；
  - offline label agreement of response siblings；
  - response similarity vs label agreement after controlling embedding/PPR proximity；
  - false-negative repulsion mass；
  - LogReg accuracy；
  - runtime。

## Kill Rules

- kNN/PPR/BMM match IRIS label agreement：kill IRIS。
- response-shuffled matches IRIS：response signature not semantic。
- removing anti-proximity improves everything：IRIS collapses to proximity mining。
- response similarity has no partial correlation with label agreement after controlling feature/embedding/PPR similarity：kill。
- CAST matches or beats IRIS on all diagnostics with lower cost：keep CAST as best idea。

## Decision

`CHALLENGER_FOR_CAST_REVIEW`

IRIS is not yet the active idea. It is a challenger to determine whether CAST is truly the best available direction.
