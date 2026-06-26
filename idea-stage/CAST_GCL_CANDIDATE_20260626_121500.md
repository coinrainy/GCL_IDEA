# CAST-GCL Candidate Brief

生成时间：2026-06-26T12:15:00Z  
子流程：`/idea-discovery` corrective refinement beyond WILLOW  
方向：图对比学习假负样本，普通节点分类，非 loss-only 机制  
状态：candidate refinement；未实现代码、未跑新实验、未产生性能 claim

## 为什么 WILLOW 仍不够

WILLOW 已经避开了 BOND 的 loss-side 风险，也比 SIVA 的 reconstruction critic 更有新意。但它仍有一个核心 reviewer 风险：

```text
WILLOW = Graph-JEPA-like latent prediction + learned augmentation scorer
```

也就是说，WILLOW 主要生成同一 anchor 的 harder positive view，false negative 仍只是被间接缓解。2025/2026 边界上已经有：

- Graph-JEPA / Predict, Cluster, Refine：latent context-target prediction；
- PMGCL：估计 true positive pair probability；
- BalanceGCL：fine-grained semantic-aware positives 与 balanced hard negatives；
- GCA/RGCL/AutoGCL：learned or rationale-aware augmentation。

因此更强的机制应直接处理 **跨节点 false negatives**，而不是只改善同一节点的 view augmentation。

## Top Candidate: CAST-GCL

**名称**：CAST-GCL / Certificate-guided Semantic Transport for Graph Contrastive Learning  
**推荐级别**：Top 1 corrective refinement  
**贡献类型**：cross-node semantic relation construction / certified positive closure；不是 loss trick

## Core Thesis

False negatives in node-level GCL should be treated as missing semantic transport relations, not merely as over-weighted denominator terms.

CAST-GCL learns a latent ego target-prediction certificate, then uses it to test whether two real nodes can be connected by a low-energy sequence of node-local interventions. If anchor `i` can be transported to candidate `j` while keeping certificate error low, then `j` is a semantic sibling: it should become a multi-positive or neutral relation rather than a negative.

## Mechanism

### 1. Latent Certificate Backbone

Reuse WILLOW's latent ego target-prediction certificate:

```text
h_c      = E_online(context_ego_i)
h_t^k    = E_target(masked_target_ego_i^k)
pred_t^k = P(h_c, pos_k)
cert_error = sum_k || pred_t^k - stopgrad(h_t^k) ||_2^2
```

This module is not the final SSL objective. It is only a semantic certificate.

### 2. Cross-node Semantic Transport

For anchor `i` and candidate node `j`, search a short path of local interventions:

```text
i = u_0 -> u_1 -> ... -> u_L = j
```

Each step edits ego context by feature mask/replacement, edge keep/drop, or neighbor-context substitution. The path has energy:

```text
E_transport(i, j)
  = min_path sum_l [
      cert_error(u_l -> u_{l+1})
      + lambda_edit * edit_cost(u_l, u_{l+1})
      + lambda_anchor * anchor_drift(u_l, i)
    ]
```

Candidate `j` enters the semantic positive closure of `i` if:

```text
E_transport(i, j) <= tau_transport
```

Thresholds and candidate budgets must be chosen by train/val diagnostics only.

### 3. Contrastive Training With Certified Positive Closure

CAST changes the relation graph used by GCL:

- same-node augmented views remain positives；
- low-energy transported real nodes become additional positives；
- uncertain reachable nodes become neutral / excluded diagnostics；
- all other nodes remain standard negatives。

The contribution is not a new denominator formula. The contribution is a certified cross-node semantic relation constructor.

## Why CAST Is Stronger Than WILLOW

| Aspect | WILLOW | CAST |
|---|---|---|
| Main relation | same-anchor hard positive view | cross-node semantic positive closure |
| False-negative handling | indirect, by strengthening positives | direct, by detecting real nodes that should not be negatives |
| Main prior risk | Graph-JEPA + augmentation | PMGCL / positive mining, but with transport certificate rather than pair posterior |
| Key evidence | harder views with stable certificate | transported positives have high offline label agreement and improve false-negative repulsion mass |

## Prior Boundary

- vs PMGCL：CAST does not estimate pair probability from similarity/BMM. It requires a low-energy intervention path certified by latent target prediction.
- vs BalanceGCL：CAST is node-level, transductive node-classification oriented, and does not use class-aware counterfactual gradients or generated hard negatives.
- vs Graph-JEPA/PCR：CAST uses latent prediction only as a certificate to build cross-node relation closure.
- vs WILLOW：CAST subsumes WILLOW as the same-node certificate module but adds cross-node semantic transport.

## Minimal Smoke

- Dataset：Cora seed=0 only。
- Protocol：project `1:1:8` split，frozen encoder + Logistic Regression。
- Systems：
  - GRACE；
  - WILLOW same-node certified positive；
  - kNN multi-positive control；
  - PMGCL-lite/BMM positive mining control；
  - CAST certificate-shuffled transport；
  - CAST one-step transport；
  - CAST two-step transport。
- Diagnostics：
  - transported positive count per anchor；
  - offline label agreement of transported positives；
  - transport energy vs label agreement correlation；
  - false-negative repulsion mass before/after closure；
  - LogReg accuracy；
  - search time and candidate budget。

## Kill Rules

- kNN multi-positive matches CAST：transport certificate unnecessary。
- PMGCL-lite matches CAST：CAST reduces to positive mining。
- certificate-shuffled transport matches CAST：certificate has no semantic content。
- transport energy does not correlate with offline label agreement：kill CAST。
- CAST only works by selecting many more positives than controls：invalid unless budget-matched variant also works。
- WILLOW same-node matches CAST：cross-node closure unnecessary; downgrade to WILLOW。

## Decision

`REVISE_TO_CAST_PRE_REVIEW`

CAST is a stronger corrective candidate than WILLOW, but it has not yet received a fresh reviewer verdict and has no experiment evidence. Next step is to replace the active report only if we accept the additional complexity, then run fresh reviewer or minimal smoke.
