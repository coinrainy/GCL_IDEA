# ORBIT-GCL Novelty and Reviewer Check

- 时间：2026-06-26T17:16:14Z
- Reviewer：fresh `gcl_scientific_reviewer`
- Verdict：`REVISE`
- Confidence：`0.73`
- Claim status：proposal-only novelty review；未实现、未跑 ORBIT smoke/pilot/formal。

## Reviewer Ranking

| Rank | Candidate | Verdict | Novelty risk |
|---:|---|---|---|
| 1 | ORBIT-GCL | `REVISE` | TTER / D-SLA / Graph-JEPA / GraphMAE2 / GraphCL / JOAO / GCA / GCIL / SpCo |
| 2 | MORPH-GCL | `PIVOT` | Graph Edit Networks / GEDGNN / topology transformation SSL |
| 3 | SCALE-GCL | `KILL/weak PIVOT` | InfoGraph / MVGRL / DGI / hierarchical contrastive / SPECTRA |
| 4 | ATLAS-GCL | `KILL` | AutoSSL / multi-pretext graph pretraining / pseudo-label function stacking |

## Reviewer Judgment

ORBIT 是四个候选中最值得保留的方向，因为它把学习对象从 node-node relation 转成 node-operator response target，确实比继续调 false-negative mining 更像机制级 pivot。

但当前不能给 `GO`。Reviewer 认为 ORBIT 仍有拼装风险：GraphCL augmentation + Graph-JEPA latent prediction + TTER/D-SLA transformation/discrepancy prediction + Barlow/VICReg guard。技术 delta 必须收缩为：

> node-level operator response field 的低秩 response-basis prediction。

## Fatal Weaknesses

1. Response basis 的因果必要性尚未证明。现在只能说它“可能包含 class-readable evidence”，但还没有证明它比 feature、degree、diffusion/spectrum 或 teacher regularization 更接近下游类别。
2. 组件过多会削弱论文主线。若加入 EMA teacher、SVD/random projection、covariance prediction、same-node consistency、operator-code contrast、Barlow guard，任何提升都可能被归因于 regularization/capacity/augmentation。
3. 与最近邻碰撞严重。TTER 已预测 topology transformation，D-SLA 已学习 original/perturbed discrepancy，Graph-JEPA/GraphMAE2 已做 latent target prediction。ORBIT 必须证明自己学的是跨 operator 的节点响应函数。

## Required Kill-Smoke Controls

Cora seed0 only，先不跑 pilot。必须包含：

- GRACE
- matched-parameter GRACE
- GCMAE / GraphMAE2-lite
- Graph-JEPA latent target
- TTER-style transformation-id prediction
- D-SLA-style discrepancy regression
- IRIS/WILLOW same-node response control
- operator-id-only
- response-magnitude-only
- degree-only / feature-only response
- random operator labels
- shuffled response basis
- random teacher
- no-SVD raw response
- no-consistency / no-regularization controls

## Hard Kill Rules

- ORBIT full 若不超过最强 matched control，kill。
- random / shuffled / degree-only / feature-only response 若接近 ORBIT full，kill。
- held-out operator response prediction 失败，kill。
- gain 在 matched budget/capacity 下消失，kill。
- 只改善 response prediction / rank / uniformity，而不改善 frozen LogReg accuracy，kill。

## Decision

`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`

下一步只允许写最小 ORBIT spec 和 Cora seed0 kill-smoke plan，然后再考虑实现。不得直接进入 Pilot-A/B 或 formal。
