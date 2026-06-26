# WILLOW-GCL Literature Boundary

日期：2026-06-26  
方向：false-negative GCL / node classification / non-loss positive view generation

## Core Boundary

WILLOW 不能被表述为普通 world model 或 Graph-JEPA 变体。当前可投稿的边界应是：

```text
use latent ego target-prediction error as a semantic certificate
for node-local hard positive intervention search in GCL
```

## Closest Prior

| Prior | Existing idea | WILLOW boundary |
|---|---|---|
| Graph-JEPA / Predict, Cluster, Refine | context-target latent prediction as graph SSL objective | WILLOW uses target-prediction error as a certificate for GCL positive view selection |
| GraphMAE | masked feature reconstruction | WILLOW avoids raw feature decoder/reconstruction as semantic criterion |
| SPGCL 2026 | positive pre-alignment from message passing | WILLOW generates anti-prealignment certified positives |
| CGC / BalanceGCL / ACGA | generated hard negatives or positive-negative graph generation | WILLOW deletes virtual negatives from the mainline |
| RGCL / GCA / AutoGCL | rationale-aware or learned graph augmentation | WILLOW requires a latent target-prediction certificate, not only augmentation scoring |
| DoG | diffusion-generated synthetic graph structures | WILLOW only creates training views, not graph expansion |

## Implication

The paper story should stay on GCL positive signal and false-negative exposure. If WILLOW becomes only Graph-JEPA pretraining, it should be killed or rewritten.
