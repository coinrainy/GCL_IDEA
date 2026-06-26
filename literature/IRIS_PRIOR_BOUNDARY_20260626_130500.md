# IRIS-GCL Prior Boundary

日期：2026-06-26  
方向：interventional response signatures / false-negative GCL / node classification

## Checked Boundary

Recent related areas:

- Graph Contrastive Invariant Learning from a causal perspective uses spectral graph augmentation as intervention and encourages invariant representations.
- CIA-GCL learns invariant subgraphs and invariance-aware augmentation for brain graph contrastive learning under distribution shifts.
- PMGCL and related positive mining methods estimate true positive pairs through mixture/probability or similarity signals.
- CAST-GCL uses cross-node semantic transport paths.

## IRIS Boundary

IRIS should claim novelty only for:

```text
cross-node false-negative discovery via interventional response-function equivalence
under anti-proximity constraints
```

It should not claim novelty for causal augmentation, invariant learning, positive mining, or multi-positive contrastive loss alone.

## Main Risk

Reviewers may argue IRIS is a complicated response-signature positive miner. To survive, it must show response similarity predicts same-label pairs after controlling feature similarity, embedding similarity, and graph proximity.
