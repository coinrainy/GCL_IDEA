# CAST-GCL Prior Boundary

日期：2026-06-26  
方向：false-negative GCL / cross-node positive closure / node classification

## Sources Checked

- Predict, Cluster, Refine proposes a joint embedding predictive graph SSL framework that avoids contrastive objectives, negative sampling, and reconstruction, using context-target latent prediction and multiple target subgraphs.
- Graph-JEPA predicts latent representations of masked subgraphs from context subgraphs.
- BalanceGCL builds balanced hard negative graphs and fine-grained semantic-aware positive graphs through class-aware counterfactual perturbations.
- PMGCL estimates true positive pair probability with a Beta Mixture Model.
- Recent generative+contrastive graph SSL literature explicitly notes that there is no principled way to generate positive node pairs.

## Boundary For CAST

CAST should not claim novelty for latent prediction alone, semantic positives alone, or pair mining alone. Its boundary is:

```text
cross-node semantic positive closure via low-energy intervention paths
certified by a latent ego target-prediction model
```

This is different from WILLOW because WILLOW only generates same-anchor hard positives. CAST uses the certificate to decide which real nodes are false-negative risks and should be moved out of the negative set.

## Required Controls

- WILLOW same-node certificate。
- kNN multi-positive。
- PMGCL-lite/BMM positive mining。
- certificate-shuffled CAST。
- budget-matched random transport。

## Remaining Risk

CAST may still be rejected if reviewers consider the transport search an expensive positive-mining heuristic. It needs compact implementation, budget-matched controls, and label-agreement diagnostics before any claim.
