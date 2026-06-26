---
type: idea
node_id: idea:willow_gcl
title: "WILLOW-GCL: Latent Ego Target-Prediction Certified Views"
stage: proposed
outcome: pending
added: 2026-06-26T11:36:56Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "willow", "latent-certificate", "view-generation", "smoke-planning"]
---

# WILLOW-GCL: Latent Ego Target-Prediction Certified Views

**stage:** `proposed`  ·  **outcome:** `pending`

Active non-loss false-negative GCL candidate using latent target-prediction certificate for hard positive view search.

## Thesis
A latent ego target-prediction certificate can select node-local hard positive views that are farther than random augmentations but semantically stable, improving GCL positive signal while reducing dependence on real hard negatives.

## Key risks
Reviewer verdict REVISE, novelty 7.0/10. The certificate may learn degree/homophily/feature smoothness instead of node-class semantics; Graph-JEPA-only, SIVA-control, random matched, and shuffled-certificate controls are mandatory.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

