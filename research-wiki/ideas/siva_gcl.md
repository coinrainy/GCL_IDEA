---
type: idea
node_id: idea:siva_gcl
title: "SIVA-GCL-positive-core: Semantic Intervention View Augmentation"
stage: proposed
outcome: pending
added: 2026-06-26T11:23:43Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "siva", "view-generation", "smoke-planning"]
---

# SIVA-GCL-positive-core: Semantic Intervention View Augmentation

**stage:** `proposed`  ·  **outcome:** `pending`

Non-loss false-negative GCL direction: critic-constrained semantic-preserving intervention positives for node classification.

## Thesis
Graph contrastive learning can reduce false-negative dependence by generating informative node-local positive views: interventions should be maximally different from the original ego view while remaining stable under a masked-context semantic critic.

## Key risks
Fresh reviewer verdict REVISE, novelty 6.7/10. Risks: GraphMAE/critic-only may explain gains, random intervention may match SIVA, semantic critic may learn identity/degree shortcuts, and SPGCL prior requires positive prealignment diagnostics.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

