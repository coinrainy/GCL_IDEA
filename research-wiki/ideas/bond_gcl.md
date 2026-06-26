---
type: idea
node_id: idea:bond_gcl
title: "BOND-GCL: Basin-capped Objective for Negative Debiasing"
stage: proposed
outcome: pending
added: 2026-06-26T10:57:08Z
based_on: []
target_gaps: ["gap:FN-G1", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "bond", "smoke-planning"]
---

# BOND-GCL: Basin-capped Objective for Negative Debiasing

**stage:** `proposed`  ·  **outcome:** `pending`

Basin-level negative mass aggregation for false-negative graph contrastive learning in node classification.

## Thesis
InfoNCE false-negative damage in node-level GCL can be reduced by capping repeated repulsive mass from anchor-local semantic basins rather than estimating each pair's true-negative probability.

## Key risks
Fresh reviewer verdict REVISE, novelty 6.0/10. Main risks: equivalence to pair reweighting, E2Neg-style small-negative sampling, and basin constructor carrying the contribution. Requires shuffled-basin, same-pair-weight, and E2Neg-style controls before pilot.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

