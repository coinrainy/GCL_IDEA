---
type: experiment
node_id: exp:beacon_be_m0_001
title: "BEACON-GCL BE-M0-001 Cora seed0 smoke"
idea_id: "idea:beacon_gcl"
verdict: no
confidence: medium
date: "2026-06-26"
hardware: "local RTX 3060"
duration: "~26 seconds wall-clock after implementation dryrun; Cora seed0 smoke"
provenance: "results/summary/be_m0_001_Cora_seed0_20260626T153505Z_summary.md; logs/beacon_smoke/be_m0_001_Cora_seed0_20260626T153505Z/"
added: 2026-06-26T15:36:27Z
tags: ["beacon-gcl", "smoke", "cora", "negative", "no-go"]
---

# BEACON-GCL BE-M0-001 Cora seed0 smoke

**verdict:** `no`  ·  **confidence:** `medium`  ·  tests `idea:beacon_gcl`

## Metrics
B5 BEACON full gate test@best-val 84.87; B1 kNN 85.29; B2 CAST proxy 85.70; B4 all-candidate 85.42; B6 shuffled 85.01; B7 no-boundary 85.15; B8 no-geometry 85.24. B5 label agreement 0.8611, accepted/anchor 8.49, margin delta 0.3063, rank delta -9.1365.

## Reasoning
BEACON full gate did not beat kNN, CAST proxy, all-candidate neutralization, or shuffled/no-boundary/no-geometry controls. The gate protected geometry but did not convert that protection into classification accuracy.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

