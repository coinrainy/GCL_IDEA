---
type: idea
node_id: idea:iris_gcl
title: "IRIS-GCL: Interventional Response Invariant Signatures"
stage: smoke_failed
outcome: pivot_required
added: 2026-06-26T12:03:10Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "iris", "response-invariance", "anti-proximity", "smoke-failed", "pivot-required"]
---

# IRIS-GCL: Interventional Response Invariant Signatures

**stage:** `smoke_failed`  ·  **outcome:** `pivot_required`

Former high-risk non-loss false-negative GCL idea using anti-proximity cross-node response invariance. Cora seed=0 smoke failed and triggered `PIVOT_REQUIRED`.

## Thesis
False negatives can be discovered as cross-node response-invariant siblings under a fixed intervention battery, with anti-proximity constraints preventing collapse into kNN/PPR/embedding positive mining.

## Key risks
Fresh reviewer selected IRIS over CAST, novelty 7.2/10, confidence 0.61. Smoke result showed the main risk materialized: anti-proximity response invariance did not produce high-quality siblings.

## Smoke Result

- Run: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md`
- Bridge report: `refine-logs/EXPERIMENT_RESULTS.md`
- Decision: `PIVOT_REQUIRED`
- I5 IRIS full: `76.48` test@best-val, label agreement `0.2418`
- I7 no anti-proximity: `84.59` test@best-val, label agreement `0.7787`
- I4 CAST proxy: `85.65` test@best-val, label agreement `0.7548`

Do not continue current IRIS anti-proximity mechanism into Pilot-A/B or formal runs unless the mechanism is explicitly redesigned and re-planned.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
