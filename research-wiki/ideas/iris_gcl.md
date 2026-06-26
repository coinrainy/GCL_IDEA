---
type: idea
node_id: idea:iris_gcl
title: "IRIS-GCL: Interventional Response Invariant Signatures"
stage: smoke_mixed
outcome: revise_toward_certified_cast
added: 2026-06-26T12:03:10Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "iris", "r2-iris", "cpr-iris", "response-invariance", "residualized-response", "response-certified-proximity", "smoke-mixed"]
---

# CPR-IRIS: Response-Certified Proximity

**stage:** `smoke_mixed`  ·  **outcome:** `revise_toward_certified_cast`

Revised high-risk non-loss false-negative GCL idea using response residuals as certificates over CAST/proximity candidate positives. Cora seed=0 smoke improved over hard anti-proximity IRIS but remains below the escalation bar.

## Thesis
False negatives can be discovered as cross-node response-invariant siblings under a fixed intervention battery, with anti-proximity constraints preventing collapse into kNN/PPR/embedding positive mining.

## Key risks
Fresh reviewer selected IRIS over CAST, novelty 7.2/10, confidence 0.61. Smoke result showed the main risk materialized: anti-proximity response invariance did not produce high-quality siblings.

## Smoke Result

- First failed run: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md`
- Revised R2 run: `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.md`
- CPR weight run: `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.md`
- Bridge report: `refine-logs/EXPERIMENT_RESULTS.md`
- Decision: `REVISE_TOWARD_CERTIFIED_CAST`
- I5 IRIS full: `76.48` test@best-val, label agreement `0.2418`
- I7 no anti-proximity: `84.59` test@best-val, label agreement `0.7787`
- I4 CAST proxy: `85.65` test@best-val, label agreement `0.7548`
- I10 R2 residualized response: `84.46` test@best-val, label agreement `0.7783`
- I13 residual response + CAST hybrid: `84.87` test@best-val, label agreement `0.7968`
- I17 CAST + residual certificate: `85.52` test@best-val, label agreement `0.7953`
- I20 kNN + residual certificate: `85.15` test@best-val, label agreement `0.8120`

Do not continue into Pilot-A/B or formal runs until the response component becomes clearly distinct from and competitive with kNN/PMGCL/CAST controls.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
