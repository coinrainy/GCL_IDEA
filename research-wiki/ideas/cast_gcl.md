---
type: idea
node_id: idea:cast_gcl
title: "CAST-GCL: Certificate-guided Semantic Transport"
stage: smoke_positive
outcome: pilot_planning_with_caution
added: 2026-06-26T12:03:10Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "cast", "semantic-transport", "latent-target-certificate", "smoke-positive", "pilot-planning"]
---

# CAST-GCL: Certificate-guided Semantic Transport

**stage:** `smoke_positive`  ·  **outcome:** `pilot_planning_with_caution`

Previously demoted to mandatory IRIS control after fresh comparison review; now reactivated because the real latent target-prediction certificate produced a positive Cora seed=0 smoke.

## Thesis
CAST treats false negatives as low-energy semantic transport positives, but is now retained as a mandatory control for IRIS.

## Key risks
Fresh comparison reviewer scored CAST novelty 6.5/10 and judged prior collision with kNN/PPR/BMM positive mining too high.

## Post-IRIS Role

CAST proxy remains the strongest accuracy control in the IRIS/CPR smoke suite. It is not automatically promoted to a paper method. A pivot is allowed only if the real latent target-prediction transport certificate is implemented and shown not to reduce to feature/embedding/PPR positive mining.

## CAST Certificate Smoke

- Result: `refine-logs/CAST_CERTIFICATE_RESULTS.md`
- Clean summary: `results/summary/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z_summary.md`
- C4 latent target certificate: `85.93`, label agreement `0.7911`
- C5 certificate + CAST score: `86.16`, label agreement `0.7886`
- CAST proxy: `85.70`, label agreement `0.7548`

Decision: `GO_TO_PILOT_PLANNING_WITH_CAUTION`.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
