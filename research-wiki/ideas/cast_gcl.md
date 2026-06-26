---
type: idea
node_id: idea:cast_gcl
title: "CAST-GCL: Certificate-guided Semantic Transport"
stage: pilot_a_negative
outcome: revise_or_pivot
added: 2026-06-26T12:03:10Z
based_on: []
target_gaps: ["gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "cast", "semantic-transport", "latent-target-certificate", "smoke-positive", "pilot-negative", "revise-or-pivot"]
---

# CAST-GCL: Certificate-guided Semantic Transport

**stage:** `pilot_a_negative`  ·  **outcome:** `revise_or_pivot`

Previously demoted to mandatory IRIS control after fresh comparison review, then reactivated after a positive Cora seed=0 smoke. Pilot-A now shows that the current certificate design is not ready for more seeds or formal evaluation.

## Thesis
CAST treats false negatives as low-energy semantic transport positives. The current implementation has a real low-overlap certificate signal, but that signal does not yet improve node classification reliably.

## Key risks
Fresh comparison reviewer scored CAST novelty 6.5/10 and judged prior collision with kNN/PPR/BMM positive mining too high.

## Post-IRIS Role

CAST proxy remains a strong accuracy control in the IRIS/CPR smoke suite. The certificate variant is not promoted to a paper method; future work must explain why low-overlap certificate positives fail to improve classification before any more Pilot-A expansion.

## CAST Certificate Smoke

- Result: `refine-logs/CAST_CERTIFICATE_RESULTS.md`
- Clean summary: `results/summary/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z_summary.md`
- C4 latent target certificate: `85.93`, label agreement `0.7911`
- C5 certificate + CAST score: `86.16`, label agreement `0.7886`
- CAST proxy: `85.70`, label agreement `0.7548`

Decision: `GO_TO_PILOT_PLANNING_WITH_CAUTION`.

## Pilot-A Diagnostic Results

- Result: `refine-logs/EXPERIMENT_RESULTS_20260626_144910.md`
- Scope: Cora seeds 0-2, CiteSeer seeds 0-2, PubMed seed 0.
- Cora: C5 `82.73±2.94`, below kNN `84.23±1.16` and CAST proxy `83.79±2.14`.
- CiteSeer: C5 `71.77±0.09`, below CAST proxy `71.99±0.36` and close to kNN `71.85±1.10`.
- PubMed seed0: C5 `85.47`, below kNN `85.67` and CAST proxy `85.62`.
- Diagnostic nuance: C4 low-overlap certificate pairs are not simply kNN/PPR/CAST pairs, but this did not convert into stable accuracy improvement.

Decision: `REVISE_OR_PIVOT_BEFORE_ANY_MORE_PILOT`.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
