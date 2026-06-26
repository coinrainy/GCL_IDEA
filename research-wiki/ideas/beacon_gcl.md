---
type: idea
node_id: idea:beacon_gcl
title: "BEACON-GCL: Boundary-Eligibility Aware Contrastive Neutralization"
stage: smoke_negative
outcome: negative
added: 2026-06-26T15:07:06Z
based_on: []
target_gaps: ["gap:FN-G1", "gap:FN-G2", "gap:FN-G3", "gap:FN-G4", "gap:FN-G8"]
tags: ["gcl", "false-negative", "node-classification", "boundary-eligibility", "contrastive-neutralization", "post-cast-pivot"]
---

# BEACON-GCL: Boundary-Eligibility Aware Contrastive Neutralization

**stage:** `smoke_negative`  ·  **outcome:** `negative`

Counterfactual boundary-utility gate for false-negative edit eligibility after CAST Pilot-A no-go.

## Thesis
False-negative debiasing should be accepted only when a counterfactual pair edit reduces harmful repulsion without damaging boundary geometry, rank, or uniformity.

## Key risks
May be judged as validation-tuned mining if the boundary gate uses train/val probes too heavily; must show shuffled-gate and no-boundary controls.

## Origin

BEACON is a mechanism-level pivot after CAST Certificate Pilot-A no-go, not a continuation of CAST C4/C5. CAST showed that low-overlap certificate pairs can be genuinely different from kNN/PPR/CAST while still failing to improve node classification. BEACON therefore tests edit eligibility rather than pair novelty.

## Required Smoke Gate

- First run: `BE-M0-001`, Cora seed0 only.
- Required files before running: `configs/beacon_smoke.yaml`, `scripts/run_beacon_smoke.py`.
- Required controls: GRACE, kNN, CAST proxy, C4 certificate, all-candidate neutralization, BEACON full gate, shuffled gate, no-boundary gate, no-geometry gate.
- Decision before implementation: `GO_TO_BEACON_SMOKE_PLANNING`.

No formal result or performance claim is allowed from this page.

## BE-M0-001 Smoke Result

- Result: `refine-logs/EXPERIMENT_RESULTS_20260626_153556.md`
- Summary: `results/summary/be_m0_001_Cora_seed0_20260626T153505Z_summary.md`
- B5 BEACON full gate: `84.87`, label agreement `0.8611`
- kNN: `85.29`
- CAST proxy: `85.70`
- all-candidate neutralization: `85.42`
- shuffled utility gate: `85.01`
- no-boundary gate: `85.15`
- no-geometry gate: `85.24`

Decision: `KILL_CURRENT_BEACON_GATE_NO_BE_M1`.

The current gate improves pair diagnostics but not node classification accuracy. Do not run BE-M1 from this state.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
