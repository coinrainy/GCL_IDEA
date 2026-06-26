# IGT-GCL: Interval-Guarded Pair Targeting for Graph Contrastive Learning

status: active_candidate  
decision: REVISE  
created_utc: 2026-06-26T09:08:32Z  
source: fresh `/idea-discovery` restart after user abandoned BCE/CER-GCL  
task: node classification  
method family: graph contrastive learning  

## Summary

IGT-GCL targets the false-negative / hard-negative ambiguity in node-level graph contrastive learning. Instead of assigning each non-augmented pair a single pseudo-positive probability, it constructs a lower/upper interval `[l_ij, u_ij]`:

- high `l_ij`: conservative soft positive；
- low `u_ij` plus high embedding similarity: reliable hard true negative；
- low `l_ij` but high `u_ij`: ambiguous pair, abstain from strong attraction or repulsion。

The method is intended as a GRACE/GCA loss-side plugin and must keep the project evaluator protocol: frozen encoder + Logistic Regression, homophilous `1:1:8`, heterophilous official fixed split.

## Problem Anchor

In node-level GCL, a high-similarity non-augmented node may be a same-class candidate positive or a true hard negative. Existing point-posterior approaches can over-attract uncertain pairs or repel false negatives.

## Core Hypothesis

Set-valued pair targets can reduce overconfident pseudo-labeling. The useful novelty is not "PU-GCL" or "soft target InfoNCE"; it is lower/upper-bound target construction with three actions: attraction, reliable hard-negative repulsion, and abstention.

## Closest Prior Work Boundary

- ProGCL: true-negative probability + hardness。
- KDD 2024 Node Similarity: ideal positives/no-false-negatives and node similarity distribution。
- IFL-GCL: PU learning view and InfoNCE similarity posterior。
- Affinity Uncertainty HNM: anchor-dependent uncertainty hard-negative weights。
- NML-GCL: learnable negative metric network and soft label。

IGT-GCL must prove that interval targets beat point posterior and simple reweighting; otherwise the idea should pivot.

## Required Ablations

- vanilla GRACE/GCA；
- ProGCL-style true-negative weighting；
- NodeSim/IFL-style point posterior；
- midpoint target `(l+u)/2`；
- only positive expansion；
- only negative downweight；
- no abstention；
- random interval matched sparsity；
- full IGT-GCL。

## Evidence Status

No smoke, pilot, development, or formal experiment has been run for IGT-GCL. Current evidence consists of literature review, idea generation, novelty review, and method/experiment planning only.

## Reviewer Verdict

Fresh `gcl_scientific_reviewer` verdict: **REVISE**, novelty score **5.5/10**, confidence **0.78**.

Allowed next step: smoke/pilot implementation planning.  
Forbidden next step: direct formal claim or SOTA/robust/comprehensive claim.

## Files

- `idea-stage/FRESH_GCL_IDEAS_20260626_090132.md`
- `idea-stage/IPU_GCL_NOVELTY_BRIEF_20260626_090347.md`
- `idea-stage/IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `.aris/traces/novelty-check/2026-06-26_run02/`
