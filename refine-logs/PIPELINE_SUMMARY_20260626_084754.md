# Pipeline Summary

**Problem**: Node-wise eligibility for strong alignment in graph contrastive learning.  
**Final Method Thesis**: CER-GCL switches anchor objective type between full InfoNCE and positive-only stop-gradient consistency using a frozen unsupervised eligibility gate.  
**Final Verdict**: READY_FOR_PILOT_PLANNING  
**Date**: 2026-06-26

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- Dominant contribution: anchor-level objective type switching for contrastive eligibility.
- Optional supporting contribution: falsification protocol separating routing from scalar weighting and known proxies.
- Explicitly rejected complexity: boundary-aware GNN, adaptive augmentation generator, pair false-negative mining, LLM/KAN/prototype modules.

## Must-Prove Claims

- Eligibility signal has residual predictive value beyond degree/uncertainty.
- Objective routing beats same-score scalar weighting on targeted diagnostics.

## First Runs to Launch

1. R001: vanilla GRACE smoke on Cora seed0.
2. R002: CER-GCL default smoke on Cora seed0.
3. R003: q_i gate diagnostic on Cora/CiteSeer seed0.

## Main Risks

- Routing equals weighting: kill routing claim.
- Eligibility equals degree/uncertainty: pivot or kill.
- Positive-only branch collapses: stop before pilot interpretation.

## Next Action

Proceed to implementation only for smoke/pilot, not formal experiments.

