# Pipeline Summary: SIVA-GCL-positive-core

**Problem**：false-negative GCL needs a more innovative non-loss mechanism  
**Final Method Thesis**：semantic stability constrained intervention positives create informative node-local positives after message passing  
**Final Verdict**：REVISE_TO_SIVA_POSITIVE_SMOKE_PLANNING  
**Date**：2026-06-26

## Final Deliverables

- Idea report: `idea-stage/IDEA_REPORT.md`
- Candidate list: `idea-stage/IDEA_CANDIDATES.md`
- Novelty check: `idea-stage/SIVA_GCL_NOVELTY_CHECK.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- Dominant contribution：critic-constrained semantic-preserving intervention positive。
- Optional supporting contribution：positive informativeness diagnostics。
- Explicitly rejected complexity：virtual negatives as main claim, diffusion generator, denominator/loss tricks。

## First Runs to Launch

1. `SIVA-R001`: masked-context critic sanity。
2. `SIVA-R002`: intervention positive generation sanity。
3. `SIVA-R003/R004/R005`: GRACE/GCA/SIVA-positive smoke comparison。

## Next Action

Proceed to smoke implementation only. Do not run formal experiments or make performance claims.

