# Pipeline Summary: BOND-GCL

**Problem**：图对比学习节点分类中的 false-negative repeated repulsion  
**Final Method Thesis**：Basin-level negative mass aggregation limits repeated repulsion from anchor-local semantic basins while preserving denominator diversity.  
**Final Verdict**：REVISE_TO_SMOKE_PLANNING  
**Date**：2026-06-26

## Final Deliverables

- Idea report: `idea-stage/IDEA_REPORT.md`
- Candidate list: `idea-stage/IDEA_CANDIDATES.md`
- Novelty check: `idea-stage/BOND_GCL_NOVELTY_CHECK.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- Dominant contribution：basin-level negative mass aggregation operator。
- Optional supporting contribution：false-negative mass exposure diagnostic。
- Explicitly rejected complexity：learned basin scorer, KAN/LLM/generator, coupled overflow main mechanism。

## Must-Prove Claims

- BOND reduces repeated false-negative basin exposure。
- BOND is not ordinary pair reweighting。
- BOND is not E2Neg-style small negative sampling。
- BOND has node-classification utility under frozen encoder + Logistic Regression。

## First Runs to Launch

1. `BOND-R001`: GRACE wrapper parity on Cora seed 0。
2. `BOND-R002`: BOND-cap-only Cora seed 0。
3. `BOND-R003/R004/R005`: pair-weight, random-basin, and E2Neg-style controls。

## Main Risks

- Reweighting equivalence：mitigate with same pair weights / shuffled basin control。
- Small-denominator explanation：mitigate with E2Neg-style center and denominator effective number。
- Basin constructor overfitting：mitigate with random basin same histogram。

## Next Action

Proceed to smoke implementation only. Do not run formal experiments or make performance claims.

