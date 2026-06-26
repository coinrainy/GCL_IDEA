# DSR-GCL: Decoupled Semantic-Residual Graph Contrastive Learning

status: active_candidate  
decision: REVISE_IDEA  
created_utc: 2026-06-26T09:21:55Z  
updated_utc: 2026-06-26T09:24:55Z  
source: `/idea-discovery` beyond IGT-GCL  
task: node classification  
method family: graph contrastive learning

## Summary

DSR-GCL targets false-negative damage in node-level graph contrastive learning by decoupling the representation into a semantic channel and a residual channel. The semantic channel uses negative-free consistency and collapse guards; the residual channel carries InfoNCE/rank-style repulsion and uniformity. The core mechanism, **Spectral Gradient Firewall**, prevents negative-sample repulsive gradients from updating the semantic channel.

## Why Beyond IGT

IGT-GCL uses lower/upper interval pair targets and was reviewed at novelty `5.5/10`, with risk of being a point posterior / reweighting / abstention variant. DSR-GCL avoids pair posterior, interval, abstention, filtering, and downweighting. It changes where negative gradients are allowed to act.

## Refined Mechanism

Default revised version uses parameter-isolated channels:

```text
h_sem = GNN_sem(X_sem, A)
h_res = GNN_res(X_res, A)
z_sem = P_sem(h_sem)
z_res = P_res(h_res)
z_eval = concat(z_sem, stopgrad(z_res))
```

- `L_sem`: negative-free alignment plus variance/covariance guard。
- `L_res`: InfoNCE or rank-style residual contrast。
- `L_orth`: branch redundancy control。
- negative loss updates only residual parameters, not semantic parameters。

## Reviewer Verdict

Fresh `gcl_scientific_reviewer` verdict: **REVISE_IDEA**.  
Novelty score: **6.2/10**.  
Confidence: **0.72**.

The reviewer judged DSR-GCL slightly stronger than IGT-GCL but not ready for `GO_TO_EXPERIMENT_BRIDGE`. The key required revision is making the gradient firewall concrete, measurable, and ablatable. This page reflects the revised parameter-isolated version.

## Closest Prior Risk

- BGRL / CCA-SSG / Graph Barlow Twins: negative-free graph SSL。
- ReGCL: InfoNCE gradient conflict and gradient-weighted objective。
- SPGCL 2026: Dirichlet energy and positive pre-alignment。
- ASPECT 2026: low/high frequency node-wise spectral fusion。
- GraphRank: margin/rank objective reducing excessive negative separation。
- LR-GCL / spectral low-rank GCL: low-frequency node-classification representation。

## Required Ablations

- GRACE / GCA。
- BGRL / CCA / Barlow-style negative-free baseline。
- same-parameter single-head。
- pure semantic-only。
- residual-only InfoNCE。
- semantic branch receives negatives。
- no residual branch。
- no orthogonality。
- random branch split。
- GraphRank residual loss。
- SPGCL/ASPECT-style spectral fusion control。
- `z_sem`, `z_res`, fixed concat evaluator。
- negative-gradient leakage diagnostic。

## Evidence Status

No smoke, pilot, development, or formal experiment has been run for DSR-GCL. Current evidence is literature review, idea generation, novelty review, and experiment planning only.

## Decision

**REVISE_IDEA**.  
Do not proceed to experiment bridge until the revised firewall definition is accepted and a smoke/pilot plan is explicitly approved.

## Files

- `idea-stage/FN_IDEAS_BEYOND_IGT.md`
- `idea-stage/IDEA_CANDIDATES.md`
- `idea-stage/DSR_GCL_NOVELTY_BRIEF.md`
- `idea-stage/IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `.aris/traces/novelty-check/2026-06-26_run03/`
