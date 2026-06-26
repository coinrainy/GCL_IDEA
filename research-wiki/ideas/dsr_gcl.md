# DSR-GCL: Decoupled Semantic-Residual Graph Contrastive Learning

status: active_candidate  
decision: REVISE_IDEA  
created_utc: 2026-06-26T09:21:55Z  
updated_utc: 2026-06-26T10:11:03Z  
source: `/idea-discovery` beyond IGT-GCL  
task: node classification  
method family: graph contrastive learning

## Summary

DSR-GCL targets false-negative damage in node-level graph contrastive learning by decoupling the representation into a semantic channel and a residual channel. The semantic channel uses negative-free consistency and collapse guards; the residual channel carries InfoNCE/rank-style repulsion and uniformity. The core mechanism, **Spectral Gradient Firewall**, prevents negative-sample repulsive gradients from updating the semantic channel.

## Why Beyond IGT

IGT-GCL uses lower/upper interval pair targets and was reviewed at novelty `5.5/10`, with risk of being a point posterior / reweighting / abstention variant. DSR-GCL avoids pair posterior, interval, abstention, filtering, and downweighting. It changes where negative gradients are allowed to act.

## Refined Mechanism

Default revised version uses **parameter-isolated two-channel firewall**. Shared trunk is explicitly excluded from the default because residual InfoNCE would otherwise update shared encoder parameters and indirectly contaminate the semantic branch.

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

Mathematical firewall condition:

```text
|| grad(theta_sem; L_res) ||_2 = 0
firewall_leak_param =
  || grad(theta_sem; L_res) ||_2 / (|| grad(theta_res; L_res) ||_2 + eps)
```

Default pass threshold:

```text
firewall_leak_param < 1e-8
```

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

## Smoke Evidence: 2026-06-26 Cora Seed 0

Stage: **smoke only**.  
Dataset / split: Cora, stratified random `1:1:8`, seed `0`.  
Evaluator: frozen encoder + Logistic Regression; evaluator `C` selected by validation accuracy, not test.

Artifacts:

- summary: `results/summary/dsr_smoke_Cora_seed0_20260626T095525Z_summary.md`
- raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_smoke_Cora_seed0_20260626T095525Z/`
- logs: `logs/dsr_smoke/dsr_smoke_Cora_seed0_20260626T095525Z/`
- config: `configs/dsr_smoke.yaml`
- runner: `scripts/run_dsr_smoke.py`

Smoke table (`test@best`, percent):

| ID | Variant | Main embedding | Test@best |
|---|---|---:|---:|
| A0 | GRACE baseline | `grace` | 84.78 |
| A2 | semantic-only negative-free | `z_sem` | 78.27 |
| A3 | residual-only InfoNCE | `z_res` | 30.12 |
| A9 | DSR-full firewall | `concat(z_sem, stopgrad(z_res))` | 77.08 |
| A5 | DSR-no-firewall | `concat` | 81.13 |
| A4 | same-parameter single-head | `single` | 83.63 |

Mechanism diagnostics:

- A9 `negative_gradient_leakage = 0.0`, so the parameter-isolated firewall is measurable and passes the mechanical leakage threshold in this implementation.
- A5 `negative_gradient_leakage = 1.2867` at the final logged epoch, confirming the diagnostic detects semantic-channel negative gradients when semantic InfoNCE is enabled.
- A9 rank/correlation diagnostics: `rank_sem=6.83`, `rank_res=7.22`, `rank_concat=13.36`, `branch_correlation=0.1168`.

Smoke interpretation:

- C1 is mechanically supported: negative loss can be kept out of semantic parameters in A9.
- C2/C3/C4 are not supported by this smoke: DSR-full does not beat semantic-only, no-firewall, same-head, or GRACE on Cora seed 0.
- This is not a formal result and cannot support SOTA, robustness, or comprehensive performance claims.

## Decision

**REVISE/PIVOT_REQUIRED**.  
The firewall diagnostic itself is valid, but the current DSR-full objective does not show smoke-level mechanism advantage. Do not proceed to formal experiments or paper claims unless a revised mechanism first passes additional pilot gates.

## Audit-Smoke Evidence: 2026-06-26 Cora Seed 0

Stage: **audit-smoke only**.  
Purpose: audit the failed smoke and correct unfair ablations, not improve results.

Artifacts:

- audit note: `refine-logs/DSR_AUDIT_SMOKE_NOTE_20260626_101103.md`
- summary: `results/summary/dsr_audit_smoke_Cora_seed0_20260626T101103Z_summary.md`
- raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`
- logs: `logs/dsr_smoke/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`

Audit additions:

- A5a: semantic InfoNCE only, no residual InfoNCE.
- A5b: budget-matched no-firewall, semantic/residual InfoNCE each weighted `0.5`.
- A5c: scaled no-firewall, loss scale matched to A9 as closely as possible.
- A4b: parameter-matched single-head control (`402120` parameters vs A9 `400256`, relative gap about `0.47%`).

Key audit-smoke results (`test@best`, percent):

| ID | Variant | Test@best |
|---|---|---:|
| A0 | GRACE baseline | 84.78 |
| A2 | semantic-only negative-free | 78.27 |
| A3 | residual-only InfoNCE | 30.21 |
| A9 | DSR-full firewall | 78.69 |
| A5b | budget-matched no-firewall | 79.57 |
| A5c | scaled no-firewall | 81.00 |
| A4 | original same-head | 83.63 |
| A4b | parameter-matched single-head | 80.81 |

Final audit-smoke decision: **PIVOT_REQUIRED**.  
Triggered rules: residual-only remains extremely low; A9 is below A4/A4b; fairer no-firewall A5b/A5c are above A9. Current evidence argues against continuing DSR-GCL formal under the present residual/firewall mechanism.

## Files

- `idea-stage/FN_IDEAS_BEYOND_IGT.md`
- `idea-stage/IDEA_CANDIDATES.md`
- `idea-stage/DSR_GCL_NOVELTY_BRIEF.md`
- `idea-stage/IDEA_REPORT.md`
- `refine-logs/DSR_MECHANISM_SPEC.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `configs/dsr_smoke.yaml`
- `scripts/run_dsr_smoke.py`
- `results/summary/dsr_smoke_Cora_seed0_20260626T095525Z_summary.md`
- `refine-logs/DSR_AUDIT_SMOKE_NOTE_20260626_101103.md`
- `results/summary/dsr_audit_smoke_Cora_seed0_20260626T101103Z_summary.md`
- `.aris/traces/novelty-check/2026-06-26_run03/`
