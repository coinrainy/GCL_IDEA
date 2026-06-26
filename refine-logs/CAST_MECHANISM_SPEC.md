# CAST-GCL Mechanism Spec

日期：2026-06-26  
状态：`REVISE_TO_CAST_REVISED_PRE_SMOKE`

## Design Goal

Make CAST technically distinct from kNN/PPR/BMM positive mining by requiring a certificate-scored intervention path between real nodes, not merely similarity.

## Frozen Certificate Schedule

1. Train a latent ego target-prediction certificate with train graph only, no labels.
2. Freeze `E_online`, `E_target`, and predictor for relation construction.
3. Construct CAST closure offline before contrastive encoder training.
4. Train the GCL encoder with the fixed closure.

No closure update during smoke. This avoids moving-target confounds.

## Legal Intervention Operators

| Operator | Definition | Cost |
|---|---|---:|
| `feature_replace(i -> j, S)` | replace anchor-ego feature dimensions `S` with candidate-ego feature statistics | `|S| / d` |
| `edge_keep(i -> j, R)` | keep anchor edges whose endpoint role also appears in candidate ego role set `R` | dropped edge ratio |
| `neighbor_substitute(i -> j, q)` | substitute at most `q` anchor neighbors with candidate neighbors matched by degree bin and feature cosine bin | `q / deg(i)` |
| `ego_mix(i, j, alpha)` | convex mix of normalized ego feature summaries for certificate query only | `alpha` |

## Certificate Error

For a step `u -> v`, construct a mixed query ego and compute:

```text
cert_error(u -> v, a)
  = mean_k || P(E_context(ego_mix(u, v, a)), pos_k)
              - stopgrad(E_target(target_ego(v)^k)) ||_2^2
```

Interpretation: `u` can be transported toward `v` only if a legal local edit of `u` predicts `v`'s target ego latent manifold.

## Revised Falsifiable Claim

CAST is worth continuing only if transport energy predicts offline label agreement better than kNN/BMM/PPR/candidate-pool-only controls and lowers false-negative repulsion mass under the same positive budget.
