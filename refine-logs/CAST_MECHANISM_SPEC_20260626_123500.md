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

All operators act inside a 1-hop or 2-hop ego context and have fixed budget.

| Operator | Definition | Cost |
|---|---|---:|
| `feature_replace(i -> j, S)` | replace anchor-ego feature dimensions `S` with candidate-ego feature statistics | `|S| / d` |
| `edge_keep(i -> j, R)` | keep anchor edges whose endpoint role also appears in candidate ego role set `R` | dropped edge ratio |
| `neighbor_substitute(i -> j, q)` | substitute at most `q` anchor neighbors with candidate neighbors matched by degree bin and feature cosine bin | `q / deg(i)` |
| `ego_mix(i, j, alpha)` | convex mix of normalized ego feature summaries for certificate query only | `alpha` |

The mixed ego is used only for certificate scoring, not inserted into the real graph.

## Certificate Error

For a step `u -> v`, construct a mixed query ego:

```text
ego_mix(u, v, a) = apply_ops(ego(u), ego(v), a)
```

Then compute:

```text
cert_error(u -> v, a)
  = mean_k || P(E_context(ego_mix(u, v, a)), pos_k)
              - stopgrad(E_target(target_ego(v)^k)) ||_2^2
```

Interpretation: `u` can be transported toward `v` only if a legal local edit of `u` predicts `v`'s target ego latent manifold.

## Transport Energy

```text
E_transport(i, j)
  = min_{a, path length <= L} [
      cert_error_path(i, j, a)
      + lambda_edit * edit_cost_path(a)
      + lambda_drift * drift_from_anchor(i, path)
    ]
```

Default smoke:

- `L=1` for minimal CAST.
- `L=2` only as an extended diagnostic.
- candidate pool size fixed at `K=20`.
- max transported positives per anchor fixed at `B=5`.

## Candidate-Certificate Decoupling

Candidate pool provides only candidates; it cannot decide positives.

Required candidate pools:

- feature kNN；
- PPR/diffusion；
- frozen embedding kNN；
- random budget-matched。

CAST must report results separately for each pool and for a union pool. If candidate-pool-only matches CAST, the certificate adds no value.

## Threshold And Neutral Rules

- `tau_transport` selected by train/val diagnostics only.
- Accept top `B` candidates with `E_transport <= tau_transport`.
- Boundary candidates within `tau_transport * [1.0, 1.15]` are neutral/excluded, not positives.
- No test-label information can affect threshold or budget.

## Revised Falsifiable Claim

CAST is worth continuing only if:

```text
transport energy predicts offline label agreement
better than kNN/BMM/PPR/candidate-pool-only controls,
and the resulting closure lowers false-negative repulsion mass
under the same positive budget.
```

This is a smoke diagnostic claim, not a paper performance claim.

## Extra Controls

- candidate-pool-only positives。
- similarity-only energy。
- edit-cost-only energy。
- anchor-drift-only energy。
- certificate-shuffled CAST。
- random transport budget-matched。
- WILLOW same-node certificate。
- PMGCL-lite/BMM。

## Stop Conditions

Kill CAST if any of the following holds:

- candidate-pool-only matches CAST；
- kNN/PPR/BMM label agreement matches CAST；
- certificate-shuffled matches CAST；
- transport energy has no monotonic relation with offline label agreement；
- performance gain only comes from more positives, more compute, or longer training。
