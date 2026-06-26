# IRIS vs CAST Fresh Review

Reviewer：fresh `gcl_scientific_reviewer`  
Decision：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`

## Scores

| Idea | Novelty | Confidence | Reviewer judgment |
|---|---:|---:|---|
| CAST-GCL | 6.5/10 | 0.74 | mature but too close to positive mining |
| IRIS-GCL | 7.2/10 | 0.61 | more novel, higher risk, better bet for sufficient innovation |

## Why Switch

CAST's novelty risk is concentrated around Graph-JEPA-like certificate + candidate mining + multi-positive GCL. IRIS has a different technical hook: cross-node interventional response-function equivalence under anti-proximity.

## Required Revisions

- Remove `positive-gradient proxy` from minimal response fingerprint.
- Add degree, homophily, feature similarity, embedding similarity, and PPR deconfounding diagnostics.
- Report anti-proximity coverage and label agreement.
- Add structural-role baseline.
- Keep CAST as mandatory control.

## Decision

`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`
