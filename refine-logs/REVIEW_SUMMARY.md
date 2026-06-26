# Review Summary: IRIS-GCL

## Fresh Reviewer Verdict

- Comparison：CAST-GCL vs IRIS-GCL
- Overall verdict：`SWITCH_TO_IRIS`
- Decision label：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`
- CAST novelty：`6.5/10`
- IRIS novelty：`7.2/10`
- Confidence for IRIS：`0.61`

## Why IRIS Was Selected

CAST is more mature, but its core novelty risk is too close to Graph-JEPA-like certificate + candidate mining + multi-positive GCL. IRIS is higher risk but has a cleaner novelty hook: cross-node interventional response-function equivalence under anti-proximity.

## Required Revisions Before Smoke

- Remove `positive-gradient proxy` from minimal fingerprint。
- Add degree, homophily, feature similarity, embedding similarity, and PPR deconfounding diagnostics。
- Report anti-proximity coverage and label agreement。
- Add structural-role baseline。
- Keep CAST one-step as mandatory control。

## Decision

`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`
