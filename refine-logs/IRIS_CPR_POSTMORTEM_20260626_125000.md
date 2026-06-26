# IRIS / CPR Postmortem

## Verdict

`STOP_IRIS_CPR_INCREMENTAL_REFINEMENT`

IRIS/CPR has produced enough Cora seed=0 smoke evidence to stop incremental refinements. The line should not continue via more I23/I24-style variants.

## What Failed

| Stage | Main variant | Outcome |
|---|---|---|
| IRIS hard anti-proximity | I5 | failed badly; anti-proximity removed useful positives |
| R2 residualized response | I10/I13 | rescued the collapse but did not beat strong controls |
| CPR additive certificate | I17/I19 | best surviving signal; near CAST accuracy with better label agreement, but not a clear win |
| Certified CAST Closure | I22 | distinct but weaker; lowered FN mass by neutralizing too many pairs |

## Evidence Summary

| Variant | Test@best-val | Label agreement | FN mass after closure | Interpretation |
|---|---:|---:|---:|---|
| I4 CAST proxy | 85.70 | 0.7548 | 0.2188 | strongest accuracy control |
| I17 CAST + residual cert | 85.56 | 0.7953 | 0.2185 | best CPR tradeoff |
| I19 CAST + residual cert strong | 85.42 | 0.8021 | 0.2184 | higher agreement, lower accuracy |
| I20 kNN + residual cert light | 85.15 | 0.8120 | 0.2183 | too close to kNN |
| I22 certified closure | 84.96 | 0.7472 | 0.2009 | neutralizes many pairs but hurts positive quality |

## Lessons

- Response residual contains some signal, but it is not strong enough as a standalone relation criterion.
- Hard anti-proximity is harmful under the current fingerprint.
- Additive response certification over CAST is the only useful remnant.
- Neutral closure can lower false-negative mass without improving classification; this is not enough for a paper claim.
- A viable next idea must avoid both failure modes: ordinary proximity mining and excessive neutralization.

## Next Gate

`PIVOT_REQUIRED`

Two allowed next paths:

1. **Real CAST implementation only**: implement the original latent target-prediction transport certificate, not the current CAST proxy. This is allowed only if it can produce a relation score that is not reducible to feature/embedding/PPR similarity.
2. **Fresh idea discovery**: restart from the false-negative direction using the above failures as hard constraints.

Forbidden next paths:

- adding more response-residual weights;
- changing quantiles on I22;
- running multi-dataset/pilot/formal from any IRIS/CPR variant;
- claiming improvement from smoke numbers.

