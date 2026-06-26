# Experiment Tracker: CAST Certificate

## Current Gate

- Status：`CAST_CERTIFICATE_SMOKE_POSITIVE`
- Decision：`GO_TO_PILOT_PLANNING_WITH_CAUTION`
- Completed action：implemented real latent target-prediction CAST certificate and ran Cora seed=0 smoke only。
- Forbidden：formal 10-seed run、SOTA/performance claim、直接把 smoke 写成论文主表。

## Smoke Matrix

| ID | Variant | Status | Test@best-val | Label agreement | Decision |
|---|---|---|---:|---:|---|
| C0 | GRACE | completed | 84.78 | 0.0000 | reference |
| C1 | embedding kNN | completed | 85.19 | 0.8152 | label-agreement control |
| C2 | CAST proxy | completed | 85.70 | 0.7548 | strong proxy control |
| C3 | CAST candidate pool only | completed | 85.70 | 0.7548 | same as proxy |
| C4 | latent target certificate | completed | 85.93 | 0.7911 | positive smoke |
| C5 | certificate + CAST score | completed | 86.16 | 0.7886 | best smoke |

## Next Allowed Step

Plan Pilot-A only:

- datasets: Cora, CiteSeer, PubMed first;
- seeds: start with 0-2 only, not formal 10 seeds;
- must include kNN, PPR, CAST proxy, C4, C5;
- add overlap/partial-correlation diagnostics to prove not kNN/PPR mining.

No formal claim is allowed until Pilot-A passes and then a separate formal protocol is frozen.

