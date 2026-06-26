# Idea Report: CAST Certificate After IRIS / CPR

## Current Decision

`GO_TO_PILOT_PLANNING_WITH_CAUTION`

IRIS/CPR incremental refinement remains stopped. The viable direction is now **CAST latent target-prediction certificate**.

## Smoke Evidence

Clean Cora seed=0 smoke:

| Variant | Test@best-val | Label agreement | kNN overlap | CAST overlap |
|---|---:|---:|---:|---:|
| GRACE | 84.78 | 0.0000 | 0.0000 | 0.0000 |
| kNN | 85.19 | 0.8152 | 1.0000 | 0.2851 |
| CAST proxy | 85.70 | 0.7548 | 0.2851 | 1.0000 |
| latent target certificate | 85.93 | 0.7911 | 0.4150 | 0.2109 |
| certificate + CAST score | 86.16 | 0.7886 | 0.3621 | 0.7606 |

## Interpretation

The real latent target-prediction certificate is the first positive smoke after several failed IRIS/CPR refinements. It improves over CAST proxy on both accuracy and label agreement, while not collapsing completely to kNN.

## Boundary

This is a smoke result only. It supports Pilot-A planning, not performance claims.

Canonical result file: `refine-logs/CAST_CERTIFICATE_RESULTS.md`.

