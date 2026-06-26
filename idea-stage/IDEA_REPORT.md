# Idea Report: After IRIS / CPR Smoke

## Current Decision

`PIVOT_REQUIRED`

IRIS/CPR is no longer the active best idea for pilot escalation. Multiple Cora seed=0 smoke runs show:

- hard anti-proximity IRIS fails;
- residual response rescues the failure but does not beat strong controls;
- response-certified CAST scoring is the only surviving weak signal;
- explicit certified closure is distinct but weaker.

## Best Available Signal

The best surviving variant is additive response-certified CAST scoring:

- I17 CAST + residual certificate: `85.56` test@best-val, label agreement `0.7953`;
- I4 CAST proxy: `85.70`, label agreement `0.7548`;
- I1 kNN: `85.24-85.29`, label agreement around `0.815`.

This is not enough for Pilot-A/B or formal experiments.

## Next Direction

Do not continue IRIS/CPR by adding more score weights or threshold variants. The next valid step is one of:

1. implement real CAST latent target-prediction transport certificate;
2. restart idea discovery using the IRIS/CPR failures as hard constraints.

Canonical next-direction file: `idea-stage/NEXT_DIRECTION_AFTER_IRIS_CPR.md`.

