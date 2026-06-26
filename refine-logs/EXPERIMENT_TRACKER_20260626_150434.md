# Experiment Tracker: BEACON-GCL

- Created: 2026-06-26T15:04:34Z
- Stage: smoke planning
- Current gate: `BEACON_SMOKE_NOT_STARTED`
- Decision: `GO_TO_BEACON_SMOKE_PLANNING`

| Run ID | Milestone | Purpose | Dataset | Seeds | Status | Notes |
|---|---|---|---|---|---|---|
| BE-M0-001 | M0 | sanity smoke | Cora | 0 | TODO | implement BEACON-lite post-hoc gate |
| BE-M1-CORA | M1 | pilot-lite | Cora | 0,1,2 | gated | run only if BE-M0 passes |
| BE-M1-CITE | M1 | pilot-lite | CiteSeer | 0,1,2 | gated | run only if BE-M0 passes |
| BE-M1-PUB | M1 | resource-cautious pilot-lite | PubMed | 0 first | gated | expand to 1,2 only if seed0 positive |
| BE-M2-001 | M2 | mechanism diagnostics | all available | mixed | gated | boundary/geometry/overlap diagnostics |

## Immediate Next Command

Do not run yet until code exists:

```bash
python scripts/run_beacon_smoke.py --dataset Cora --seed 0 --run-tag be_m0_001 --status smoke
```

## No-go Carryover

CAST Certificate Pilot-A remains `REVISE_OR_PIVOT_BEFORE_ANY_MORE_PILOT`. BEACON is a pivot, not a CAST continuation.
