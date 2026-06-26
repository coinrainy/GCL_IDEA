# Experiment Tracker: CAST Certificate Pilot-A

Fixed entry for the latest Pilot-A tracker.

- Timestamped source: `refine-logs/EXPERIMENT_TRACKER_20260626_140139.md`
- Current gate: `PA_M0_SANITY_COMPLETED`
- Status: Pilot-A main runs not started

## Current Run Status

| Run ID | Milestone | Purpose | Dataset | Seeds | Status | Notes |
|--------|-----------|---------|---------|-------|--------|-------|
| PA-M0-001 | M0 | sanity | CiteSeer | 0 | completed | 2 GRACE epochs + 2 certificate epochs; chain passed; not a Pilot result |
| PA-M1-CORA | M1 | Pilot-A main | Cora | 0,1,2 | TODO | C0-C5 |
| PA-M1-CITE | M1 | Pilot-A main | CiteSeer | 0,1,2 | TODO | C0-C5 |
| PA-M1-PUB | M1 | Pilot-A main | PubMed | 0,1,2 | TODO | C0-C5 |
| PA-M2-001 | M2 | novelty diagnostics | all | 0,1,2 | TODO | overlap + partial correlation |
| PA-M3-001 | M3 | decision audit | all | 0,1,2 | TODO | GO/REVISE/PIVOT |

## PA-M0-001 Sanity Output

- Summary JSON: `results/summary/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z_summary.json`
- Summary MD: `results/summary/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z_summary.md`
- Logs: `logs/cast_certificate/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z/`

First full Pilot-A run to launch next:

1. `PA-M1-CORA`: Cora seeds 0-2 with C0-C5.

No formal 10-seed run or performance claim is allowed.

