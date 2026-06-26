# Experiment Tracker: CAST Certificate Pilot-A

Fixed entry for the latest Pilot-A tracker.

- Timestamped source: `refine-logs/EXPERIMENT_TRACKER_20260626_140139.md`
- Current gate: `PA_M1_PARTIAL_COMPLETED_NO_GO`
- Status: Pilot-A stopped before PubMed seeds 1-2

## Current Run Status

| Run ID | Milestone | Purpose | Dataset | Seeds | Status | Notes |
|--------|-----------|---------|---------|-------|--------|-------|
| PA-M0-001 | M0 | sanity | CiteSeer | 0 | completed | 2 GRACE epochs + 2 certificate epochs; chain passed; not a Pilot result |
| PA-M0-001b | M0 | diagnostic sanity | CiteSeer | 0 | completed | C6 graph diffusion control + PPR overlap + sampled partial-correlation passed with 1+1 epochs |
| PA-M1-CORA | M1 | Pilot-A main | Cora | 0,1,2 | completed | C0-C6; C4/C5 below strongest mining controls |
| PA-M1-CITE | M1 | Pilot-A main | CiteSeer | 0,1,2 | completed | C0-C6; C4/C5 below strongest mining controls |
| PA-M1-PUB | M1 | Pilot-A main | PubMed | 0 | stopped_after_seed0 | seed0 ran; seeds 1-2 stopped by no-go signal |
| PA-M2-001 | M2 | novelty diagnostics | available runs | 0,1,2 / PubMed 0 | completed | C4 low overlap with kNN/PPR/CAST, but no accuracy conversion |
| PA-M3-001 | M3 | decision audit | available runs | mixed | completed | REVISE_OR_PIVOT_BEFORE_ANY_MORE_PILOT |

## PA-M0-001 Sanity Output

- Summary JSON: `results/summary/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z_summary.json`
- Summary MD: `results/summary/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z_summary.md`
- Logs: `logs/cast_certificate/pa_m0_001_sanity_CiteSeer_seed0_20260626T140957Z/`

## PA-M0-001b Diagnostic Sanity Output

- Summary JSON: `results/summary/pa_m0_001_sanity_v3_CiteSeer_seed0_20260626T141904Z_summary.json`
- Summary MD: `results/summary/pa_m0_001_sanity_v3_CiteSeer_seed0_20260626T141904Z_summary.md`
- Logs: `logs/cast_certificate/pa_m0_001_sanity_v3_CiteSeer_seed0_20260626T141904Z/`
- Code review: `refine-logs/EXPERIMENT_CODE_REVIEW_20260626_141946.md`

## PA-M1 Outputs

- Cora summaries:
  - `results/summary/pa_m1_cora_Cora_seed0_20260626T142147Z_summary.md`
  - `results/summary/pa_m1_cora_Cora_seed1_20260626T142213Z_summary.md`
  - `results/summary/pa_m1_cora_Cora_seed2_20260626T142240Z_summary.md`
- CiteSeer summaries:
  - `results/summary/pa_m1_cite_CiteSeer_seed0_20260626T142421Z_summary.md`
  - `results/summary/pa_m1_cite_CiteSeer_seed1_20260626T142458Z_summary.md`
  - `results/summary/pa_m1_cite_CiteSeer_seed2_20260626T142537Z_summary.md`
- PubMed seed0 summary:
  - `results/summary/pa_m1_pub_PubMed_seed0_20260626T142701Z_summary.md`
- Aggregate result report: `refine-logs/EXPERIMENT_RESULTS_20260626_144910.md`

No formal 10-seed run, Pilot-B, `/auto-review-loop`, or performance claim is allowed from this state.
