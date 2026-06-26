# Experiment Results: Certified CAST Closure Smoke

Fixed entry for the latest IRIS/R2/CPR closure result.

- Timestamped source: `refine-logs/EXPERIMENT_RESULTS_20260626_124728.md`
- Summary JSON: `results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.json`
- Summary Markdown: `results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.md`
- Decision: `KILL_CERTIFIED_CLOSURE_KEEP_CPR_SCORE`

The three-way certified closure mechanism is distinct but not strong: I22 reaches `84.96` test@best-val and label agreement `0.7472`, below CAST/I17. It lowers false-negative mass after closure (`0.2009`) mainly by neutralizing many pairs, which is not enough to justify Pilot-A/B. The best surviving signal is still additive response-certified CAST scoring (I17/I18/I19).

