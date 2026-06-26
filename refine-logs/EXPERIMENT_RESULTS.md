# Experiment Results: CPR-IRIS Weight Smoke

Fixed entry for the latest IRIS/R2/CPR smoke result.

- Timestamped source: `refine-logs/EXPERIMENT_RESULTS_20260626_123827.md`
- Summary JSON: `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.json`
- Summary Markdown: `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.md`
- Decision: `REVISE_TOWARD_CERTIFIED_CAST`

Response residual works best as a certificate over CAST/kNN candidates, not as an independent selector. I17 reaches `85.52` test@best-val and label agreement `0.7953`, close to CAST accuracy `85.65` while improving CAST label agreement `0.7545`; however it does not clearly beat CAST or kNN, so no Pilot-A/B or formal run is supported.

