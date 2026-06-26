# Experiment Results: R2-IRIS Smoke

Fixed entry for the latest IRIS/R2-IRIS smoke result.

- Timestamped source: `refine-logs/EXPERIMENT_RESULTS_20260626_123130.md`
- Summary JSON: `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.json`
- Summary Markdown: `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.md`
- Decision: `REVISE_NOT_PILOT`

R2 residualization rescues the old IRIS hard anti-proximity failure but does not yet beat the strongest controls. I13 hybrid reached `84.87` test@best-val and label agreement `0.7968`; I4 CAST proxy remains stronger on accuracy (`85.65`), and I1 kNN remains stronger on label agreement (`0.8151`). No Pilot-A/B or formal run is supported.

