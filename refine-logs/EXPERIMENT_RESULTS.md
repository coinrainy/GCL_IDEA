# Experiment Results: IRIS-GCL Smoke

Fixed entry for the latest `/experiment-bridge` result.

- Timestamped source: `refine-logs/EXPERIMENT_RESULTS_20260626_122250.md`
- Summary JSON: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.json`
- Summary Markdown: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md`
- Decision: `PIVOT_REQUIRED`

IRIS-GCL failed the planned Cora seed=0 smoke. I5 IRIS full reached `76.48` test@best-val and label agreement `0.2418`, while I7 no anti-proximity reached `84.59` / `0.7787`, and I4 CAST proxy reached `85.65` / `0.7548`. The current IRIS anti-proximity rule is therefore not ready for Pilot-A/B or formal experiments.

