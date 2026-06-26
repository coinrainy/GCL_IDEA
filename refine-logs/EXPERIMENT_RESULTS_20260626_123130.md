# Experiment Results: R2-IRIS Smoke

## Scope

- Skill context: continuation after `/experiment-bridge`
- Method: `R2-IRIS / Residualized Response Invariant Signatures`
- Status label: `smoke`
- Dataset: `Cora`
- Seed: `0`
- Split: project stratified random `1:1:8`
- Evaluator: frozen encoder + `LogisticRegression`, `C` selected by train/validation only
- Result JSON: `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.json`
- Result Markdown: `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.md`
- Raw directory: `results/raw/Cora/IRIS_GCL_SMOKE/r2_iris_smoke_Cora_seed0_20260626T123039Z/`
- Log directory: `logs/iris_smoke/r2_iris_smoke_Cora_seed0_20260626T123039Z/`

This is still a single-seed diagnostic smoke. It does not support formal, SOTA, robust, or comprehensive claims.

## Key Result

The residualized response idea fixes the hard anti-proximity collapse, but does not yet justify pilot escalation.

| ID | Variant | Test@best-val | Label agreement | FN mass after closure |
|---|---|---:|---:|---:|
| I1 | kNN multi-positive | 85.29 | 0.8151 | 0.2182 |
| I3 | PMGCL-lite/BMM | 85.52 | 0.7699 | 0.2187 |
| I4 | CAST one-step proxy | 85.65 | 0.7548 | 0.2188 |
| I5 | old IRIS hard anti-proximity | 76.29 | 0.2419 | 0.2221 |
| I7 | raw response no anti-proximity | 84.64 | 0.7783 | 0.2186 |
| I10 | R2 residualized response | 84.46 | 0.7783 | 0.2186 |
| I12 | residual response + soft proximity penalty | 84.18 | 0.7601 | 0.2187 |
| I13 | residual response + CAST hybrid | 84.87 | 0.7968 | 0.2184 |

## Interpretation

- The hard anti-proximity rule is the main failure source: I5 remains bad.
- Residualized response (I10) recovers to the raw response/no-anti-proximity level, so the residualization fix is useful as a rescue, not as a breakthrough.
- I13 is the strongest new variant on label agreement among non-kNN controls, but it still does not beat kNN on label agreement and does not beat PMGCL/CAST on accuracy.
- The result suggests response residuals are not useless, but current IRIS/R2-IRIS is still too close to ordinary positive mining to justify Pilot-A/B.

## Decision

`REVISE_NOT_PILOT`

Do not run multi-dataset or formal experiments yet. The next mechanism change should target why response residualization selects almost the same siblings as raw response/no-anti-proximity and whether a response-transport hybrid can become meaningfully distinct from kNN/PPR/CAST controls.

