# Experiment Results: CPR-IRIS Weight Smoke

## Scope

- Method branch: `CPR-IRIS / response-certified proximity`
- Status label: `smoke`
- Dataset: `Cora`
- Seed: `0`
- Split: project stratified random `1:1:8`
- Evaluator: frozen encoder + `LogisticRegression`, `C` selected by train/validation only
- Result JSON: `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.json`
- Result Markdown: `results/summary/cpr_weight_smoke_Cora_seed0_20260626T123736Z_summary.md`
- Raw directory: `results/raw/Cora/IRIS_GCL_SMOKE/cpr_weight_smoke_Cora_seed0_20260626T123736Z/`
- Log directory: `logs/iris_smoke/cpr_weight_smoke_Cora_seed0_20260626T123736Z/`

This is still a single-seed diagnostic smoke. It does not support formal, SOTA, robust, or comprehensive claims.

## Key Result

| ID | Variant | Test@best-val | Label agreement | FN mass after closure |
|---|---|---:|---:|---:|
| I1 | kNN multi-positive | 85.29 | 0.8153 | 0.2182 |
| I3 | PMGCL-lite/BMM | 85.38 | 0.7698 | 0.2186 |
| I4 | CAST one-step proxy | 85.65 | 0.7545 | 0.2188 |
| I17 | CAST score + residual certificate w=0.30 | 85.52 | 0.7953 | 0.2184 |
| I18 | CAST score + residual certificate w=0.15 | 85.47 | 0.7817 | 0.2185 |
| I19 | CAST score + residual certificate w=0.45 | 85.38 | 0.8020 | 0.2183 |
| I20 | kNN score + residual certificate w=0.15 | 85.15 | 0.8120 | 0.2182 |

## Interpretation

- Response residual is more useful as a certificate over strong proximity/CAST candidates than as a standalone selector.
- CPR variants do not clearly beat CAST on accuracy, so there is still no pilot gate.
- CPR variants improve over CAST proxy on offline sibling label agreement while keeping accuracy near CAST.
- kNN remains the strongest label-agreement control, so the mechanism must still prove it is not ordinary kNN positive mining.

## Decision

`REVISE_TOWARD_CERTIFIED_CAST`

Do not run multi-dataset or formal experiments yet. Continue mechanism refinement around CAST/proximity candidates certified by response residuals.

