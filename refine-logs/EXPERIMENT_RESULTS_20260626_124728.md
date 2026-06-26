# Experiment Results: Certified CAST Closure Smoke

## Scope

- Method branch: `Certified CAST Closure`
- Status label: `smoke`
- Dataset: `Cora`
- Seed: `0`
- Split: project stratified random `1:1:8`
- Evaluator: frozen encoder + `LogisticRegression`, `C` selected by train/validation only
- Result JSON: `results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.json`
- Result Markdown: `results/summary/closure_smoke_Cora_seed0_20260626T124638Z_summary.md`
- Raw directory: `results/raw/Cora/IRIS_GCL_SMOKE/closure_smoke_Cora_seed0_20260626T124638Z/`
- Log directory: `logs/iris_smoke/closure_smoke_Cora_seed0_20260626T124638Z/`

This is a single-seed diagnostic smoke and does not support formal, SOTA, robust, or comprehensive claims.

## Key Result

| ID | Variant | Test@best-val | Label agreement | FN mass after closure | kNN overlap | CAST overlap |
|---|---|---:|---:|---:|---:|---:|
| I4 | CAST one-step proxy | 85.70 | 0.7548 | 0.2188 | 0.2851 | 1.0000 |
| I17 | CAST + residual cert w=0.30 | 85.56 | 0.7953 | 0.2185 | 0.4668 | 0.6274 |
| I18 | CAST + residual cert w=0.15 | 85.56 | 0.7817 | 0.2186 | 0.3855 | 0.7978 |
| I19 | CAST + residual cert w=0.45 | 85.42 | 0.8021 | 0.2184 | 0.5146 | 0.4938 |
| I20 | kNN + residual cert w=0.15 | 85.15 | 0.8120 | 0.2183 | 0.7453 | 0.2829 |
| I22 | certified CAST closure | 84.96 | 0.7472 | 0.2009 | 0.2200 | 0.1669 |

## Interpretation

- I22 strongly lowers false-negative mass after closure (`0.2009`) because it neutralizes many candidate pairs.
- This reduction is not useful enough: positive label agreement drops below CAST (`0.7472` vs `0.7548`) and accuracy drops below I4/I17/I18/I19.
- I22 has low overlap with kNN/CAST, so it is distinct, but the distinct relation set is not high-quality.
- The best surviving signal remains I17/I18/I19: additive response-certified CAST scoring, not three-way certified closure.

## Decision

`KILL_CERTIFIED_CLOSURE_KEEP_CPR_SCORE`

Do not continue the I22 neutral-closure mechanism in its current form. If the project continues this line at all, retain only the CPR score/certificate idea as a minor CAST refinement; otherwise pivot back to CAST or restart idea discovery.

