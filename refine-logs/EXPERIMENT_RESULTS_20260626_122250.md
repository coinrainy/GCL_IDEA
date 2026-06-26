# Experiment Results: IRIS-GCL Smoke

## Scope

- Skill: `/experiment-bridge`
- Method: `IRIS-GCL / Interventional Response Invariant Signatures`
- Status label: `smoke`
- Dataset: `Cora`
- Seed: `0`
- Split: project stratified random `1:1:8`
- Evaluator: frozen encoder + `LogisticRegression`, `C` selected by train/validation only
- Result JSON: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.json`
- Result Markdown: `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md`
- Raw directory: `results/raw/Cora/IRIS_GCL_SMOKE/iris_smoke_Cora_seed0_20260626T121843Z/`
- Log directory: `logs/iris_smoke/iris_smoke_Cora_seed0_20260626T121843Z/`

This is a single-seed relation-diagnostic smoke. It does not support formal, SOTA, robust, or comprehensive claims.

## Variant Results

| ID | Variant | Test@best-val | Label agreement | FN mass after closure | Partial corr |
|---|---|---:|---:|---:|---:|
| I0 | GRACE | 84.78 | 0.0000 | 0.2746 | 0.3707 |
| I1 | kNN multi-positive | 85.24 | 0.8154 | 0.2058 | 0.3707 |
| I2 | PPR/diffusion positives | 83.58 | 0.7272 | 0.2180 | 0.3707 |
| I3 | PMGCL-lite/BMM | 85.38 | 0.7699 | 0.2017 | 0.3707 |
| I4 | CAST one-step proxy | 85.65 | 0.7548 | 0.2188 | 0.3707 |
| I5 | IRIS full | 76.48 | 0.2418 | 0.2221 | 0.3707 |
| I6 | IRIS response-shuffled | 80.07 | 0.1027 | 0.2380 | 0.3707 |
| I7 | IRIS no anti-proximity | 84.59 | 0.7787 | 0.2052 | 0.3707 |
| I8 | IRIS structural-signature-only | 77.68 | 0.1125 | 0.2384 | 0.3707 |
| I9 | IRIS no gradient-proxy | 76.48 | 0.2418 | 0.2221 | 0.3707 |

## Pass/Kill Check

- Pass condition `I5 label agreement beats I1/I2/I3/I8`: failed. I5 only beats I8 and is far below I1/I2/I3.
- Pass condition `positive partial correlation after controls`: passed as a diagnostic signal (`0.3707`).
- Pass condition `I5 beats I6 and I7`: failed. I7 dominates I5 on test accuracy and label agreement; I6 also has higher test accuracy.
- Pass condition `I5 not dominated by CAST`: failed. I4 CAST proxy dominates I5 on test accuracy, label agreement, and false-negative mass after closure.
- Kill rule `I7 no anti-proximity beats I5`: triggered.
- Kill rule `CAST matches/dominates IRIS with lower cost`: triggered.
- I9 duplicates I5 by design because positive-gradient proxy had already been removed before smoke; this is a disclosure item, not evidence of a gradient-proxy contribution.

## Decision

`PIVOT_REQUIRED`

IRIS-GCL as currently specified should not proceed to Pilot-A/B or formal experiments. The anti-proximity response-invariance mechanism appears harmful in this smoke: removing anti-proximity turns the method back into stronger proximity mining. The useful residual signal is that response similarity has positive partial correlation with labels after controls, but the current selection rule fails to convert that signal into good sibling relations.

