# Experiment Tracker: BOND-GCL

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| BOND-R001 | M0 | implementation sanity | GRACE baseline rerun wrapper | Cora `1:1:8` seed 0 | loss, LogReg, logs | MUST | TODO | confirm runner parity |
| BOND-R002 | M1 | cap-only smoke | BOND-cap-only | Cora `1:1:8` seed 0 | acc, exposure, denom effective number | MUST | TODO | no overflow |
| BOND-R003 | M2 | pair reweight control | ProGCL-style proxy / same pair weights | Cora `1:1:8` seed 0 | acc, exposure | MUST | TODO | test reweighting explanation |
| BOND-R004 | M2 | random basin control | random basins same size histogram | Cora `1:1:8` seed 0 | acc, exposure | MUST | TODO | test basin semantics |
| BOND-R005 | M2 | sampling control | E2Neg-style center/small negatives | Cora `1:1:8` seed 0 | acc, denom effective number | MUST | TODO | test small denominator explanation |
| BOND-R006 | M2 | overflow ablation | overflow-only / boundary reps only | Cora `1:1:8` seed 0 | acc, exposure | MUST | TODO | ensure overflow not hidden main source |
| BOND-R007 | M3 | citation pilot | best BOND | Cora/CiteSeer/PubMed seeds 0-4 | mean/std, diagnostics | GATED | BLOCKED | only after B1-B3 pass |
| BOND-R008 | M4 | broader dev | best BOND | DBLP/Computers/Photo seeds 0-4 | mean/std, diagnostics | GATED | BLOCKED | only after citation pilot |

## Current Gate

`BLOCKED_FOR_IMPLEMENTATION`: 当前只有 idea/refinement 文档和 reviewer 审查，无 BOND 代码或结果。不得进入 formal。

