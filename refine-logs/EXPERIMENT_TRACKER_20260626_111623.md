# Experiment Tracker: SIVA-GCL-positive-core

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| SIVA-R001 | M0 | critic sanity | masked-context critic | Cora `1:1:8` seed 0 | recon, stability | MUST | TODO | no claim |
| SIVA-R002 | M1 | intervention sanity | SIVA-positive generator | Cora `1:1:8` seed 0 | view distance, stability | MUST | TODO | greedy search first |
| SIVA-R003 | M2 | baseline parity | GRACE | Cora `1:1:8` seed 0 | LogReg, positive metrics | MUST | TODO | wrapper parity |
| SIVA-R004 | M2 | adaptive aug control | GCA-style augmentation | Cora `1:1:8` seed 0 | LogReg, positive metrics | MUST | TODO | rule out normal aug |
| SIVA-R005 | M2 | main smoke | SIVA-positive core | Cora `1:1:8` seed 0 | LogReg, gradient, similarity | MUST | TODO | core mechanism |
| SIVA-R006 | M3 | critic control | GraphMAE-only / critic-only | Cora `1:1:8` seed 0 | LogReg, recon | MUST | TODO | rule out reconstruction |
| SIVA-R007 | M3 | shuffled critic control | critic-shuffled SIVA | Cora `1:1:8` seed 0 | LogReg, stability | MUST | TODO | rule out critic artifact |
| SIVA-R008 | M3 | random intervention control | random matched intervention | Cora `1:1:8` seed 0 | LogReg, stability | MUST | TODO | rule out stronger perturbation |
| SIVA-R009 | M4 | citation pilot | best SIVA-positive core | Cora/CiteSeer/PubMed seeds 0-4 | mean/std, mechanism metrics | GATED | BLOCKED | only after smoke passes |

## Current Gate

`BLOCKED_FOR_IMPLEMENTATION`: 当前只有 idea/refinement 文档和 reviewer 审查，无 SIVA 代码或结果。不得进入 formal。

