# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | smoke | vanilla GRACE | Cora seed0 `1:1:8` | loss/logreg/rank | MUST | TODO | baseline sanity |
| R002 | M0 | smoke | CER-GCL default | Cora seed0 `1:1:8` | mask/rank/variance | MUST | TODO | no performance claim |
| R003 | M1 | gate diagnostic | q_i stats only | Cora/CiteSeer seed0 | corr degree/uncertainty | MUST | TODO | no training variant |
| R004 | M2 | routing-vs-weighting | vanilla/global weak/scalar/CER/random | Cora seeds0-2 | acc + buckets | MUST | TODO | first decisive pilot |
| R005 | M2 | routing-vs-weighting | vanilla/global weak/scalar/CER/random | CiteSeer seeds0-2 | acc + buckets | MUST | TODO | replicate pilot |
| R006 | M3 | denominator policy | keep-stopgrad/detach/mask | Cora seeds0-2 | buckets/rank/acc | MUST | TODO | decide default |

