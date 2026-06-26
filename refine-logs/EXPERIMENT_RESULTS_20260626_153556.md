# BEACON-GCL BE-M0-001 Smoke Results

- 时间：2026-06-26T15:35:56Z
- Skill：`/experiment-bridge`
- 阶段：smoke，不是 pilot/formal。
- 数据集：Cora seed0，stratified random `1:1:8`
- Evaluator：frozen GRACE encoder + Logistic Regression train/val C selection
- 结果路径：`results/summary/be_m0_001_Cora_seed0_20260626T153505Z_summary.md/json`
- 日志路径：`logs/beacon_smoke/be_m0_001_Cora_seed0_20260626T153505Z/`

## Results

| ID | Variant | Test@best-val | Agreement | Accepted/anchor | kNN ov | PPR ov | CAST ov | C4 ov | Margin delta | Rank delta | Uniformity delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | GRACE | 84.73 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B1 | embedding_knn | 85.29 | 0.8150 | 10.00 | 1.0000 | 0.4064 | 0.2852 | 0.2552 | 0.0988 | -11.2454 | 0.0708 |
| B2 | cast_proxy | 85.70 | 0.7548 | 10.00 | 0.2852 | 0.2017 | 1.0000 | 0.1501 | 0.0767 | -12.4232 | 0.1450 |
| B3 | c4_latent_certificate | 84.82 | 0.7215 | 10.00 | 0.2552 | 0.1765 | 0.1501 | 1.0000 | 0.0352 | -15.2329 | 0.2027 |
| B4 | all_candidate_neutralization | 85.42 | 0.7285 | 28.32 | 0.3531 | 0.3274 | 0.3531 | 0.3531 | 0.0525 | -13.7872 | 0.1729 |
| B5 | beacon_full_gate | 84.87 | 0.8611 | 8.49 | 0.5884 | 0.4493 | 0.5537 | 0.3219 | 0.3063 | -9.1365 | 0.0760 |
| B6 | shuffled_utility_gate | 85.01 | 0.8198 | 8.75 | 0.4026 | 0.3380 | 0.3628 | 0.3509 | 0.3936 | -13.1646 | 0.1388 |
| B7 | no_boundary_gate | 85.15 | 0.8323 | 8.54 | 0.6119 | 0.4959 | 0.5821 | 0.3200 | 0.0976 | -8.3762 | 0.0767 |
| B8 | no_geometry_gate | 85.24 | 0.8479 | 8.75 | 0.4757 | 0.4323 | 0.6079 | 0.3284 | 0.3703 | -10.3370 | 0.1036 |

## Reading

- B5 has the highest offline label agreement (`0.8611`) and a positive probe-margin delta (`0.3063`), so the gate is selecting geometrically safer pairs.
- That safety did not become accuracy: B5 `84.87` is below kNN `85.29`, CAST proxy `85.70`, and all-candidate neutralization `85.42`.
- B6 shuffled utility (`85.01`) is not worse than B5, which triggers the shuffled-gate kill warning.
- B7 no-boundary (`85.15`) and B8 no-geometry (`85.24`) both exceed B5, so the current full utility combination is over-conservative or miscalibrated.
- This result repeats the broader lesson from CAST: better-looking pair diagnostics do not automatically imply better node classification.

## Decision

`KILL_CURRENT_BEACON_GATE_NO_BE_M1`

Do not run BE-M1 Cora/CiteSeer/PubMed seeds. Do not run Pilot-B, formal experiments, paper tables, or `/auto-review-loop`.

Allowed next actions:

1. revise the BEACON utility model on Cora seed0 only, with a pre-registered change that directly addresses why B6/B7/B8 beat B5; or
2. pivot again to a different mechanism that does not rely on post-hoc relation smoothing.

No performance claim is supported.
