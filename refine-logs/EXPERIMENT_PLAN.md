# Experiment Plan: CAST-GCL Revised Smoke

## Status

`REVISE_TO_CAST_REVISED_PRE_SMOKE`

本计划只定义最小 smoke。不得把 smoke 结果写成 formal、SOTA、robust 或 comprehensive。

## Dataset And Protocol

- Dataset：Cora only。
- Seed：0 only。
- Split：project stratified random `1:1:8` split；必须复用/保存 JSON split。
- Evaluator：frozen encoder + Logistic Regression；`C` 等参数只能由 train/val 或 train 内部 CV 决定。
- Reporting：保存 config、commit、seed、命令、日志、raw JSON、summary。

## Systems

| ID | System | Purpose |
|---|---|---|
| C0 | GRACE | base contrastive reference |
| C1 | WILLOW same-node certificate | certificate without cross-node closure |
| C2 | kNN multi-positive | simple positive mining control |
| C3 | PMGCL-lite/BMM positive mining | probabilistic mining control |
| C4 | PPR/diffusion positives | graph proximity control |
| C5 | candidate-pool-only positives | candidate source control |
| C6 | similarity-only transport | removes certificate/edit path |
| C7 | edit-cost-only / anchor-drift-only | energy component controls |
| C8 | certificate-shuffled CAST | certificate semantic control |
| C9 | random transport budget-matched | search budget control |
| C10 | CAST one-step transport | main minimal variant |
| C11 | CAST two-step transport | main extended variant |

## Smoke Metrics

- LogReg test accuracy at best validation checkpoint。
- transported positives per anchor。
- offline label agreement of transported positives。
- transport energy vs label agreement correlation。
- false-negative repulsion mass before/after closure。
- positive gradient norm。
- search time and candidate budget。

## Pass Conditions

CAST can only move beyond pre-review if:

- C10/C11 have higher transported-positive label agreement than C2/C3/C4/C5/C6 diagnostics。
- C10/C11 reduce false-negative repulsion mass without simply increasing positive count。
- C10/C11 beat C8 and C9 under the same budget。
- C1 cannot fully explain C10/C11。

## Kill Rules

- C2/C3/C4/C5/C6 matches C10/C11：ordinary positive mining/proximity is enough。
- C8 matches C10/C11：certificate is not semantic。
- C9 matches C10/C11：transport path is not meaningful。
- transport energy does not correlate with offline label agreement：kill。
- gains only come from more positives or more compute：invalid。

## Next Gate

Fresh `gcl_scientific_reviewer` returned `REVISE`, novelty `6.8/10`. Next step can only be Cora seed=0 smoke with all controls above. No Pilot-A or formal run is allowed yet.
