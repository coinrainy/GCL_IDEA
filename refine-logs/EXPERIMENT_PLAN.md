# Experiment Plan: CAST-GCL Pre-review Smoke

## Status

`REVISE_TO_CAST_PRE_REVIEW`

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
| C4 | certificate-shuffled CAST | certificate semantic control |
| C5 | random transport budget-matched | search budget control |
| C6 | CAST one-step transport | main minimal variant |
| C7 | CAST two-step transport | main extended variant |

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

- C6/C7 have higher transported-positive label agreement than C2/C3/C5 diagnostics。
- C6/C7 reduce false-negative repulsion mass without simply increasing positive count。
- C6/C7 beat C4 and C5 under the same budget。
- C1 cannot fully explain C6/C7。

## Kill Rules

- C2 or C3 matches C6/C7：ordinary positive mining is enough。
- C4 matches C6/C7：certificate is not semantic。
- C5 matches C6/C7：transport path is not meaningful。
- transport energy does not correlate with offline label agreement：kill。
- gains only come from more positives or more compute：invalid。

## Next Gate

Before implementation-heavy work, run fresh `gcl_scientific_reviewer` on CAST or implement the Cora seed=0 smoke only. No Pilot-A or formal run is allowed yet.
