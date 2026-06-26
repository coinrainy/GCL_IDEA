# Experiment Plan: IRIS-GCL Smoke

## Status

`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`

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
| I0 | GRACE | base reference |
| I1 | kNN multi-positive | proximity mining |
| I2 | PPR/diffusion positives | graph proximity |
| I3 | PMGCL-lite/BMM | probabilistic positive mining |
| I4 | CAST one-step | strongest previous challenger/control |
| I5 | IRIS full | main candidate |
| I6 | IRIS response-shuffled | response semantic test |
| I7 | IRIS no anti-proximity | collapse-to-proximity test |
| I8 | IRIS structural-signature-only | role-equivalence test |
| I9 | IRIS no gradient-proxy | circularity test |

## Smoke Metrics

- LogReg test accuracy at best validation checkpoint。
- response sibling count per anchor。
- anti-proximity retained coverage。
- offline label agreement of response siblings。
- partial correlation between response similarity and label agreement after controlling feature similarity, embedding similarity, PPR proximity, and degree。
- false-negative repulsion mass before/after relation closure。
- runtime and candidate budget。

## Pass Conditions

IRIS only moves beyond smoke if:

- I5 label agreement beats I1/I2/I3/I8 under the same positive budget。
- I5 response similarity has positive partial correlation with label agreement after controls。
- I5 beats I6 and I7 on diagnostics, proving response semantics and anti-proximity matter。
- I5 is not dominated by I4 CAST on label agreement, false-negative repulsion mass, and runtime。

## Kill Rules

- I6 response-shuffled matches I5：kill。
- I8 structural-signature-only matches I5：kill or major pivot。
- I7 no anti-proximity beats I5：IRIS collapses to proximity mining。
- response similarity has no partial correlation with label agreement after controlling feature/embedding/PPR/degree：kill。
- CAST matches IRIS on diagnostics with lower cost：reconsider CAST or pivot。
- Any gain only comes from more positives, more compute, or broader candidate budget：invalid。

## Next Gate

Only Cora seed=0 smoke is allowed. No Pilot-A or formal run is allowed yet.
