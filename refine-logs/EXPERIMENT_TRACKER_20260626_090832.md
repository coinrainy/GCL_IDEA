# EXPERIMENT_TRACKER: IGT-GCL

生成时间：2026-06-26T09:08:32Z  
方法：IGT-GCL / Interval-Guarded Pair Targeting  
阶段：planned only  
当前 decision：**REVISE**  
说明：本 tracker 只记录未来 smoke/pilot 计划。当前没有运行任何实验。

| ID | Stage | Dataset | Split | Seeds | Variants | Status | Required evidence | Decision gate |
|---|---|---|---|---|---|---|---|---|
| IGT-A0 | Smoke | Cora | `1:1:8` | 0 | V0,V3,V9 | TODO | no NaN, masks non-empty, logreg works, diagnostic JSON | pass -> Stage B |
| IGT-B1 | Pair diagnostic | Cora | `1:1:8` | 0,1,2 | V0,V2,V3,V8,V9 | TODO | high-l precision, low-u precision, ambiguous FN enrichment | fail -> PIVOT/KILL |
| IGT-B2 | Pair diagnostic | CiteSeer | `1:1:8` | 0,1,2 | V0,V2,V3,V8,V9 | TODO | same as B1 | fail -> PIVOT/KILL |
| IGT-B3 | Pair diagnostic | PubMed | `1:1:8` | 0,1,2 | V0,V2,V3,V8,V9 | TODO | same as B1 | fail -> PIVOT/KILL |
| IGT-C1 | Mechanism pilot | Cora | `1:1:8` | 0,1,2 | V0-V9 | TODO | full vs simple alternatives | revise if simple variant matches |
| IGT-C2 | Mechanism pilot | CiteSeer | `1:1:8` | 0,1,2 | V0-V9 | TODO | full vs simple alternatives | revise if simple variant matches |
| IGT-C3 | Mechanism pilot | PubMed | `1:1:8` | 0,1,2 | V0-V9 | TODO | full vs simple alternatives | revise if simple variant matches |
| IGT-C4 | Mechanism pilot | Computers | `1:1:8` | 0,1,2 | V0-V9 | TODO | transfer beyond Planetoid | stop broad claim if degraded |
| IGT-C5 | Mechanism pilot | Photo | `1:1:8` | 0,1,2 | V0-V9 | TODO | transfer beyond Planetoid | stop broad claim if degraded |
| IGT-D0 | Formal design review | TBD | frozen protocol | 0-9 planned | frozen subset | BLOCKED | pilot tracker complete, audit approved | only then formal |

## Variant Legend

- V0: vanilla GRACE。
- V1: vanilla GCA if adapted。
- V2: NodeSim/IFL-style point posterior。
- V3: midpoint target `(l+u)/2`。
- V4: ProGCL-style true-negative weighting。
- V5: only positive expansion。
- V6: only negative downweight。
- V7: no abstention。
- V8: random interval matched sparsity。
- V9: full IGT-GCL。

## Current Notes

- No smoke, pilot, development, or formal result exists for IGT-GCL.
- Any future run must write raw JSON/CSV before summary tables.
- Negative and failed results must remain in the tracker.
