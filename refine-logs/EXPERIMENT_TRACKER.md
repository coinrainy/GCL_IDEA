# EXPERIMENT_TRACKER: DSR-GCL

生成时间：2026-06-26T09:24:55Z  
方法：DSR-GCL  
阶段：planned only  
当前 decision：**REVISE_IDEA**  
说明：当前没有运行任何实验。

| ID | Stage | Dataset | Split | Seeds | Variants | Status | Gate |
|---|---|---|---|---|---|---|---|
| DSR-A0 | smoke | Cora | `1:1:8` | 0 | V0,V2,V6,V9 | TODO | diagnostics runnable |
| DSR-B1 | pilot | Cora | `1:1:8` | 0,1,2 | V0,V2,V3,V4,V5,V6,V7,V8,V9 | TODO | firewall and branch ablation |
| DSR-B2 | pilot | CiteSeer | `1:1:8` | 0,1,2 | V0,V2,V3,V4,V5,V6,V7,V8,V9 | TODO | firewall and branch ablation |
| DSR-B3 | pilot | PubMed | `1:1:8` | 0,1,2 | V0,V2,V3,V4,V5,V6,V7,V8,V9 | TODO | firewall and branch ablation |
| DSR-C1 | pilot | Computers | `1:1:8` | 0,1,2 | V0,V2,V3,V6,V9 | TODO | transfer beyond Planetoid |
| DSR-C2 | pilot | Photo | `1:1:8` | 0,1,2 | V0,V2,V3,V6,V9 | TODO | transfer beyond Planetoid |
| DSR-D0 | review | TBD | frozen protocol | TBD | frozen subset | BLOCKED | only after pilot supports mechanism |

## Variant Legend

- V0: GRACE。
- V1: GCA, if adapted。
- V2: BGRL/CCA/Barlow-style negative-free baseline。
- V3: same-parameter single-head。
- V4: DSR semantic-only。
- V5: DSR residual-only。
- V6: DSR no firewall, semantic branch receives negatives。
- V7: DSR no orthogonality。
- V8: DSR random branch split。
- V9: full DSR-GCL。
- V10: CNG-GCL backup, not primary.

## Current Notes

- No smoke, pilot, development, or formal result exists.
- Tracker is a plan only.
- Formal remains blocked until reviewer and experiment-audit gates pass.
