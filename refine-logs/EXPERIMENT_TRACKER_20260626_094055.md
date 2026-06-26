# EXPERIMENT_TRACKER: DSR-GCL

生成时间：2026-06-26T09:40:55Z  
方法：DSR-GCL  
阶段：planned only  
当前 decision：**REVISE_IDEA**  
说明：当前没有运行任何实验。

| ID | Stage | Dataset | Split | Seeds | Variants | Status | Gate |
|---|---|---|---|---|---|---|---|
| DSR-A0 | smoke | Cora | `1:1:8` | 0 | A0,A2,A5,A9 | TODO | no NaN, leakage diagnostic, non-collapse |
| DSR-B1 | pilot | Cora | `1:1:8` | 0,1,2 | A0,A1,A2,A3,A4,A5,A6,A7,A9,A11 | TODO | mechanism screening |
| DSR-B2 | pilot | CiteSeer | `1:1:8` | 0,1,2 | A0,A1,A2,A3,A4,A5,A6,A7,A9,A11 | TODO | mechanism screening |
| DSR-B3 | pilot | PubMed | `1:1:8` | 0,1,2 | A0,A1,A2,A3,A4,A5,A6,A7,A9,A11 | TODO | mechanism screening |
| DSR-C1 | pilot | Computers | `1:1:8` | 0,1,2 | A0,A2,A4,A5,A9,A11 | TODO | transfer beyond Planetoid |
| DSR-C2 | pilot | Photo | `1:1:8` | 0,1,2 | A0,A2,A4,A5,A9,A11 | TODO | transfer beyond Planetoid |
| DSR-D0 | review | TBD | frozen protocol | TBD | frozen subset | BLOCKED | only after pilot supports C1-C5 |

## Variant Legend

- A0: GRACE。
- A1: GCA if available。
- A2: Negative-free `L_sem` only。
- A3: Residual-only `L_res`。
- A4: Same-parameter single-head。
- A5: No firewall, semantic branch receives negatives。
- A6: No orthogonality。
- A7: Random branch split。
- A8: Shared trunk diagnostic。
- A9: Full DSR-GCL。
- A10: Spectral fusion control。
- A11: Larger projector control。

## Current Notes

- No smoke, pilot, development, or formal result exists.
- Tracker is a plan only.
- Formal remains blocked until reviewer and experiment-audit gates pass.
