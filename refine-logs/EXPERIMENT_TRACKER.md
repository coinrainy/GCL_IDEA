# EXPERIMENT_TRACKER: DSR-GCL

生成时间：2026-06-26T09:40:55Z  
方法：DSR-GCL  
阶段：audit-smoke completed for Cora seed 0 only  
当前 decision：**PIVOT_REQUIRED**  
说明：已有一次最小 smoke 和一次 audit-smoke；无 pilot/formal 结果，无性能 claim。

| ID | Stage | Dataset | Split | Seeds | Variants | Status | Gate |
|---|---|---|---|---|---|---|---|
| DSR-A0 | smoke | Cora | `1:1:8` | 0 | A0,A2,A3,A4,A5,A9 | DONE_REVISE | no NaN, leakage diagnostic recorded, but DSR-full lacks mechanism advantage |
| DSR-A1 | audit-smoke | Cora | `1:1:8` | 0 | A0,A2,A3,A4,A4b,A5,A5a,A5b,A5c,A9 | DONE_PIVOT_REQUIRED | residual branch ineffective; fair no-firewall and param-matched single-head beat A9 |
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

- Smoke result exists for Cora seed=0 only:
  - summary: `results/summary/dsr_smoke_Cora_seed0_20260626T095525Z_summary.md`
  - raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_smoke_Cora_seed0_20260626T095525Z/`
  - logs: `logs/dsr_smoke/dsr_smoke_Cora_seed0_20260626T095525Z/`
- A9 / DSR-full firewall diagnostic passed mechanically (`negative_gradient_leakage=0.0`), while A5 / no-firewall showed semantic negative-gradient leakage (`1.2867` at final logged epoch).
- Mechanism advantage was not supported in this smoke: A9 concat `77.08` test@best was below A2 semantic-only `78.27`, A5 no-firewall `81.13`, A4 same-parameter single-head `83.63`, and A0 GRACE `84.78`.
- This triggers `REVISE/PIVOT_REQUIRED` at smoke level. Do not proceed to formal experiments or performance claims.
- Audit-smoke result exists for Cora seed=0 only:
  - audit note: `refine-logs/DSR_AUDIT_SMOKE_NOTE_20260626_101103.md`
  - summary: `results/summary/dsr_audit_smoke_Cora_seed0_20260626T101103Z_summary.md`
  - raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`
  - logs: `logs/dsr_smoke/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`
- Audit-smoke added fairer no-firewall variants A5a/A5b/A5c and A4b parameter-matched single-head control. A9 DSR-full `78.69` was below A4 `83.63`, A4b `80.81`, A5b `79.57`, A5c `81.00`, and A0 GRACE `84.78`; A3 residual-only remained extremely low at `30.21`.
- Final audit-smoke decision: `PIVOT_REQUIRED`. Do not continue DSR-GCL formal under the current residual/firewall mechanism.
- Formal remains blocked until reviewer and experiment-audit gates pass.
