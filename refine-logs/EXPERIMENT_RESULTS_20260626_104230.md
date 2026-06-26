# EXPERIMENT_RESULTS: DSR-GCL Experiment Bridge

时间：2026-06-26T10:42:30Z  
计划：`refine-logs/EXPERIMENT_PLAN.md`  
当前真实状态源：`refine-logs/EXPERIMENT_TRACKER.md`  
阶段：experiment-bridge sanity/fix-audit collection only  
禁止：不跑 formal，不跑 10 seeds，不跑多数据集 pilot，不宣称性能提升。

## Bridge Summary

- Milestones parsed: sanity / smoke, audit-smoke, fix-audit-smoke, pilot, review。
- Must-run completed in current safe scope: 1/1 fix-audit-smoke sanity collection。
- Nice-to-have / blocked: Pilot-A、Pilot-B、formal、review。
- New GPU-hours in this bridge invocation: 0；复用已完成 Cora seed=0 fix-audit-smoke。
- Ready for `/auto-review-loop`: **NO**。

当前 tracker decision 为：

```text
REVISE_IMPLEMENTATION_BEFORE_PIVOT
```

原因：上一轮 `PIVOT_REQUIRED` 不能作为最终判断，因为发现 VICReg-normalization、evaluation representation、formula-code mismatch；但修复后的 Cora seed=0 audit-smoke 仍不支持 formal。

## Results by Milestone

### M0: Sanity / Fix-Audit-Smoke — DONE

Command represented by saved run:

```bash
python scripts/run_dsr_smoke.py --dataset Cora --seed 0 --run-tag dsr_fix_audit_smoke --status audit-smoke
```

Artifacts:

- Summary: `results/summary/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z_summary.md`
- Raw JSON: `results/raw/Cora/DSR_GCL_SMOKE/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z/`
- Logs: `logs/dsr_smoke/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z/`
- Fix audit note: `refine-logs/DSR_FIX_AUDIT_NOTE_20260626_102655.md`
- Code review: `refine-logs/EXPERIMENT_CODE_REVIEW_20260626_104230.md`

Primary metric: `test@best` percent under frozen encoder + Logistic Regression.

| Run | System | Main embedding | Key metric | Status |
|---|---|---:|---:|---|
| A0 | GRACE baseline | `grace` | 84.78 | DONE |
| A2 | semantic-only fixed VICReg | `h_sem` | 68.68 | DONE |
| A3 | residual-only | `h_res` | 29.94 | DONE |
| A9 | fixed DSR-full | `h_concat` | 69.05 | DONE |
| A5b | budget-matched no-firewall | `h_concat` | 76.57 | DONE |
| A5c | scaled no-firewall | `h_concat` | 78.09 | DONE |
| A4b | param-matched single-head | `h_single` | 81.96 | DONE |

### M1/M2/M3: Pilot / Main / Ablations — BLOCKED

Blocked by current gate:

- A3 residual-only remains near unusable (`29.94`)；
- A9 fixed DSR-full (`69.05`) remains far below A4b param-matched single-head (`81.96`)；
- A5b/A5c no-firewall controls remain above A9；
- Current evidence does not support full experiment bridge deployment, auto-review-loop, or formal claims.

## Success Criteria Check

| Criterion | Status | Note |
|---|---|---|
| Training loop runs without errors | PASS | Saved raw JSON/logs for all variants |
| Metrics are computed and saved | PASS | Summary JSON/MD exists |
| Evaluation uses ground truth | PASS | Logistic Regression fitted/evaluated against dataset labels |
| Firewall diagnostic recorded | PASS | A9 leakage `0.0`; no-firewall controls non-zero |
| DSR mechanism advantage | FAIL | A9 below controls |
| Ready for broader pilot | NO | Requires new residual/low-pass plan first |

## Next Step

Do **not** run Pilot-A/Pilot-B from the old plan.  
Next safe step is a new, narrower implementation-refinement plan focused on:

- residual branch construction；
- low-pass/residual operator correctness；
- whether the firewall claim should be abandoned or reformulated。
