# DSR-GCL Smoke Handoff

时间：2026-06-26T10:02:00Z  
范围：Cora seed=0 only，stratified random `1:1:8`，frozen encoder + Logistic Regression evaluator。  
阶段：smoke / pilot diagnostic only；无 formal 实验，无性能 claim。

## 当前 Idea 核心 Claim

DSR-GCL 的核心假设不是识别或过滤 false negatives，而是限制 InfoNCE negative repulsion 的作用通道：

```text
negative repulsion may shape residual/boundary representation,
but must not update semantic representation.
```

当前实现对应 parameter-isolated two-channel firewall：

- semantic channel：negative-free alignment + variance/covariance collapse guard。
- residual channel：InfoNCE negative repulsion。
- DSR-full 默认评估：`concat(z_sem, stopgrad(z_res))`。
- firewall 诊断：`||grad(theta_sem; L_neg)|| / (||grad(theta_res; L_neg)|| + eps)`。

## 必须验证的问题

- C1：negative loss 是否真的不更新 semantic 参数。
- C2：semantic branch 是否不只是一个普通 negative-free baseline。
- C3：residual branch 是否提供有用的 uniformity / boundary 信息，而不是退化噪声。
- C4：效果是否不是参数量或 single-head 训练目标造成的。
- C5：低通 / residual split 是否真的有意义。

## 本轮变体

- A0：GRACE baseline。
- A2：semantic-only negative-free branch。
- A3：residual-only InfoNCE branch。
- A9：DSR-full，semantic negative-free + residual InfoNCE + orthogonality。
- A5：DSR-no-firewall，semantic branch 也接收 negative loss。
- A4：same-parameter single-head control。

## Smoke Result Table

| ID | Variant | Main embedding | Valid@best | Test@best | Final test |
|---|---|---:|---:|---:|---:|
| A0 | GRACE baseline | `grace` | 85.19 | 84.78 | 84.78 |
| A2 | semantic-only negative-free | `z_sem` | 77.78 | 78.27 | 78.27 |
| A3 | residual-only InfoNCE | `z_res` | 31.11 | 30.12 | 30.12 |
| A9 | DSR-full firewall | `concat` | 75.93 | 77.08 | 77.08 |
| A5 | DSR-no-firewall | `concat` | 78.89 | 81.13 | 81.13 |
| A4 | same-parameter single-head | `single` | 83.70 | 83.63 | 83.63 |

## Mechanism Diagnostics

| ID | Leakage | Firewall pass | Rank sem | Rank res | Rank concat | Branch corr |
|---|---:|---|---:|---:|---:|---:|
| A9 | 0.0 | true | 6.83 | 7.22 | 13.36 | 0.1168 |
| A5 | 1.2867 | false | 9.95 | 4.15 | 13.66 | 0.0659 |

## Kill Rule Readout

- C1 firewall diagnostic: mechanically passes for A9 and fails for A5, so the diagnostic is useful.
- C2/C3: not supported in this smoke; A9 full is below semantic-only A2.
- C4: not supported in this smoke; same-head A4 is much stronger than A9.
- Current conclusion: `REVISE/PIVOT_REQUIRED` at smoke level. Do not run formal experiments or write performance claims from this result.

## Key Paths

- Config: `configs/dsr_smoke.yaml`
- Runner: `scripts/run_dsr_smoke.py`
- Raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_smoke_Cora_seed0_20260626T095525Z/`
- Logs: `logs/dsr_smoke/dsr_smoke_Cora_seed0_20260626T095525Z/`
- Summary: `results/summary/dsr_smoke_Cora_seed0_20260626T095525Z_summary.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Wiki idea page: `research-wiki/ideas/dsr_gcl.md`
- Wiki log: `research-wiki/log.md`
