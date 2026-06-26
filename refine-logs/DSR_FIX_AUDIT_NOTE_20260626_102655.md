# DSR-GCL Fix Audit Note

时间：2026-06-26T10:26:55Z  
范围：Cora seed=0 only，stratified random `1:1:8`，frozen encoder + Logistic Regression。  
状态：audit-smoke only；不跑 formal，不跑多数据集 pilot，不产生性能 claim。

## 为什么上一轮 PIVOT_REQUIRED 暂时降级

上一轮 audit-smoke 标记为 `PIVOT_REQUIRED`，但本轮发现三个实现/公式一致性问题，因此该结论暂时降级为：

```text
REVISE_IMPLEMENTATION_BEFORE_PIVOT
```

原因：

1. VICReg-normalization mismatch：`z_sem / z_res` 在进入 `vicreg_loss` 前已被 L2 normalize，导致 `variance_gamma=1.0` 对单位向量几乎不可满足，semantic negative-free 分支被错误约束。
2. Evaluation representation mismatch：GRACE baseline 评估 encoder output，但 DSR 主表评估 projector-level normalized `z_concat`，不公平。
3. Formula-code mismatch：机制规格中写了 stop-gradient alignment，但实现没有 predictor/target stop-gradient 结构；本轮改为 symmetric VICReg-style objective，不引入 predictor。

## 本轮代码修复

- DSRModel 现在显式输出：
  - encoder-level：`h_sem`, `h_res`, `h_concat`
  - raw projected：`p_sem`, `p_res`
  - normalized projected：`z_sem`, `z_res`, `z_concat`
- VICReg 的 alignment / variance / covariance 使用 raw projected `p_sem`。
- InfoNCE 接收 raw projected `p_res` / `p_sem`，并在 loss 内部 normalize。
- DSR 主评估默认改为 `h_concat`；projector-level `z_concat` 仅作为辅助诊断。
- 新增 `make_undirected_after_dropout` 开关，本轮开启；用于 DSR low-pass/residual 输入，缓解 edge dropout 后非对称图导致的谱低通解释不严格问题。
- `refine-logs/DSR_MECHANISM_SPEC.md` 已同步为 VICReg-style，无 predictor、无 stop-gradient alignment。

## Fix Audit-Smoke 结果

Artifacts:

- summary: `results/summary/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z_summary.md`
- raw results: `results/raw/Cora/DSR_GCL_SMOKE/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z/`
- logs: `logs/dsr_smoke/dsr_fix_audit_smoke_Cora_seed0_20260626T102655Z/`

主评估 (`test@best`, percent):

| ID | Variant | Main embedding | Test@best |
|---|---|---:|---:|
| A0 | GRACE baseline | `grace` | 84.78 |
| A2 | semantic-only fixed VICReg | `h_sem` | 68.68 |
| A3 | residual-only | `h_res` | 29.94 |
| A9 | fixed DSR-full | `h_concat` | 69.05 |
| A5b | budget-matched no-firewall | `h_concat` | 76.57 |
| A5c | scaled no-firewall | `h_concat` | 78.09 |
| A4b | param-matched single-head | `h_single` | 81.96 |

Embedding-level observations:

- A9 `h_concat=69.05`, but its auxiliary `z_concat=36.72`，说明 projector-level normalized eval 不应作为主口径。
- A2 `h_sem=68.68`, `z_sem=33.58`，再次确认 projector z 与 encoder h 表示差异很大。
- A3 `h_res=29.94`, `z_res=30.21`，residual-only 仍接近不可用。
- A5b/A5c 仍高于 A9，说明修复后 firewall 机制仍没有在该 smoke 上体现优势。

## 当前判断

本轮不恢复 formal，也不直接做最终 kill。阶段判断为：

```text
REVISE_IMPLEMENTATION_BEFORE_PIVOT
```

解释：上一轮 `PIVOT_REQUIRED` 不能作为最终判断，因为实现和公式不一致；但修复后的 Cora seed=0 audit-smoke 仍不支持进入 formal。若继续，只能先围绕 residual branch 与 low-pass/residual 构造做更小范围实现审计，不能扩大到多数据集 pilot。
