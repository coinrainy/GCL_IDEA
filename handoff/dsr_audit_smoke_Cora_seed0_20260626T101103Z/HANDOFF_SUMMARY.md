# DSR-GCL Audit-Smoke Handoff

时间：2026-06-26T10:11:03Z  
范围：Cora seed=0 only，stratified random `1:1:8`，frozen encoder + Logistic Regression evaluator。  
阶段：audit-smoke only；未跑 10 seeds，未进入 formal，未产生性能 claim。

## 本轮目标

本轮不是提高 A9 结果，而是审计上轮 smoke 的失败原因并修正不公平 ablation：

- A9 DSR-full 结果差；
- A3 residual-only 几乎无效；
- 原 A5 no-firewall 比 A9 多 semantic InfoNCE，contrastive budget 不公平；
- 原 A4 same-head 参数量不严格匹配；
- 旧 handoff summary 未完整展示 embedding-level 和参数量信息。

## 新增 / 修正

- `A5a_semantic_infonce_only`：semantic branch 使用 InfoNCE，不使用 residual InfoNCE。
- `A5b_budget_matched_no_firewall`：semantic InfoNCE 与 residual InfoNCE 各 `0.5`。
- `A5c_scaled_no_firewall`：在 A5 基础上缩放总 loss，使 loss scale 尽量接近 A9。
- `A4b_param_matched_single_head`：参数量最接近 A9 的 single-head control，`402120` vs A9 `400256`。

## 关键结果

| ID | Variant | Main embedding | Test@best | Params |
|---|---|---:|---:|---:|
| A0 | GRACE baseline | `grace` | 84.78 | 433024 |
| A2 | semantic-only negative-free | `z_sem` | 78.27 | 200128 |
| A3 | residual-only InfoNCE | `z_res` | 30.21 | 200128 |
| A9 | DSR-full firewall | `concat` | 78.69 | 400256 |
| A5 | original no-firewall | `concat` | 80.95 | 400256 |
| A5a | semantic InfoNCE only | `z_sem` | 79.43 | 200128 |
| A5b | budget-matched no-firewall | `concat` | 79.57 | 400256 |
| A5c | scaled no-firewall | `concat` | 81.00 | 400256 |
| A4 | original same-head | `single` | 83.63 | 433024 |
| A4b | param-matched single-head | `single` | 80.81 | 402120 |

## 机制诊断

- A9 leakage：`0.0`，firewall 机械诊断仍通过。
- A5 leakage：`2.4781`，no-firewall 确实让 negative gradient 进入 semantic branch。
- A5b leakage：`11648.9246`，budget-matched no-firewall 仍明显泄漏且高于 A9。
- A5c leakage：`11247.6425`，scaled no-firewall 仍明显泄漏且高于 A9。
- A3 residual-only `30.21`，说明当前 residual branch 对节点分类几乎不可用。

## Kill Rule 判断

触发规则：

- residual-only 仍然接近随机或极低；
- A9 低于 A4/A4b；
- A5b/A5c 高于 A9。

最终判断：**PIVOT_REQUIRED**。不建议继续推进当前 DSR-GCL formal；应先重构 residual branch 或放弃当前 firewall claim。

## 包含文件

- `configs/dsr_smoke.yaml`
- `scripts/run_dsr_smoke.py`
- `refine-logs/DSR_AUDIT_SMOKE_NOTE_20260626_101103.md`
- `results/raw/Cora/DSR_GCL_SMOKE/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`
- `logs/dsr_smoke/dsr_audit_smoke_Cora_seed0_20260626T101103Z/`
- `results/summary/dsr_audit_smoke_Cora_seed0_20260626T101103Z_summary.md`
- `results/summary/dsr_audit_smoke_Cora_seed0_20260626T101103Z_summary.json`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `research-wiki/log.md`
- `research-wiki/ideas/dsr_gcl.md`
- `MANIFEST.md`
- `AGENTS.md`

## Windows PowerShell 下载命令

```powershell
scp root@<服务器IP>:/root/autodl-tmp/GCL_IDEA/handoff/dsr_audit_smoke_Cora_seed0_20260626T101103Z.tar.gz .\
```
