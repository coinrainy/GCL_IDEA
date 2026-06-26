# DSR-GCL Audit-Smoke 审计说明

时间：2026-06-26T10:11:03Z  
范围：Cora seed=0 only，stratified random `1:1:8`，frozen encoder + Logistic Regression。  
状态：audit-smoke only；非 formal，不支持性能 claim。

## 上轮 Smoke 的主要问题

1. A9 DSR-full 结果差：上一轮 A9 concat `77.08` 低于 A2 semantic-only、A5 no-firewall、A4 same-head 和 A0 GRACE，本轮 audit-smoke 中 A9 为 `78.69`，仍低于 A4/A4b、A5b/A5c 和 GRACE。
2. A3 residual-only 几乎无效：本轮 A3 `z_res` test@best 为 `30.21`，接近不可用，说明当前 residual branch 没有形成节点分类可用表示。
3. 旧 A5 no-firewall 不公平：旧 A5 比 A9 多了 semantic InfoNCE，contrastive loss budget 更大；本轮新增 A5b 和 A5c 后，公平化/缩放后的 no-firewall 仍高于 A9。
4. A4 same-head 参数量不严格匹配：A4 为 `433024` 参数，A9 为 `400256` 参数；本轮新增 A4b，参数量 `402120`，与 A9 的相对差距约 `0.47%`。
5. 旧 handoff summary 不完整：没有完整展示 `z_sem / z_res / concat / params`。本轮 summary 已补齐 embedding-level 表、参数量表、leakage 表、raw JSON 路径和 log 路径。

## 本轮关键结果

| ID | Variant | Main embedding | Test@best | Params |
|---|---|---:|---:|---:|
| A0 | GRACE baseline | `grace` | 84.78 | 433024 |
| A2 | semantic-only negative-free | `z_sem` | 78.27 | 200128 |
| A3 | residual-only InfoNCE | `z_res` | 30.21 | 200128 |
| A9 | DSR-full firewall | `concat` | 78.69 | 400256 |
| A5b | budget-matched no-firewall | `concat` | 79.57 | 400256 |
| A5c | scaled no-firewall | `concat` | 81.00 | 400256 |
| A4 | same-head original | `single` | 83.63 | 433024 |
| A4b | param-matched single-head | `single` | 80.81 | 402120 |

## Kill Rule 判断

触发以下规则：

- A3 residual-only 仍然极低，说明 residual branch 当前不可用。
- A9 低于 A4/A4b，说明双分支/firewall 没有超过普通 mixed-loss/single-head control。
- A5b/A5c 高于 A9，说明当前 firewall claim 不成立。

最终判断：**PIVOT_REQUIRED**。不建议继续推进 DSR-GCL formal；应先重构 residual branch 或放弃当前 firewall 叙事。
