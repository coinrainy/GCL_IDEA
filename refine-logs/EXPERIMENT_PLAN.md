# EXPERIMENT_PLAN: DSR-GCL

生成时间：2026-06-26T09:24:55Z  
方法：DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning  
核心机制：Spectral Gradient Firewall  
阶段：plan only  
当前 decision：**REVISE_IDEA**  
禁止：不实现代码，不跑 formal，不产生性能 claim。

## 1. Goal

本计划只用于未来 smoke/pilot 设计，目标不是证明性能，而是证伪 DSR-GCL 的核心机制：

> negative repulsion 是否能被限制在 residual channel，从而避免破坏 semantic channel，同时 residual channel 是否真的补充了 pure negative-free baseline 缺少的 uniformity / boundary signal？

## 2. Protocol

- 同配图：stratified random `1:1:8` split。
- 异配图：官方或公认 fixed split。
- GCL evaluation：frozen encoder + Logistic Regression。
- evaluator 超参只能由 train/val 或 train-only CV 决定。
- pilot seeds 可从 `0,1,2` 开始；formal 若未来批准才用 `0-9`。
- 所有结果必须标注 smoke / pilot / development / formal。
- 本阶段没有任何结果可支持 SOTA、robust、comprehensive claim。

## 3. Claim-to-Test Matrix

| ID | Mechanism claim | Required test | Kill / revise condition |
|---|---|---|---|
| C1 | negative gradient 不进入 semantic channel | `||Proj_sem(g_neg)||` 或参数隔离检查接近 0 | leakage 非 0 或无法记录 -> REVISE |
| C2 | semantic branch 不只是 BGRL/CCA | DSR > pure negative-free in diagnostics or pilot | pure negative-free 匹配 -> PIVOT_REQUIRED |
| C3 | residual branch 提供 uniformity / boundary | full DSR > no-residual / semantic-only；`z_res` 有 non-collapse rank | no-residual 匹配 -> REVISE |
| C4 | 不是参数量/双 head 效应 | same-parameter single-head control | single-head 匹配 -> KILL/PIVOT |
| C5 | spectral/residual split 有意义 | random branch split and no-orthogonality controls | random split 匹配 -> REVISE |
| C6 | 不只适合同配图 | Computers/Photo and optional heterophily fixed split | 明显退化 -> restrict or pivot |

## 4. Datasets

| Stage | Dataset | Split | Seeds | Purpose |
|---|---|---|---|---|
| Smoke | Cora | `1:1:8` | 0 | 检查双通道、loss、diagnostics 能跑 |
| Pilot-A | Cora, CiteSeer, PubMed | `1:1:8` | 0,1,2 | Planetoid 机制筛查 |
| Pilot-B | Amazon-Computers, Amazon-Photo | `1:1:8` | 0,1,2 | 迁移到更大属性图 |
| Optional | heterophily benchmark | official fixed | official seeds | 检查 high-frequency 分类信息风险 |

## 5. Variants

| Variant | Description | Required |
|---|---|---|
| V0 GRACE | existing contrastive baseline | yes |
| V1 GCA | augmentation-aware baseline if adapted | preferred |
| V2 BGRL/CCA/Barlow-style | negative-free baseline | yes |
| V3 same-param single-head | same total hidden/projector params | yes |
| V4 DSR semantic-only | `L_sem` only | yes |
| V5 DSR residual-only | `L_res` only | yes |
| V6 DSR no firewall | semantic branch also receives negatives | yes |
| V7 DSR no orthogonality | remove `L_orth` | yes |
| V8 DSR random split | random channel split instead of low/residual split | yes |
| V9 DSR full | semantic + residual + firewall + orthogonality | yes |
| V10 CNG-GCL | backup idea, only if DSR stalls | optional |

## 6. Diagnostics

必须保存 raw JSON/CSV：

- `leakage_sem_negative_grad`: negative gradient projection into semantic channel。
- `rank_sem`, `rank_res`, `rank_concat`: effective rank。
- `variance_sem`, `covariance_sem`: collapse guard health。
- `alignment_sem`, `uniformity_res`, `uniformity_concat`。
- `logreg_acc_sem`, `logreg_acc_res`, `logreg_acc_concat` under frozen evaluator。
- class-wise and degree-bucket diagnostics。
- parameter count, runtime, memory。

标签只用于 evaluator 和 offline diagnostics，不能参与 self-supervised training 或 test-time selection。

## 7. Run Order

### Stage A: Design Smoke

Dataset: Cora seed 0  
Variants: V0, V2, V6, V9  
Pass criteria:

- no NaN；
- `z_sem`, `z_res` non-collapse；
- leakage diagnostic produced；
- Logistic Regression evaluator works；
- raw result JSON saved。

Failure:

- leakage cannot be measured -> `REVISE_IDEA`；
- semantic collapse -> `PIVOT_REQUIRED`；
- code cannot run -> `BLOCKED`。

### Stage B: Mechanism Pilot

Datasets: Cora, CiteSeer, PubMed  
Seeds: 0,1,2  
Variants: V0, V2, V3, V4, V5, V6, V7, V8, V9

Decision:

- V9 not better than V2/V3 in diagnostics -> `PIVOT_REQUIRED`。
- V6 matches V9 -> firewall not needed -> `KILL` or `PIVOT_REQUIRED`。
- V8 matches V9 -> spectral/residual split not needed -> `REVISE_IDEA`。

### Stage C: Transfer Pilot

Datasets: Computers, Photo  
Seeds: 0,1,2  
Variants: V0, V2, V3, V6, V9

Decision:

- consistent degradation -> restrict claim or pivot。
- positive diagnostic but weak accuracy -> continue refinement, not formal。
- strong diagnostic across datasets -> ask for experiment-audit before experiment bridge。

## 8. Formal Gate

`GO_TO_EXPERIMENT_BRIDGE` requires:

- final DSR architecture frozen；
- leak diagnostic works；
- pilot supports C1-C5；
- negative/failed runs preserved；
- reviewer re-checks the revised proposal；
- experiment-audit approves split/evaluator/config protocol。

Current state does **not** satisfy this gate.

## 9. Current Decision

**REVISE_IDEA**。

DSR-GCL is more promising than IGT-GCL, but it still needs mechanism refinement and smoke/pilot design before experiment bridge.
