# World-model 非 Loss-only 假负样本 GCL 候选

生成时间：2026-06-26T11:27:25Z  
子流程：`/idea-discovery` second corrective rerun  
触发原因：SIVA-GCL 虽非 loss trick，但 novelty 仅 `6.7/10`，且 semantic critic 仍有 GraphMAE / GCA 拼装风险  
方向：图对比学习假负样本，面向普通节点分类，追求更强 2026 机制创新  
状态：candidate brief；未实现代码、未跑新实验、未产生性能 claim

## 为什么继续超越 SIVA

SIVA-positive-core 已将 BOND 的 loss-side 问题修正为 view-generation 机制，但 fresh reviewer 仍给出 `REVISE`、novelty `6.7/10`，最大 blocking weakness 是：

- masked-context semantic critic 可能退化为 GraphMAE / reconstruction；
- intervention-positive 可能被 GCA / RGCL / AutoGCL 解释；
- 需要证明 positive prealignment 被真正缓解。

因此本轮尝试把 critic 从 reconstruction 模型升级为 **latent world model**，让“语义稳定”不再由重构原始特征判定，而由抽象 latent target prediction 判定。这更接近 2025-2026 的 JEPA / world-model SSL 趋势，也更不像普通 augmentation。

## 新近边界

| Prior | 关键机制 | 对本轮约束 |
|---|---|---|
| Graph-JEPA / Predict, Cluster, Refine | context predicts target embeddings in latent space, avoids contrastive negatives and reconstruction | 不能只是套 JEPA；必须服务 GCL false-negative / view generation |
| SPGCL 2026 | message passing causes positive pre-alignment | 必须证明 generated positives 提供更强 positive signal |
| GraphMAE | masked feature reconstruction | semantic certificate 不能靠 raw feature reconstruction |
| CGC / BalanceGCL / ACGA | counterfactual / positive-negative graph generation | 不主打 generated hard negatives |
| DoG | graph diffusion synthetic structures | 不生成 synthetic nodes/edges 加入原图 |

## Top Candidate: WILLOW-GCL

**名称**：WILLOW-GCL / World-model Intervention Learning for Latent-node cOntrastive vieWs  
**推荐级别**：Top 1 for fresh review  
**贡献类型**：latent ego-world-model certified view generation；不是 loss trick，不是 GraphMAE reconstruction

### 方法直觉

普通 GCL 的 positive view 来自随机边/特征 dropout，但这些 views 经 message passing 后可能已经过度相似；真实节点 negatives 又会带来 false negatives。SIVA 用 reconstruction critic 约束 intervention positives，但 reconstruction critic 仍可能学 identity/degree shortcut。

WILLOW 的核心转向是：先学习一个 **ego latent world model**，让 context ego view 预测多个 masked target ego views 的 latent embeddings，而不是重构原始特征。这个 world model 形成一个语义证书：

```text
如果一个干预 view 仍可被 latent world model 预测为同一 target manifold，
它才是 semantic-preserving positive。
```

然后用这个证书搜索 **certified hard positives**：它们在输入/中间表示上尽可能远离原 view，但在 latent target manifold 上仍被 world model 认为属于同一 anchor semantics。

### 核心模块

#### 1. Ego Latent World Model

给 anchor `i`，采样 context ego-subgraph `c_i` 和多个 masked target ego-subgraphs `t_i^1...t_i^m`。

```text
h_c = E_online(c_i)
h_t^k = E_target(t_i^k)   # EMA target encoder
pred_t^k = P(h_c, pos_k)
L_world = sum_k || pred_t^k - stopgrad(h_t^k) ||_2^2
```

不重构 feature，不用 decoder，不用 test label。

#### 2. Certified Intervention View Search

对每个 anchor 搜索干预 `a`：

```text
maximize distance(E_online(view_i(a)), E_online(view_i(original)))
subject to world_error(i, a) <= epsilon
```

`world_error` 是 latent target prediction error，不是 feature reconstruction error。

#### 3. Contrastive Training with Certified Positives

用 certified intervention positive 替代普通 random positive；真实节点 negatives 降为 easy/random subset 或诊断项。

```text
positive pair = (original ego view, certified intervention view)
loss = existing GRACE/GCA-style objective or negative-light objective
```

贡献不在 loss，而在 **latent world-model certificate + hard-positive generation**。

### 与 SIVA 的差异

- SIVA critic 可能是 GraphMAE-like reconstruction；WILLOW 用 JEPA-like latent target prediction。
- SIVA 只是稳定性约束；WILLOW 形成 context-target world model，可生成多个 target certificate。
- SIVA 仍像 learned augmentation；WILLOW 的 positive 是由 latent prediction manifold 认证的 hard positive。

### 与 prior 的差异

- vs Graph-JEPA：Graph-JEPA 本身是非对比 SSL；WILLOW 用 world model 作为 view certification engine，服务 GCL positive generation 和 false-negative reduction。
- vs GraphMAE：不重构原始特征，不训练 decoder。
- vs SPGCL：不是调整 propagation / Dirichlet energy，而是生成 certified anti-prealignment positives。
- vs CGC/BalanceGCL/ACGA：不把 generated hard negatives 作为主贡献。
- vs DoG：不生成 synthetic graph structures 加入数据图。

### 为什么更有创新性

1. 主机制是 **latent world-model-certified view synthesis**，不是 loss / sampler。
2. 它直接绕开 false-negative hard negatives：先强化 positive signal，减少对 real hard negatives 的依赖。
3. 它用 2025-2026 趋势里的 JEPA / world model 语义证书，避免 GraphMAE reconstruction shortcut。

### 最小 smoke

- Cora seed=0。
- 对照：GRACE、SIVA-reconstruction critic、GraphMAE-only、Graph-JEPA-only、random hard positive、WILLOW-certified positive。
- 指标：world prediction error、view distance、post-GNN positive similarity、positive gradient norm、LogReg。
- 成功信号：WILLOW 生成的 positive 比 GRACE/SIVA 更非平凡；world_error 保持低；positive gradient 更 informative；Graph-JEPA-only 不能解释全部收益。

### Kill rules

- 若 Graph-JEPA-only 匹配 WILLOW，说明 view generation 无贡献，kill WILLOW。
- 若 SIVA-reconstruction critic 匹配 WILLOW，说明 latent world model 不必要，降回 SIVA。
- 若 random hard positive 匹配 WILLOW，说明 certificate 无效。
- 若 world_error 低但 LogReg/positive metrics 不好，说明 world model 学到非分类语义，pivot。

## Backup: SIVA-GCL-positive-core

若 WILLOW 被 reviewer 判定为 Graph-JEPA 包装，则保留 SIVA-positive-core 作为更轻量的 smoke 方向，但它不是最终“足够创新”的首选。

## Reviewer Questions

1. WILLOW 是否比 SIVA 更有创新性，还是只是 Graph-JEPA + augmentation？
2. latent world-model certificate 是否能形成可投稿主贡献？
3. 最小核心是 world model、certified intervention search，还是两者组合？
4. 它是否仍属于 GCL 主线，还是会漂移到非对比 Graph SSL？

