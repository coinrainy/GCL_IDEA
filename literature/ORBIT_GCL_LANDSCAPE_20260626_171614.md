# ORBIT-GCL Literature Boundary and Gap Map

- 时间：2026-06-26T17:16:14Z
- 方向：ordinary node-classification GCL after MEGA no-go
- 状态：proposal-level literature/context synthesis
- Claim status：未实现、未跑 ORBIT smoke/pilot/formal；不支持性能 claim。

## 背景结论

本轮重新构思不再继续 false-negative pair mining、positive/neutral closure、rank/boundary diagnostic、train-label probe routing、masked evidence group assignment。项目内已有证据显示这些代理目标在 Cora seed0 或 partial pilot 下没有稳定转化为 frozen encoder + Logistic Regression 的节点分类 accuracy。

最新吸收的关键负结果是 MEGA-GCL：MEGA full 在 Cora seed0 kill-smoke 中为 `67.44`，明显低于 GRACE `84.69`、GCMAE-style `84.36` 与 matched-parameter GRACE `82.52`，并触发 `KILL_MEGA_AS_WEAKER_THAN_GRACE`、`KILL_MEGA_AS_MASKED_AUTOENCODER_VARIANT`、`KILL_MASKING_STORY`。因此“masked evidence group prediction”不足以支撑主线。

新的 gap 应从如下角度定义：

> 现有 GCL / Graph SSL 大多学习 node-node agreement、masked input/latent reconstruction、augmentation invariance 或 pseudo-task stacking；但少有方法把节点表征定义为其在一组无标签图干预算子下的响应函数，并把这个 response field 作为自监督目标。

## 近邻边界

### Node-node contrastive learning

GRACE 通过结构和属性 corruption 生成两视图，并最大化同一节点在两视图中的 agreement。它是本项目 GCL backbone，但 node-node denominator 在少标签节点分类下带来 false-negative 和 objective mismatch 风险。

参考：Deep Graph Contrastive Representation Learning, arXiv:2006.04131, https://arxiv.org/abs/2006.04131

### Masked graph autoencoding

GraphMAE 将重点从结构重建转到 masked feature reconstruction，并使用 scaled cosine error；GraphMAE2 进一步用 remask decoding 与 latent prediction 正则化重建目标。这一族吃掉了“mask 输入后重建/预测 latent”的大部分空间。

参考：GraphMAE, https://arxiv.org/abs/2205.10803；GraphMAE2, https://arxiv.org/abs/2304.04779

### Positional / structural reconstruction

GraphPAE 指出传统 feature/edge masking 主要捕获低频信号，并加入位置路径与相对距离重建；StructPosGSSL 将结构和位置编码引入 SSL。这意味着 ORBIT 不能只说“加入结构/位置目标”，必须强调 operator response basis。

参考：Graph Positional Autoencoders, https://arxiv.org/html/2505.23345v2；StructPosGSSL, https://arxiv.org/html/2502.16233v1

### Causal / spectral invariance

GCIL 从 SCM 角度分析 GCL，不变性目标和 independence objective 试图减少非因果因素影响；AS-GCL 用 spectral augmentation 和 asymmetric encoders 构造视图。这意味着 ORBIT 不能只是“谱干预 + invariance loss”。

参考：GCIL, https://arxiv.org/html/2401.12564；AS-GCL, https://arxiv.org/html/2502.13525v1

### JEPA / predictive graph SSL

PCR / Graph-JEPA 类方法使用 context-target latent prediction，说明“预测 latent target”不是新颖点。ORBIT 的差异必须是：目标不是某个 masked target latent，而是节点对多个干预算子的响应函数低秩基。

参考：Predict, Cluster, Refine, https://arxiv.org/html/2502.01684v4

## 新机会

ORBIT-GCL 的潜在新问题定义：

> 对节点分类有用的表征，不一定来自节点间相似性或输入重建，而可能来自节点在不同无标签图干预下的稳定/敏感响应模式。类别边界往往反映“哪些证据被扰动时节点语义会改变”，因此 response field 可能比 raw feature、degree、diffusion score 或 masked evidence group 更接近 class-readable evidence。

这一路线的价值不在单个 loss，而在一个完整论文骨架：

1. Problem：node-node / reconstruction SSL 与少标签节点分类的 objective mismatch。
2. Method：operator response field induction。
3. Theory/analysis：response basis 与 class-readable perturbation sensitivity 的联系。
4. Experiments：与 GRACE、GraphMAE2、GCMAE、Graph-JEPA、TTER/D-SLA style controls、IRIS/WILLOW response controls、operator-id shortcut controls 对照。
5. Diagnostics：held-out operator response prediction、degree/feature shortcut partial controls、response-label alignment without test-label selection。

## 结论

`ORBIT-GCL` 是当前最值得保留的方向，但必须标为 `REVISE`。它不能直接进入多数据集 pilot，只允许先做 Cora seed0 kill-smoke。若 ORBIT full 不超过最强 matched controls，或 random/shuffled/degree-only response 接近 full，应立即 kill。
