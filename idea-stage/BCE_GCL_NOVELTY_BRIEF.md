# BCE-GCL Novelty Reviewer Brief

生成时间：2026-06-26T08:34:42Z

用途：供 fresh `gcl_scientific_reviewer` 对 BCE-GCL 做独立 novelty / significance 审查。

注意：本文件只整理待审 idea、项目约束、必须核查的公开近邻工作与输出格式，不给 reviewer 预设结论。审查目标是判断该 idea 是否值得进入 `/research-review` 与 `/research-refine-pipeline`，不是证明它可发表。

## 待审 Idea

名称：BCE-GCL — Boundary-Conditioned Contrastive Eligibility for Graph Contrastive Learning

任务范围：图对比学习用于节点分类。

项目默认协议：

- homophilous graph：stratified random `1:1:8` split，即 train / validation / test = 10% / 10% / 80%。
- 默认 seeds：0-9。
- GCL evaluator：frozen encoder + Logistic Regression。
- evaluator 超参只能由 train/val 或 train 内部 CV 决定，禁止 test set 调参。
- 当前阶段无代码实现、无 pilot、无 performance claim。

核心方法草案：

1. 估计每个节点的 boundary / eligibility score，候选信号包括 feature-structure disagreement、augmentation instability、local semantic conflict。
2. 将节点路由为三类：core nodes、boundary nodes、unstable nodes。
3. core nodes 使用标准 InfoNCE；boundary nodes 使用弱 boundary-preserving consistency；unstable nodes 下调权重或只做 stability regularization。
4. 预训练结束后 frozen encoder，用 Logistic Regression 做节点分类评估。

核心假设：

- 标准 GCL 默认所有节点都适合强 cross-view alignment。
- core nodes 可能受益于强 contrastive alignment。
- boundary / unstable nodes 可能因增强视图语义漂移或结构-特征冲突，被强行拉齐后损害节点分类边界。

## 需要审查的 Core Claims

1. “node-wise contrastive eligibility” 是否是清晰且相对新的问题定义，而不是 hard-node weighting、uncertainty weighting、degree weighting、adaptive augmentation 的换名。
2. “core / boundary / unstable objective routing” 是否和已有 GCL loss weighting、false-negative filtering、prototype / uncertainty / spectral gating 有实质差异。
3. 使用 feature-structure disagreement + augmentation instability 估计 boundary / eligibility，是否已被近期 boundary-aware GNN/GCL 或 semantic drift GCL 覆盖。
4. 该 idea 的最小可发表贡献是否应从 “method” 降级为 “diagnostic + lightweight routing”，或应直接 PIVOT/KILL。

## 本次已检索到的近邻公开工作

### Boundary / boundary-node 相关

1. Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement
   - URL: https://arxiv.org/html/2606.20283
   - arXiv: 2606.20283v1
   - 日期：2026-06-18
   - 相关点：该文明确以 nodes near class boundaries 为主要瓶颈，提出 Boundary Embedding Shaping (BES)，作为 adaptive contrastive learning GNN plug-in，选择性抑制 decision boundaries 上的 spurious structural noise。
   - 待核查差异：BES 更像 supervised / end-to-end GNN plug-in，BCE-GCL 目标是无监督 GCL 预训练 objective routing；但 “boundary + adaptive contrastive learning” 的命名和叙事风险很高。

2. Leveraging Label Non-Uniformity for Node Classification in Graph Neural Networks
   - URL: https://proceedings.mlr.press/v202/ji23a.html
   - 相关点：用 label non-uniformity 衡量节点是否接近 class boundary，提出 boundary/hard-to-classify node 视角。
   - 待核查差异：不是 GCL 预训练方法，但 boundary score / hard node framing 相关。

### Node-wise / uncertainty / stability / spectral routing 相关

3. ASPECT: Node-Level Adaptive Spectral Fusion for Graph Contrastive Learning
   - URL: https://arxiv.org/html/2604.01878v2
   - arXiv: 2604.01878v2
   - 日期：2026-05-06
   - 相关点：提出 node-wise spectral policy，并有 ASPECT-S stability-aware extension；使用 perturbation estimates 与 channel-wise sensitivity。
   - 待核查差异：ASPECT 路由的是 spectral channels / frequency mixture，而 BCE-GCL 路由的是 contrastive objective strength/type；但 node-wise policy + stability-aware GCL 是强近邻。

4. Uncertainty-guided Graph Contrastive Learning from a Unified Perspective
   - URL: https://www.ijcai.org/proceedings/2025/629
   - 会议：IJCAI 2025
   - 相关点：引入 sample uncertainty 来统一指导 data augmentation 与 weighted graph contrastive loss；关注 class ambiguity within individual samples。
   - 待核查差异：UGCL 已经把 sample uncertainty、augmentation、weighted GCL loss 绑定，BCE-GCL 必须证明 boundary eligibility 不等同于 uncertainty。

5. Uncertainty in Graph Contrastive Learning with Bayesian Neural Networks
   - URL: https://arxiv.org/abs/2312.00232
   - 日期：2023-11-30
   - 相关点：提出 contrastive learning uncertainty measure，基于 different positive samples 的 likelihood disagreement，用于半监督节点分类。
   - 待核查差异：Bayesian uncertainty vs deterministic boundary/eligibility routing。

### Hard-to-learn / degree / false-negative 相关

6. Mitigating Degree Bias Adaptively with Hard-to-Learn Nodes in Graph Contrastive Learning
   - URL: https://arxiv.org/abs/2506.05214
   - 日期：2025-06-05
   - 相关点：提出 HAR contrastive loss 与 SHARP framework，针对 hard-to-learn nodes / degree bias，adaptive reweight positives and negatives。
   - 待核查差异：HAR/SHARP 以 degree bias 和 hard-to-learn nodes 为主，BCE-GCL 以 boundary eligibility / objective routing 为主；但 node-level reweighting 风险高。

7. Uncovering the Structural Fairness in Graph Contrastive Learning
   - URL: https://openreview.net/forum?id=RJemsN3V_kt
   - 会议：NeurIPS 2022
   - 相关点：分析 GCL 在 degree bias / community boundary 上的行为，提出 GRADE，对 low/high-degree nodes 应用不同增强策略。
   - 待核查差异：degree-conditioned augmentation vs boundary-conditioned objective routing。

8. ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning
   - URL: https://proceedings.mlr.press/v162/xia22b.html
   - 会议：ICML 2022
   - 相关点：区分 hard negatives 与 false negatives，估计 negative 为 true negative 的概率并加权。
   - 待核查差异：pair-level negative reliability vs node-level contrastive eligibility。

9. Enhancing Graph Contrastive Learning with Node Similarity
   - URL: https://arxiv.org/abs/2208.06743
   - 日期：2022-08-13
   - 相关点：基于 node similarity 建模 positive / negative sampling distribution，处理 false negatives 和 positive scarcity。
   - 待核查差异：pair sampling distribution vs node-wise objective routing。

### Semantic drift / adaptive augmentation 相关

10. Topology Reorganized Graph Contrastive Learning with Mitigating Semantic Drift
    - URL: https://arxiv.org/abs/2407.16726
    - 日期：2024-07-23
    - 相关点：指出 blind local augmentations 可能让 augmented data traverse to other classes，并用 prototype-based negative pair selection 减轻 semantic drift。
    - 待核查差异：view construction / false-negative filtering vs boundary-specific weak alignment。

11. Graph Contrastive Learning with Adaptive Augmentation
    - URL: https://arxiv.org/abs/2010.14945
    - 会议：WWW 2021
    - 相关点：GCA 针对拓扑和属性重要性设计 adaptive augmentation，避免 uniform augmentation 破坏重要结构或语义。
    - 待核查差异：importance-preserving augmentation vs post-augmentation objective eligibility。

12. SOLA-GCL: Subgraph-Oriented Learnable Augmentation Method for Graph Contrastive Learning
    - URL: https://arxiv.org/html/2503.10100v1
    - 会议：AAAI 2025
    - 相关点：learnable augmentation / subgraph-oriented view generator。
    - 待核查差异：learnable view generator vs lightweight node eligibility routing。

## 请求 Reviewer 输出格式

请输出中文报告，包含：

1. Verdict：`GO` / `REVISE` / `PIVOT` / `KILL` / `BLOCKED`。
2. Novelty score：0-10。
3. 最接近的 5 篇 prior work，逐一说明 overlap 与 remaining delta。
4. 对 4 个 core claims 分别判定：HIGH / MEDIUM / LOW novelty。
5. 若 verdict 不是 KILL：给出最小可保留版本，包括必须改名/改叙事的点、必须做的 ablation、必须避免的 claim。
6. 若 verdict 是 PIVOT/KILL：说明更值得转向 BCE-GCL 的哪个变体或 backup idea。
7. 最后给出一句话结论：是否允许进入 `/research-refine-pipeline`。

