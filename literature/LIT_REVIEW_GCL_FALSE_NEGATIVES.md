# 图对比学习假负样本方向专题综述

生成时间：2026-06-26  
任务：围绕“图对比学习中的假负样本、hard negative、负采样偏差、正样本扩展”继续加深文献调研。  
项目约束：普通节点分类；主协议 stratified random `1:1:8`；GCL 评估默认 frozen encoder + Logistic Regression；本文不产生性能 claim。

## 结论先行

本次新增聚焦论文 24 篇，其中图/GCL 直接相关 12 篇，通用 contrastive false-negative / hard-negative / PU learning 机制 12 篇。核心判断是：**普通 hard negative mining 不能直接迁移到节点级 GCL**。在图上，最相似的负样本往往并不是“最有价值的真负样本”，而是因为同配、消息传递、社群结构或特征相似造成的假负样本。假负样本方向真正有潜力的切口，是把“相似度高”拆成三件事：同类概率、真负概率、边界信息量。

## 直接相关：Graph / GCL 假负样本论文

| # | 论文 | 类型 | 核心方法 | 对比目标 / 训练目标 | 正负样本处理 | 与普通节点分类关系 | 可迁移点 | 迁移风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | [Graph Debiased Contrastive Learning with Joint Representation Clustering, Zhao et al., IJCAI 2021](https://www.ijcai.org/proceedings/2021/473) | clustering debias | 联合表示学习与聚类，用聚类伪标签减少随机负采样中的同类假负 | debiased graph contrastive loss + clustering objective | 同簇更可能是假负；跨簇更可作为负样本 | 直接面向无监督图表示和节点任务 | 用聚类估计类别结构，是 prototype 方向早期图版本 | 早期聚类不可靠；簇数选择和类别数绑定风险 |
| 2 | [ProGCL, Xia et al., ICML 2022](https://proceedings.mlr.press/v162/xia22b.html) | true-negative probability | 指出 GNN 消息传递导致“最相似负样本多为假负”；用概率估计 negative 是否为 true negative | ProGCL-weight / ProGCL-mix 插入 GCL loss | 同时考虑 similarity 和 true-negative probability，而不是只看 hardness | 非常贴合普通节点分类 false negative 问题 | 可做 GRACE/GCA 插件；区分 hard true negatives 与 false negatives | 需要拟合分布，训练早期估计可能不稳；实现需防止引入大调参 |
| 3 | [Structure-Aware Hard Negative Mining for Heterogeneous GCL, Zhu et al., 2021](https://arxiv.org/abs/2108.13886) | structure-aware HNM | 在异构图中按 metapath/schema 多语义视图建模，并用结构特征度量 hard negatives | heterogeneous multi-view contrastive objective | 结构特征决定 hard negative 权重，并可合成更难负样本 | 主任务是异构图，不是普通同构节点分类 | “hardness 应依赖图结构，不只是 embedding 相似度” | 异构 metapath 假设不适合 Planetoid/Amazon 同构图 |
| 4 | [Generating Counterfactual Hard Negative Samples for GCL, Yang et al., 2022](https://arxiv.org/abs/2207.00148) | counterfactual hard negatives | 用反事实机制生成与 anchor 相似但语义不同的 hard negatives | counterfactual graph contrastive loss | 生成式 hard negatives，避免直接从 batch 采到 false negatives | 可启发普通节点分类中的边界负样本合成 | “真负样本可以生成，不一定采样” | 文中提到不同标签假设，普通无监督节点分类不可用 test label；反事实生成复杂 |
| 5 | [False Negative Sample Detection for Graph Contrastive Learning, Tsinghua Science and Technology 2023](https://www.sciopen.com/article/10.26599/TST.2023.9010043) | FN detection | 用属性和结构感知模块检测 false negatives | GCL loss + false-negative detection / filtering | 检测疑似假负，降低其负样本作用 | 直接针对图节点表示 | 结构+属性双信号适合普通节点图 | 需要核查源码和具体公式；检测误差可能伤害真负边界 |
| 6 | [Affinity Uncertainty-based Hard Negative Mining in GCL, Niu et al., 2023/2024](https://arxiv.org/abs/2301.13340) | uncertainty-weighted HNM | 用 collective affinity 建立判别模型，以不确定性决定 hard negative 权重 | uncertainty-weighted GCL loss；理论等价于 adaptive-margin triplet | 高相似负样本不直接当 hard negative，而看 affinity uncertainty | 直接覆盖 node classification 和 graph classification | “不确定性高的边界负样本”比“相似度高”更合理 | collective affinity 计算开销和实现复杂度需评估 |
| 7 | [Enhancing Graph Contrastive Learning with Node Similarity, Chi & Ma, KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3671898) / [arXiv](https://arxiv.org/abs/2208.06743) | ideal objective / node similarity | 从“所有正样本、无假负样本”的理想目标出发，用 node similarity 建模正/负采样分布 | enhanced GCL objective | 扩展正样本并降低 false-negative 负样本影响 | 极其贴合普通节点分类，尤其同配图 | 可将 structural/feature/embedding similarity 转为 soft positive / soft negative | 相似度在异配图、噪声特征或低 homophily 数据上可能误导 |
| 8 | [Debiased GCL based on Positive and Unlabeled Learning, 2023/2024](https://www.researchgate.net/publication/376616057_Debiased_graph_contrastive_learning_based_on_positive_and_unlabeled_learning) | PU learning debias | 将同簇样本视为 positive，其余视为 unlabeled，用 PU learning 估计负倾向分数 | weighted graph contrastive loss | negative propensity score 降低 false negatives 的权重 | 直接关联 node classification / clustering / link prediction | PU learning 是少标签/无标签 false-negative debias 的自然数学框架 | 当前可检索元数据分散，正式引用前需核验发表版本 |
| 9 | [Negative Sample Debiased Sampling Contrastive Learning for Node Classification, 2024](https://link.springer.com/article/10.1007/s40747-024-01441-z) | pseudo-label debiased sampling | 用训练出的分类器给未标注节点伪标签，再做负样本去偏采样 | semi-supervised contrastive + classifier pseudo-label | 避免把伪同类节点采作负样本 | 与半监督节点分类直接相关 | train/val 标签可用于轻量校准 false negatives | 容易变成半监督方法；必须严格禁止 test label 和 test-tuned pseudo-label |
| 10 | [Graph Contrastive Learning via Cluster-refined Negative Sampling, 2024](https://arxiv.org/html/2410.18130v1) | cluster-refined negative sampling | 面向文本分类图的 cluster-refined 负采样，缓解相似节点错误配为负样本 | GCL objective + cluster refined negatives | 根据聚类过滤/修正负样本 | 更偏文本图分类，但机制可迁移 | cluster refinement 可作为普通图节点的 conservative filter | 文本分类图场景不同；需避免过度依赖文本语义 |
| 11 | [Positive Mining in GCL, OpenReview withdrawn 2024](https://openreview.net/forum?id=V0Hyw9Tz5W) | positive mining | 用 mixture model 估计 anchor 与其他节点成为 true positive 的概率 | positive-mining graph contrastive loss | 将高概率 true positive 加入正样本集合 | 方向相关，但非正式发表 | 提示“扩正样本”比“挖 hard negative”可能更稳 | withdrawn，不能作为强相关工作支柱 |
| 12 | [Khan-GCL, Wang et al., 2025](https://arxiv.org/abs/2505.15103) | generated hard negatives | KAN encoder + 基于关键特征识别构造语义 hard negatives | hard-negative graph contrastive loss | 从关键特征扰动中生成 hard negatives | 较新 GCL hard-negative 方向 | “生成式 hard negatives 需保留语义差异” | KAN/关键特征模块较重，可能不适合作为当前仓库第一主线 |

## 可迁移源头：通用 SSL / CL 假负样本论文

| # | 论文 | 机制 | 对 GCL 假负样本的启发 | 迁移风险 |
|---|---|---|---|---|
| 1 | [Debiased Contrastive Learning, Chuang et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) | 修正负样本中同类样本造成的 sampling bias | 可以把 InfoNCE denominator 中的 negative mass 视为被 false positives 污染，需要 debiased estimator | 原理论多基于 i.i.d. 数据；图节点相关性更强 |
| 2 | [Contrastive Learning with Hard Negative Samples, Robinson et al., ICLR 2021](https://arxiv.org/abs/2010.04592) | 按 hardness 重采样负样本 | 提供 hard-negative sampling 的基本数学框架 | ProGCL 已指出它直接迁移到图上会选中大量假负 |
| 3 | [Boosting Contrastive SSL with False Negative Cancellation, Huynh et al., WACV 2022](https://research.google/pubs/boosting-contrastive-self-supervised-learning-with-false-negative-cancellation/) | false negative elimination / attraction | 对检测出的假负有两种处理：移除或转成吸引项 | 图上检测 precision 若低，attraction 会把真负拉近 |
| 4 | [Incremental False Negative Detection, Chen et al., 2021](https://arxiv.org/abs/2106.03719) | 随训练进展逐步检测并移除 false negatives | 适合 GCL 训练早期不信任 embedding、后期逐步使用 pseudo semantic structure | 需要 schedule；过早检测会自我强化错误 |
| 5 | [Positive Unlabeled Contrastive Learning, 2022](https://arxiv.org/html/2206.01206v3) | puNCE / positive-unlabeled objective | 将 unknown negatives 当 unlabeled mixture，而不是硬标签负样本 | 类先验或 mixture proportion 难估计；多类图更复杂 |
| 6 | [Contrastive Learning with Negative Sampling Correction / PUCL, 2024](https://arxiv.org/html/2401.08690v1) | 用 positive-unlabeled correction 修正负采样偏差 | 可作为 DGCL-PU 的通用理论背景 | 仍需估计污染比例；图上 class prior 不均衡 |
| 7 | [Difficulty-Based Sampling for Debiased Contrastive Representation Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Jang_Difficulty-Based_Sampling_for_Debiased_Contrastive_Representation_Learning_CVPR_2023_paper.pdf) | 同时处理 hard/easy bias 与 true/false negative bias | 支持“hardness 和 false-negative probability 必须分开估计” | 视觉方法迁移到图需替换 difficulty estimator |
| 8 | [Hard Negative Mixing, Kalantidis et al., NeurIPS 2020](https://arxiv.org/abs/2010.01028) | feature-space hard negative synthesis | 生成边界负样本可以减少采样依赖 | 图上合成负样本需先防 false negative |
| 9 | [AdCo, Hu et al., CVPR 2021](https://arxiv.org/abs/2011.08435) | self-trained negative adversaries | 可启发学习一组动态负样本原型，而不是枚举所有节点负样本 | adversarial negatives 可能追着同类节点跑，需 true-negative guard |
| 10 | [Nearest-Neighbor Contrastive Learning, Dwibedi et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html) | nearest neighbors 作为额外 positives | 在图上可把可靠近邻从负样本池移到正样本池 | 图近邻不一定同类，异配图风险高 |
| 11 | [Prototypical Contrastive Learning, Li et al., ICLR 2021](https://openreview.net/forum?id=KmykpuSrjcq) | prototypes/clusters 替代纯 instance discrimination | 可减少同类节点互相排斥，转向 prototype-level contrast | prototype 数、更新和 collapse 控制难 |
| 12 | [Supervised Contrastive Learning, Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362) | 同类全为正样本，异类为负样本 | train/val 少量标签可作为校准 false-negative detector 的弱监督 | 不能用 test label；容易偏离“无监督 GCL”叙事 |

## 方法谱系：假负样本怎么处理

### 1. 过滤：把疑似假负从 denominator 中拿掉

代表：FNC、IFND、FNSD。  
优点是简单，能直接插入 GRACE/GCA 的 InfoNCE；风险是过滤过多会降低 uniformity，导致表示过度聚团。适合当前仓库做第一批 development ablation，因为改动小、失败也清楚。

### 2. 降权：保留负样本但调低其 repulsion

代表：DCL、ProGCL-weight、AUGCL、DGCL-PU、node similarity objective。  
这是最适合当前仓库的方向：不需要硬判断一个节点是不是假负，而是给每个 pair 一个 soft weight。关键问题是 weight 的来源必须不泄露 test：结构相似度、特征相似度、当前 embedding、train/val 少量标签都可以，但要分清无监督版和半监督版。

### 3. 转正：把高置信假负变成额外正样本

代表：FNC attraction、NNCLR、PCL、node similarity objective、positive mining。  
这条线更接近“扩正样本”，可能比挖 hard negatives 更符合节点分类。但风险也更大：一旦把真负误转正，会直接损害类别边界。适合用 conservative threshold 或 top-k small budget。

### 4. 生成：构造真 hard negatives

代表：CGC、Hard Negative Mixing、AdCo、Khan-GCL。  
理论上最有边界判别价值，但工程和审稿风险都较高。对当前仓库，生成式方法应作为第二阶段，而不是第一个 idea。

### 5. 概率建模：估计 true-negative probability / negative propensity

代表：ProGCL、DGCL-PU、PUCL、AUGCL。  
这是最值得深挖的机制，因为它把“hardness”与“false-negative risk”拆开。潜在创新点在于：把图上的结构、特征、embedding、degree/class imbalance 统一成一个可解释的 pair-level negative reliability。

## 当前领域还没有很好解决的问题

1. **hardness 与 false-negative risk 缠在一起**：许多方法仍用相似度作为主要信号，但图上高相似负样本常常是假负。
2. **训练早期估计不可靠**：embedding 初期没有语义结构，过早过滤或转正会制造 confirmation bias。
3. **结构相似不等于类别相同**：同配图上相似度好用，异配图或 Amazon 等属性噪声图上可能失效。
4. **负样本修正与正样本扩展没有统一**：多数方法只处理 denominator，少数处理 numerator，但两者本质是同一个 pair semantics 问题。
5. **缺少严格协议下的复现**：很多论文使用 public split、不同 evaluator 或没有统一 10 seeds；当前仓库需要重新在 `1:1:8` + logreg_val 下验证。
6. **少标签信息如何合法使用不清楚**：在 `1:1:8` 中 train/val 标签是合法的，但方法需要明确区分无监督 GCL、self-training 和 semi-supervised auxiliary。

## 面向当前仓库的优先方向

### FN-1：Soft Negative Reliability for GRACE/GCA

- 核心：为每个 anchor-negative pair 估计 `r_ij = P(true negative | structure, feature, embedding)`，用 `r_ij` 缩放 denominator。
- 来源：ProGCL、AUGCL、DGCL-PU、DCL。
- 最小实现：只改 GRACE loss；先做无标签版，使用 feature cosine、shortest-hop/adjacency、embedding similarity 的 conservative 组合。
- 失败判据：Cora/CiteSeer 有提升但 Amazon/Photo 明显下降，说明过度依赖同配假设。

### FN-2：Warmup then False-Negative Filtering

- 核心：前若干 epoch 原始 GRACE；embedding 稳定后才启用 false-negative filter。
- 来源：IFND、FNC。
- 最小实现：按 epoch schedule + top-k similarity / cluster consistency 过滤。
- 失败判据：embedding rank 或 class compactness 变差，说明过滤过强。

### FN-3：Conservative Positive Expansion

- 核心：只把高置信“可能同类”的少量节点加入 positives，不大规模转正。
- 来源：Node Similarity objective、NNCLR、PCL、GDCL。
- 最小实现：每个 anchor 只选 top-1/top-3 reliable positives；设置 confidence threshold。
- 失败判据：少数类或低 degree 节点被大类原型吞掉。

### FN-4：Uncertainty-aware Hard True Negative Mining

- 核心：先排除/降权高 false-negative risk，再从剩余样本中找 hard true negatives。
- 来源：AUGCL、ProGCL、Robinson HNS。
- 最小实现：`hardness = similarity * reliability`，而不是只用 similarity。
- 失败判据：hardness 越高性能越差，说明 reliability estimator 没有分开真/假负。

### FN-5：Train/Val-Calibrated Semi-supervised Debias

- 核心：用 10% train 和 10% val 中的标签校准 pair reliability，但不使用 test。
- 来源：SupCon、NDSCL、PU learning。
- 最小实现：只用 train 标签构建少量 pair prior，用 val 选择阈值；test 只报告。
- 失败判据：无标签版失败但半监督版好，说明 idea 可能不再是纯 GCL 贡献，需要调整论文叙事。

## 建议后续阅读顺序

1. ProGCL：先理解“图上高相似 hard negatives 多为 false negatives”的核心论证。
2. KDD 2024 Node Similarity：看理想目标如何同时扩正样本和去假负。
3. AUGCL：看 uncertainty 如何进入 hard negative 权重。
4. GDCL / DGCL-PU：看 clustering 与 PU learning 如何建模负倾向。
5. FNC / IFND：看通用 false-negative detection 的过滤与转正策略。
6. DCL / Robinson HNS：作为理论背景，帮助解释为什么直接 hard negative mining 不够。

## 对 GPT Idea Creator 的新增约束

- 生成假负样本 idea 时必须明确：是过滤、降权、转正、生成，还是概率建模。
- 必须把 `hardness` 与 `true-negative reliability` 分成两个量，不允许只用 embedding similarity 定义 hard negative。
- 必须说明是否使用 train/val 标签；若使用，方法应标为 semi-supervised auxiliary，不能伪装成完全无监督。
- 首轮 idea 优先落在 GRACE/GCA loss 插件，不建议先做 KAN、LLM 或复杂反事实生成器。
- 所有实验先标为 `development`，不能产生 SOTA 或 robust claim。
