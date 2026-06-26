# GCL / Graph SSL 核心文献综述

生成时间：2026-06-26  
任务范围：普通图节点分类；主协议为 stratified random `1:1:8`；目标是为后续 GPT idea 生成准备文献与 gap map。  
检索来源：本地无可复用 PDF；使用 arXiv API、OpenReview、ACM/NeurIPS/CVPR/ACL 官方页、GitHub 官方实现页与网页检索元数据。  
注意：本文只记录机制与可迁移 gap，不产生任何性能 claim；若论文原文声称 SOTA，此处不把它转写为本项目结论。

## 快速结论

本次整理出 21 篇 GCL / Graph SSL 直接相关论文。核心脉络从 DGI/MVGRL 的互信息和多视图学习，发展到 GRACE/GCA 的双视图节点对比，再到 BGRL/CCA-SSG/Graph Barlow Twins 的 negative-free 与冗余约束，随后转向 GraphMAE/MaskGAE/GCMAE 的 masked modeling，以及 2024-2026 的 hard negative、prototype、LLM-TAG、低秩与谱域机制。对普通节点分类而言，最值得继续挖的不是“再堆一个 augmentation”，而是：如何在统一 `1:1:8` split 和 frozen Logistic Regression evaluator 下，让视图、正负样本、样本权重与鲁棒性目标真正围绕节点类别语义工作。

## 文献表

| # | 论文 | 类型 | 核心方法 | 对比/训练目标 | 正负样本 / 视图构造 | 与 GCL 节点分类关系 | 可迁移点 | 迁移风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | [Deep Graph Infomax, Velickovic et al., ICLR 2019](https://openreview.net/forum?id=rklz9iAcKQ) | Graph SSL 基础 | 最大化局部节点表示与全图 summary 的互信息 | local-global discrimination | 正样本为真实图的节点-summary；负样本由特征打乱图产生 | 早期节点表示 SSL baseline，常用于 Cora/CiteSeer/PubMed 线性评估 | local-global 互信息可作为全局语义约束 | summary 太粗，负样本 corruption 可能过于容易；难处理类别内多模态 |
| 2 | [MVGRL, Hassani & Khasahmadi, ICML 2020](https://arxiv.org/abs/2006.05582) | 多视图 GCL | 对比邻接视图与 diffusion 视图 | 跨视图互信息最大化 | 原图邻接与扩散图；节点-图、图-节点对比 | 直接针对节点/图表示，常作为 GCL baseline | diffusion view 可补充高阶结构 | diffusion 计算和存储较重；异配图上可能放大错误平滑 |
| 3 | [GRACE, Zhu et al., 2020](https://arxiv.org/abs/2006.04131) | 节点级 GCL | 两个随机增强图上做节点实例判别 | NT-Xent / InfoNCE | edge dropping + feature masking；同一节点两视图为正，其他节点为负 | 当前仓库已复现的重要 baseline | 简洁、适合统一 split 和 frozen evaluator | false negative 严重；随机增强不保证保留类别语义 |
| 4 | [GraphCL, You et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) | 图级 GCL | 系统研究图增强组合 | graph-level NT-Xent | node dropping、edge perturbation、attribute masking、subgraph | 主要是图分类，但定义了图增强设计空间 | 增强搜索与数据类型匹配思路可迁移 | 图级经验不能直接搬到节点分类；可能改变节点标签语义 |
| 5 | [GCA, Zhu et al., WWW 2021](https://arxiv.org/abs/2010.14945) | 自适应增强 GCL | 按节点中心性和特征重要性自适应扰动 | 跨视图节点对比 | 重要边/特征更少被扰动 | 直接节点分类 baseline，项目已有官方仓库 | augmentation 不应均匀随机，应考虑结构/属性重要性 | 中心性不等于类别语义；可能偏向高阶/高 degree 节点 |
| 6 | [BGRL, Thakoor et al., ICLR 2022](https://arxiv.org/abs/2102.06514) | negative-free GCL | BYOL 式 online/target GNN，预测另一视图表示 | bootstrapped prediction | 简单图增强；无显式负样本；EMA target | 大规模节点表示学习强相关 baseline | 避免 false negative 和全量负样本内存压力 | collapse 防控依赖工程细节；普通小图可能收益不稳定 |
| 7 | [CCA-SSG, Zhang et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf) | negative-free / CCA | 用 CCA 目标对齐两视图并去冗余 | alignment + decorrelation | 两个增强视图；无负样本 | 节点分类常用强 baseline | 低成本、可作为 GRACE 的负样本替代 | batch 统计与图规模相关；小 batch/小图不稳定 |
| 8 | [MERIT, Jin et al., IJCAI 2021](https://arxiv.org/abs/2105.05682) | Siamese / 多尺度 | 多尺度 Siamese 自蒸馏与跨视图/跨网络对比 | cross-view + cross-network contrast | local/global 增强视图；online/target 网络 | 节点表示学习直接相关 | 多尺度一致性可补足单视图局部偏差 | 组件较多，调参空间大；不利于公平 baseline 预算 |
| 9 | [Graph Barlow Twins, Bielak et al., 2021/2022](https://arxiv.org/abs/2106.02466) | redundancy reduction | 图 SSL 中引入 Barlow Twins cross-correlation loss | cross-correlation 接近单位阵 | 两个增强视图；无负样本 | 可替代 InfoNCE 的节点/图表示目标 | 避免 negative 设计，目标简单 | 高维投影头和 batch 统计可能带来隐藏调参 |
| 10 | [GraphMAE, Hou et al., KDD 2022](https://dl.acm.org/doi/10.1145/3534678.3539321) | masked graph modeling | mask 节点属性并重构，使用 scaled cosine error | masked feature reconstruction | feature masking；无正负样本 | 节点分类 Graph SSL 强相关 | generative pretext 可补充 GCL 的全局判别目标 | 依赖原始特征质量；对 bag-of-words 特征可能学到浅层复原 |
| 11 | [GraphMAE2, Hou et al., WWW 2023](https://dl.acm.org/doi/fullHtml/10.1145/3543507.3583379) | enhanced masked SSL | 改进 decoder 与重构正则，提高 masked graph SSL 稳定性 | regularized masked reconstruction | feature/structure corruption 与增强 decoder | 节点分类和大图预训练直接相关 | 用 decoder 约束避免只记输入特征 | 复杂 decoder 可能改变公平计算预算 |
| 12 | [MaskGAE, Li et al., KDD 2023](https://arxiv.org/pdf/2205.10053) | masked structure SSL | mask 边/结构并重构缺失图结构 | masked edge reconstruction | 对部分边/结构做 mask | 可作为结构重构方向 baseline | 可把结构预测与节点分类语义联系起来 | link reconstruction 与 label-discriminative 表示不一定一致 |
| 13 | [COSTA, Zhang et al., KDD 2022](https://arxiv.org/abs/2206.04726) | feature-space augmentation | 在隐藏空间做 covariance-preserving feature augmentation | 对比增强后的特征 | 单视图/多视图 feature sketch；不直接扰动图结构 | 直接节点分类 GCL 相关 | 避免输入级结构扰动破坏语义；计算更省 | feature-space 增强语义可解释性较弱 |
| 14 | [RGCL, Li et al., 2022](https://arxiv.org/abs/2206.07869) | rationale-aware GCL | 用 invariant rationale discovery 生成语义保持视图 | rationale-aware contrastive pretraining | rationale 子结构作为保留视图 | 更多面向图级任务，但视图语义问题与节点分类高度相关 | “增强必须保留判别 rationale”的思想很关键 | 节点级 rationale 无标签时难定义；生成器可能不稳定 |
| 15 | [GCMAE, Wu et al., 2023](https://arxiv.org/abs/2310.15523) | generative + contrastive | 统一 masked autoencoding 与 contrastive learning | local reconstruction + global contrast | masked graph modeling 与多视图 contrast | 直接提示 MAE 与 GCL 互补 | 可设计局部/全局双目标的节点 SSL | 两个目标权重难调；容易变成“拼盘式方法” |
| 16 | [Affinity Uncertainty-based Hard Negative Mining, 2023](https://arxiv.org/html/2301.13340v2) | hard negative mining | 用 affinity uncertainty 区分 hard negative 与 false negative | 加权/筛选负样本对比 | 从相似节点中识别更可靠 hard negatives | 直接针对 GCL false negative 问题 | 将“难负样本”与“不确定假负样本”分开 | 需要可靠不确定性估计；在 1:1:8 少标签下验证困难 |
| 17 | [GCL-LRR / LR-GCL, 2024](https://arxiv.org/abs/2402.09600) | prototype + low-rank | 原型对比学习叠加低秩正则以抗噪 | prototypical contrast + low-rank regularization | 节点原型作为语义锚；低秩约束表示 | 转导节点分类直接相关 | prototype 可缓解 instance-level false negative | 原型数、初始化、更新策略可能引入 test-like tuning 风险 |
| 18 | [GAugLLM, KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3672035) | LLM-enhanced TAG GCL | 用 LLM 改进 text-attributed graph 的增强 | text/structure-aware contrast | LLM 生成或筛选文本属性增强 | 对 Cora/ogbn-arxiv 等文本图有关系 | LLM 可作为语义视图或噪声过滤器 | 普通非文本图不可直接用；LLM 成本和可复现性风险高 |
| 19 | [Enhancing Contrastive Learning on Graphs with Node Similarity, KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3671898) | node similarity objective | 用节点相似性改进正样本和 false-negative 处理 | ideal positives + false-negative aware objective | 扩展可靠正样本，弱化错误负样本 | 直接切中普通节点分类痛点 | 可借助结构/特征相似度构造软标签式对比 | 相似度度量在异配图或噪声特征上会误导 |
| 20 | [Khan-GCL, Wang et al., 2025](https://arxiv.org/html/2505.15103v1) | KAN + hard negatives | KAN encoder 与关键特征识别生成 hard negatives | hard-negative graph contrastive loss | 基于 KAN 系数找关键特征并扰动 | 较新 GCL hard-negative 方向 | “关键特征扰动”可迁移为轻量 feature importance 机制 | KAN 组件可能过重；节点分类未必需要架构替换 |
| 21 | [CORE, Bo & Fang, 2025](https://arxiv.org/abs/2512.13235) | contrastive masked reconstruction | 将 masked feature reconstruction 转为 contrastive objective | 原始/重构 masked node 为正，masked nodes 作负 | masked nodes 内部构造对比 | 与节点/图分类 Graph SSL 直接相关 | 重构目标可变成节点级对比任务 | 预印本需审慎；负样本仍可能包含同类节点 |

## 主题归纳

### 1. 视图构造从随机增强走向语义保持

GRACE 的随机 edge dropping / feature masking 是很好的起点，但普通节点分类的核心问题是：同一节点两视图是否仍保留类别相关信息。GCA、COSTA、RGCL、GAugLLM 代表了几种不同答案：结构/属性重要性、隐藏空间协方差保持、rationale 保留、文本语义增强。当前 gap 是这些机制大多只证明“比随机强”，但很少在统一 `1:1:8` split 下显式测量增强是否保留节点类别语义、是否偏向高 degree 节点。

### 2. false negative 是节点级 GCL 的中心矛盾

节点分类中相邻或同社群节点常常同类，把所有其他节点当负样本会与分类目标冲突。BGRL、CCA-SSG、Graph Barlow Twins 直接绕开负样本；AUHNM、KDD 2024 node similarity、Khan-GCL 则试图修正 hard negatives。后续 idea 应优先围绕“可靠正样本扩展 + 假负样本降权 + 难负样本边界”展开，而不是单纯增加负样本数量。

### 3. masked modeling 与 contrastive learning 正在重新合流

GraphMAE/GraphMAE2/MaskGAE 证明了 reconstruction 在图 SSL 中不是旧路线；GCMAE/CORE 表明 reconstruction 可以转成更判别的 contrastive pretext。对本仓库最有价值的方向是设计最小化的节点级双目标：一个目标负责局部属性/结构补全，另一个目标负责跨视图类别可分性，且用统一 Logistic Regression evaluator 评估。

### 4. robustness / OOD / LLM-TAG 还没有自然融入普通节点分类协议

LR-GCL、GCL robustness 评估、LLM-TAG 方向都在说真实图有噪声、文本属性和结构漂移。但普通 benchmark 的 `1:1:8` 协议只看平均节点分类准确率，难暴露 degree bias、class imbalance、heterophily、feature noise 等问题。后续 GPT idea 应把这些压力测试转成“最小可验证实验”，但不能把 stress test 结果混入主表 claim。

## 对当前仓库的直接启发

- 强 baseline 候选优先级：GRACE、GCA、BGRL、CCA-SSG、GraphMAE/GraphMAE2、COSTA。
- 最可实现的机制方向：false-negative-aware weighting、prototype positive expansion、feature-space semantic augmentation、masked reconstruction + contrastive hybrid。
- 暂不适合立刻主攻的方向：大规模 LLM-TAG 管线、KAN 架构替换、复杂 rationale generator；这些会显著增加实现和调参预算。
- 所有后续 claim 必须继续遵守：固定 split、10 seeds、frozen encoder + Logistic Regression、train/val 选择 evaluator 超参、自动从原始 JSON/CSV 汇总。
