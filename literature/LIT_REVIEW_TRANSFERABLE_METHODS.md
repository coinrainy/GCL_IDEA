# 外部可迁移方法文献综述

生成时间：2026-06-26  
任务范围：从视觉/文本 SSL、bootstrap learning、masked modeling、prototype learning、hard negative mining、样本重加权、鲁棒表示学习、OOD、LLM-GNN 等方向提炼可迁移机制。  
注意：以下论文不直接构成本项目节点分类性能证据，只提供可迁移机制和风险。

## 快速结论

本次整理出 18 篇外部可迁移论文。最有迁移价值的机制包括：SimCLR 的增强组合与投影头、BYOL/SimSiam/DINO 的 teacher-student 和 stop-gradient、MAE/BERT 的高比例 masking 与非对称 encoder-decoder、SwAV 的原型/聚类分配、Barlow Twins/VICReg 的 redundancy reduction、Debiased CL/Hard Negative Mixing/SupCon/Focal Loss 的样本权重与正负样本重定义，以及 IRM/GroupDRO/SAM 的鲁棒泛化思想。

## 文献表

| # | 论文 | 领域 | 核心方法 | 对比/训练目标 | 正负样本 / 视图构造 | 与 GCL 节点分类的关系 | 可迁移点 | 迁移风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | [SimCLR, Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709) | 视觉 SSL | 简化对比学习框架，强调增强组合、投影头、大 batch | NT-Xent | 两个图像增强视图；同图为正，batch 内其他为负 | GRACE/GCA 的直接视觉源头 | 系统 ablation 增强强度、投影头、temperature | 图节点负样本更容易同类；大 batch 不一定可行 |
| 2 | [MoCo, He et al., CVPR 2020](https://arxiv.org/abs/1911.05722) | 视觉 SSL | momentum encoder + queue 提供大量负样本 | InfoNCE | query/key 编码器，队列负样本 | 可作为大图 mini-batch GCL 的负样本缓存思路 | memory queue、momentum update | 旧表示滞后；图节点相关性导致队列假负样本更多 |
| 3 | [BYOL, Grill et al., NeurIPS 2020](https://arxiv.org/abs/2006.07733) | bootstrap SSL | online 网络预测 EMA target 网络 | positive-only prediction | 两个增强视图，无负样本 | BGRL 的直接来源 | negative-free 避免 false negative | collapse 防控依赖 predictor/EMA/BN 等细节 |
| 4 | [SimSiam, Chen & He, CVPR 2021](https://arxiv.org/abs/2011.10566) | bootstrap SSL | 去掉 momentum encoder，仅靠 predictor + stop-gradient | positive-only cosine loss | 两视图，无负样本 | 可简化 BGRL/MERIT 类图方法 | stop-gradient 是轻量 collapse 防控机制 | 图上过平滑可能仍导致表示塌缩 |
| 5 | [DINO, Caron et al., ICCV 2021](https://arxiv.org/abs/2104.14294) | self-distillation | student-teacher 自蒸馏，无标签学习语义结构 | teacher distribution prediction | multi-crop views；centering/sharpening | 可启发节点多尺度 view 与 teacher soft target | soft assignment、teacher sharpening、多尺度局部-全局 | 节点没有自然 crop；teacher 可能放大同配偏见 |
| 6 | [MAE, He et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper) | masked vision modeling | 高比例 patch mask + 非对称 encoder-decoder 重构 | pixel reconstruction | visible patches 编码，masked patches 解码 | GraphMAE/MaskGAE 的关键源头 | 高比例 mask、轻 decoder、只编码可见部分 | 图结构稀疏且节点相关，mask 过强可能破坏连通语义 |
| 7 | [BERT, Devlin et al., NAACL 2019](https://aclanthology.org/N19-1423/) | 文本 SSL | masked language modeling + NSP 预训练 | token prediction | mask token 与上下文预测 | text-attributed graph 与 masked node feature 的源头 | 上下文预测、预训练-微调分离 | 节点特征通常不是自然语言 token；重构目标可能过浅 |
| 8 | [SwAV, Caron et al., NeurIPS 2020](https://arxiv.org/abs/2006.09882) | prototype SSL | 比较聚类分配而非所有 pairwise features | swapped prediction between cluster assignments | multi-crop views；在线 clustering | 可替代 instance discrimination 的 prototype GCL | 原型分配、避免大规模 pairwise negatives | 原型数选择敏感；聚类可能对少数类不友好 |
| 9 | [Barlow Twins, Zbontar et al., ICML 2021](https://proceedings.mlr.press/v139/zbontar21a.html) | redundancy reduction | cross-correlation 接近单位阵，避免冗余 | invariance + redundancy reduction | 两个增强视图，无负样本 | Graph Barlow Twins / CCA-SSG 的源头 | 无负样本、低假负风险 | 依赖 batch 统计；高维投影头可能加大预算 |
| 10 | [VICReg, Bardes et al., ICLR 2022](https://arxiv.org/abs/2105.04906) | redundancy/collapse control | variance、invariance、covariance 三项正则 | 表示对齐 + 方差下界 + 去相关 | 两个增强视图，无负样本 | 可用于防止 GCL/GraphMAE hybrid 塌缩 | 显式方差下界适合诊断 collapse | loss 权重多，容易引入调参优势 |
| 11 | [Supervised Contrastive Learning, Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362) | 监督表示学习 | 同类样本全作正样本，异类作负样本 | supervised contrastive loss | label-defined positives/negatives | 1:1:8 train/val 少标签可用于轻量 semi-supervised auxiliary | 用少量标签校准正负样本语义 | 不能用 test label；少标签下可能过拟合 train classes |
| 12 | [Debiased Contrastive Learning, Chuang et al., NeurIPS 2020](https://mitibm.mit.edu/research/blog/debiased-contrastive-learning/) | false-negative debias | 修正负样本中含同类样本的偏差 | debiased InfoNCE estimator | 无标签估计 false-negative 影响 | 节点 GCL false-negative 的外部理论来源 | 负样本校正项、无需显式标签 | 原假设多来自 i.i.d. 数据，图相关性更强 |
| 13 | [Hard Negative Mixing, Kalantidis et al., NeurIPS 2020](https://arxiv.org/abs/2010.01028) | hard negative mining | 在特征空间混合 hard negatives | harder contrastive negatives | feature-level negative mixing | 可启发节点 embedding hard-negative 生成 | on-the-fly hard negative 合成 | 图上 hard negative 可能是同类节点，需 false-negative guard |
| 14 | [Focal Loss, Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) | 样本重加权 | 降低易样本权重，聚焦难样本 | reweighted CE | 无视图；按预测置信度重权 | 可迁移为 contrastive pair/node hardness weighting | hard node / hard pair 自适应权重 | 难样本可能是噪声或异配结构，不宜盲目放大 |
| 15 | [Invariant Risk Minimization, Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893) | OOD 泛化 | 学习跨环境不变表示 | invariant classifier optimality | 多环境训练分布 | 可把 split、degree bucket、augmentation environment 作为环境 | 强化跨增强/跨结构环境不变性 | 环境划分难；IRM 在深网中训练不稳定 |
| 16 | [GroupDRO, Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731) | 鲁棒优化 | 最小化最差 group loss | worst-group risk minimization | group-defined losses | 可按 degree/class/feature sparsity/社区划分 worst-group 监控 | 抑制平均准确率掩盖弱群体 | 需要 group 定义；主协议不能擅自换评价口径 |
| 17 | [SAM, Foret et al., ICLR 2021](https://openreview.net/forum?id=6Tm1mposlrM) | 泛化优化 | 同时最小化 loss 与邻域 sharpness | min-max sharpness-aware objective | 无视图 | 可用于稳定 GCL pretraining 或 evaluator 训练 | 平坦最小值、泛化稳定性 | 额外反向传播成本；可能改变 baseline 预算公平性 |
| 18 | [STAGE, Zolnai-Lucas et al., ACL KaLLM 2024](https://aclanthology.org/2024.kallm-1.10.pdf) | LLM-TAG | 用预训练 LLM 生成 text-attributed graph 节点特征 | downstream node classification with LLM features | 文本属性嵌入，无显式对比 | 启发 Cora/ogbn-arxiv 等文本节点的特征增强 | LLM 语义特征可作为替代视图或增强 | 非文本图不可用；API/模型版本影响可复现 |

## 机制归纳

### Bootstrap / negative-free

BYOL、SimSiam、DINO 说明 positive-only 学习可以通过 EMA teacher、stop-gradient、centering/sharpening、predictor 等机制避免塌缩。迁移到 GCL 节点分类时，最大价值是避开 false negative；最大风险是图消息传递已经会让相邻节点预先相似，positive-only 可能进一步过平滑。

### Masked modeling

BERT 和 MAE 的共同经验是：足够难的 reconstruction pretext 可以迫使模型学习上下文。图上需要谨慎：节点特征有时是稀疏 bag-of-words，直接重构可能学到输入复制；结构重构又可能偏向 link prediction 而不是 class discrimination。适合的迁移方式是“masked context prediction + 轻量 contrastive/prototype 校准”。

### Prototype / clustering

SwAV/DINO 的 cluster assignment 机制可把 instance-level 对比改为 prototype-level 对比，正好缓解同类节点被当负样本的问题。风险是原型会吞掉少数类，或在同配图中退化为 degree/community 聚类。

### Pair weighting / hard sample

Debiased CL、Hard Negative Mixing、Focal Loss、SupCon 都在处理“哪些 pair 更该被拉近/推远”。节点分类最自然的迁移是：用 train/val 标签、结构相似度、特征相似度、当前 embedding uncertainty 共同决定 pair 权重，但必须避免 test leakage。

### Robustness / OOD

IRM、GroupDRO、SAM 提供了从平均性能之外思考泛化的框架。当前仓库主表仍应坚持普通节点分类准确率，但可以设置 development-only stress tests：degree bucket、feature noise、edge perturbation、class imbalance、heterophily split。它们用于 idea 筛选，不直接构成最终性能 claim。

## 对后续 GPT idea 的提示

- 优先考虑轻量可实现机制，而不是引入大模型或复杂架构。
- 每个 idea 都应明确它解决的是：视图语义、false negative、masked objective、degree/class bias、还是鲁棒泛化。
- 设计时必须写清楚：是否需要标签；若需要，只能使用 train/val，不得使用 test。
- 若使用外部机制，应给出最小消融：原始 GRACE/GCA + 单一机制，不要一次拼多个组件。
