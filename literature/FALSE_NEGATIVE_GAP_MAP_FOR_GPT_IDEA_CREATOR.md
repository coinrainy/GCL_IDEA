# 假负样本方向 Gap Map

生成时间：2026-06-26  
阶段建议：`READY_FOR_FALSE_NEGATIVE_IDEA_CREATOR`

## 核心问题定义

在节点级图对比学习中，anchor 节点的“其他节点”通常被当作负样本。但由于图同配性、社群结构、相似文本属性、GNN 消息传递和类别不均衡，许多高相似负样本其实可能与 anchor 同类或语义接近。这类假负样本会在 InfoNCE denominator 中被推远，造成：

- 类内表示被撕裂；
- 线性分类边界变差；
- hard negative mining 选错对象；
- 少数类和低 degree 节点更容易受伤；
- 训练变慢或需要更强 augmentation 才能抵消。

## 精细化 Gap

| Gap ID | Gap | 文献线索 | 可形成 idea 的切口 | 当前仓库适合度 |
|---|---|---|---|---|
| FN-G1 | hard negative 与 false negative 没有分离 | ProGCL、AUGCL、Robinson HNS | 定义 `hardness` 和 `true-negative reliability` 两个独立量 | 高 |
| FN-G2 | 负样本只降权，正样本不扩展 | DCL、ProGCL、Node Similarity、NNCLR | 同时修正 numerator 和 denominator | 高 |
| FN-G3 | 训练早期 pseudo semantics 不可靠 | IFND、FNC、GDCL | warmup / curriculum false-negative handling | 高 |
| FN-G4 | 图结构相似度在异配/噪声图上失效 | Node Similarity、FNSD、GCA | 多信号 reliability：结构 + 特征 + embedding + uncertainty | 高 |
| FN-G5 | clustering/prototype 容易吞掉少数类 | GDCL、PCL、PMGCL | balanced / degree-aware prototype 或 conservative top-k positives | 中高 |
| FN-G6 | 生成式 hard negatives 复杂且可能泄露语义 | CGC、Khan-GCL、HNM | 先做 embedding-space 小扰动，不做复杂生成器 | 中 |
| FN-G7 | 少标签信息边界模糊 | SupCon、NDSCL、PUCL | 明确无监督版与 train/val-calibrated 版 | 中高 |
| FN-G8 | 缺少统一协议验证 | 多数历史 GCL 论文 | 固定 `1:1:8` + logreg_val + 10 seeds 复验 | 极高 |

## 最高优先级 Idea 方向

### 1. Reliability-weighted InfoNCE

- 目标 gap：FN-G1、FN-G4、FN-G8。
- 机制：每个 negative pair 得到一个 reliability 权重；越像假负，repulsion 越弱。
- 最小版本：`weight = stopgrad(sigmoid(a * feature_distance + b * graph_distance - c * embedding_similarity))`，参数只用 train/val 或固定启发式。
- 为什么适合当前仓库：GRACE/GCA loss 插件，工程最轻。
- 主要风险：同配假设过强，Amazon/Photo 可能不稳定。

### 2. Curriculum False-negative Filter

- 目标 gap：FN-G3、FN-G8。
- 机制：前期不处理 false negatives；中后期根据稳定 embedding / cluster consistency 过滤少量高风险 negatives。
- 最小版本：epoch warmup + 每个 anchor 过滤 top-k 最相似且结构/特征一致的节点。
- 主要风险：过滤太多导致 uniformity 下降。

### 3. Joint Positive Expansion and Negative Debias

- 目标 gap：FN-G2、FN-G5。
- 机制：高置信同类候选进入 soft positives；低置信相似节点只降权，不转正。
- 最小版本：每个 anchor 只扩 1-3 个正样本，阈值由 val 或无标签稳定性决定。
- 主要风险：误转正破坏类别边界；少数类被大簇吸收。

### 4. Reliability-gated Hard True Negative Mining

- 目标 gap：FN-G1、FN-G6。
- 机制：先估计 true-negative reliability，再在可靠负样本中选 hard negatives。
- 最小版本：`score = similarity * reliability`；只在 denominator 加权，不生成样本。
- 主要风险：若 reliability 不准，会比原始 GRACE 更差。

### 5. Train/Val-calibrated False-negative Debias

- 目标 gap：FN-G7、FN-G8。
- 机制：用 train 标签形成 pair prior，用 val 选择阈值；测试集完全隔离。
- 最小版本：有标签节点之间估计 same-class similarity 分布和 different-class similarity 分布，用于校准无标签 pair。
- 主要风险：论文叙事从无监督 GCL 变成 semi-supervised GCL，需要诚实标注。

## 最小验证实验

| 实验 | 目的 | 方法候选 | 数据集 | 对照 | 输出 |
|---|---|---|---|---|---|
| FN-E1 | 验证降权是否有效 | Reliability-weighted InfoNCE | Cora/CiteSeer/PubMed | GRACE 原版 | 10 seeds development，logreg_val |
| FN-E2 | 检查同配依赖 | 同 FN-E1 | Computers/Photo/DBLP | GRACE/GCA | 分数据集稳定性，不做正式 claim |
| FN-E3 | 验证 warmup 必要性 | Curriculum filter | Cora/PubMed | no-warmup filter | embedding variance/rank + accuracy |
| FN-E4 | 比较过滤/降权/转正 | 三种 pair 处理 | Cora/CiteSeer | 同一 backbone | ablation 表 |
| FN-E5 | 弱群体诊断 | 最佳 development 方法 | 全部小中图 | GRACE | degree bucket / class bucket diagnostic |

## 不建议首轮做的方向

- 复杂反事实图生成器：难以控制语义，审稿时容易被问是否使用标签或隐式 oracle。
- KAN/大架构替换：贡献会混入 encoder capacity。
- LLM 语义视图：只适合文本属性图，无法覆盖 Amazon/Photo 等普通属性图。
- 直接照搬视觉 hard negative sampling：ProGCL 已指出图上会大量采到假负。

## 推荐 GPT 下一步读取

1. `literature/LIT_REVIEW_GCL_FALSE_NEGATIVES.md`
2. `literature/FALSE_NEGATIVE_GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
3. `literature/LIT_REVIEW_GCL_CORE.md`
4. `literature/GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
5. `idea-stage/GPT_IDEA_CREATOR_CONTEXT.md`
