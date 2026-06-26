# 面向 GPT Idea Creator 的 Gap Map

生成时间：2026-06-26  
阶段决策：`READY_FOR_GPT_IDEA_CREATOR`  
输入来源：

- `literature/LIT_REVIEW_GCL_CORE.md`
- `literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`
- 当前仓库研究契约：普通节点分类、GCL 主线、stratified random `1:1:8`、frozen encoder + Logistic Regression evaluator。

## 现有 GCL 的主要不足

| Gap ID | 不足 | 代表文献脉络 | 为什么影响普通节点分类 | 当前仓库相关性 |
|---|---|---|---|---|
| G1 | 视图增强缺少类别语义保证 | GRACE、GCA、GraphCL、COSTA、RGCL | 随机 edge/feature perturbation 或中心性规则可能删除节点分类所需信号；同一节点两视图一致不等于类别可分 | 高；GRACE 已复现，便于在同一脚本上替换增强 |
| G2 | false negative / hard negative 定义混乱 | GRACE、AUHNM、KDD 2024 node similarity、Khan-GCL、Debiased CL | 同类节点在 GCL 中常被推远，尤其同配图、社群图和少标签设置下会伤害线性分类 | 很高；可通过 loss weighting 小改实现 |
| G3 | instance-level 对比与 class-level 分类目标错位 | SimCLR-style GCL、SupCon、SwAV、prototype GCL、LR-GCL | 节点分类最终需要类别边界；单纯实例判别可能过度均匀化，类内结构不紧凑 | 高；prototype/soft positive 可作为 GRACE/GCA 插件 |
| G4 | masked modeling 与 contrastive objective 未形成简洁统一 | GraphMAE、GraphMAE2、MaskGAE、GCMAE、CORE、MAE/BERT | 重构目标捕捉局部上下文，对比目标捕捉判别关系；现有组合常复杂且权重敏感 | 中高；需要新增 pretext，但可先做 feature-mask 最小版 |
| G5 | 评估协议不一致导致 claim 不可迁移 | DGI/MVGRL/GRACE/GCA/GraphMAE 等历史协议 | public split、随机 split、不同 evaluator、test 调参混杂时，数字无法直接比较 | 极高；本仓库已规定统一 split 与 logreg_val |
| G6 | degree / class imbalance / heterophily 弱群体被平均结果掩盖 | GroupDRO、Focal Loss、HAR/degree-bias GCL、robust GCL | 平均准确率可能提升但低 degree、少数类、异配边节点变差 | 高；可加 development-only 诊断，不进入主 claim |
| G7 | LLM/TAG 方法与普通图节点分类之间有语义鸿沟 | GAugLLM、STAGE、LangGSL、LLM-BP | 文本图可借 LLM 语义；非文本图无法直接使用，且成本/复现风险高 | 中；适合 Cora/ogbn-arxiv 辅助探索，不适合作为第一主线 |
| G8 | 计算预算与调参空间可能压倒方法贡献 | MVGRL diffusion、MERIT 多组件、KAN/LLM 方法 | 复杂模块可能带来不公平搜索空间，难与 GRACE/GCA/GCN/GAT 公平比较 | 高；首轮 idea 应偏轻量插件 |

## 可迁移外部机制

| 机制 | 来源 | 可迁移成什么 | 最关键约束 | 风险 |
|---|---|---|---|---|
| stop-gradient / EMA teacher | BYOL、SimSiam、DINO、BGRL | negative-free 节点一致性目标 | 需要 collapse 诊断：embedding variance、rank、class separability | 过平滑或塌缩 |
| prototype / swapped assignment | SwAV、DINO、LR-GCL | cluster/prototype-level positive expansion | prototype 数只能由 train/val 或无标签准则确定 | 少数类被大原型吞掉 |
| debiased negative weighting | Debiased CL、AUHNM、node similarity GCL | false-negative-aware InfoNCE | 不得用 test label；相似度需结构/特征/embedding 多信号校准 | 相似度错则强化错误 |
| hard negative mixing | Hard Negative Mixing、Khan-GCL | 在 embedding/feature 空间合成边界负样本 | hard negative 必须先过 false-negative guard | 把同类节点推得更远 |
| focal / hardness reweighting | Focal Loss、HAR | 对 hard nodes / hard pairs 加权 | 难度来自 train/val 或自监督信号，不来自 test | 噪声节点被过度放大 |
| masked context prediction | BERT、MAE、GraphMAE、MaskGAE | feature/edge mask + lightweight decoder | reconstruction 不能成为主表性能 claim；只作 pretraining objective | 复制输入或偏向 link prediction |
| redundancy reduction | Barlow Twins、VICReg、CCA-SSG | 无负样本 decorrelation regularizer | 记录投影维度与 loss 权重，控制预算 | batch 统计敏感 |
| robust optimization | IRM、GroupDRO、SAM | development-only stress test 或辅助正则 | 主评价仍是统一节点分类；stress 只筛 idea | 过度复杂，难证明贡献 |
| LLM semantic view | STAGE、GAugLLM、LangGSL | 文本图上用 LLM embedding/description 作语义 view | 必须固定 LLM 版本、缓存输出、记录成本 | 非文本图不可迁移 |

## 可能形成的 Idea 方向

这些不是最终 idea，只是供 GPT 下一步生成方法时读取的方向边界。

### D1. False-negative-aware GRACE

- 核心：在 GRACE/GCA 的 InfoNCE 中加入 soft negative mask 或 pair weight。
- 信号：结构近邻、feature cosine、当前 embedding similarity、train/val label consistency 可作为候选；test label 禁止使用。
- 预期验证：与 GRACE 同 split、同 encoder、同 logreg_val evaluator；只替换 loss weighting。
- 风险：相似度度量把真负样本误判为假负，导致类间边界变软。
- 当前仓库适合度：高。实现改动小，baseline 已有。

### D2. Prototype-guided Positive Expansion

- 核心：用无标签 clustering/prototype 将可靠同簇节点作为软正样本，缓解 instance discrimination 与 class-level 目标错位。
- 信号：EMA prototype、Sinkhorn/SwAV-style balanced assignment、低置信节点不参与。
- 预期验证：Cora/CiteSeer/PubMed + Amazon；报告 prototype 数敏感性。
- 风险：prototype 数和更新策略调参大；少数类可能被吞并。
- 当前仓库适合度：中高。需要新增聚类/prototype 模块，但不必改数据管线。

### D3. Feature-space Semantic Augmentation

- 核心：不直接删边/删特征，而在 encoder hidden space 做 covariance-preserving 或 uncertainty-aware perturbation。
- 来源：COSTA、Barlow Twins、VICReg。
- 预期验证：替换 GRACE 的输入增强或加入单视图 feature augmentation。
- 风险：隐藏空间扰动难解释；可能只是一种 regularization。
- 当前仓库适合度：高。计算成本较低。

### D4. Masked-Contrastive Node Pretraining

- 核心：对 masked nodes 做 feature/structure context prediction，同时将原始与重构的 masked-node 表示作为正对，保留 contrastive 判别性。
- 来源：GraphMAE、MaskGAE、GCMAE、CORE、MAE。
- 预期验证：只做 feature-mask 最小版，再比较是否优于单纯 GRACE/GraphMAE-style 重构。
- 风险：目标权重敏感；decoder 复杂度可能影响公平性。
- 当前仓库适合度：中。可实现，但比 loss weighting 复杂。

### D5. Degree/Class-aware Hardness Reweighting

- 核心：根据 degree bucket、embedding uncertainty、train/val class rarity 对节点或 pair 动态加权。
- 来源：Focal Loss、GroupDRO、HAR、robust GCL。
- 预期验证：主表仍用平均 accuracy；额外 development-only 诊断 low-degree / minority-class 表现。
- 风险：若用标签过多，会偏离无监督 GCL；若只用无标签信号，难度估计可能不准。
- 当前仓库适合度：中高。适合作为辅助正则或 ablation。

### D6. Lightweight Teacher-Student GCL

- 核心：用 BGRL/SimSiam 式 teacher-student 替代负样本，再加 variance/covariance collapse guard。
- 来源：BYOL、SimSiam、DINO、BGRL、VICReg。
- 预期验证：对比 GRACE/BGRL/CCA-SSG，记录 embedding rank 和 variance。
- 风险：贡献容易被认为是 BGRL 变体；需要明确新点。
- 当前仓库适合度：中。可实现，但新颖性需要强约束。

### D7. LLM-assisted Semantic View for Text Graphs

- 核心：仅在 Cora/ogbn-arxiv 等文本属性图上，用固定 LLM embedding 或摘要作为额外 view。
- 来源：STAGE、GAugLLM、LangGSL。
- 预期验证：先做缓存特征 smoke/development，不进入主线 claim。
- 风险：非文本数据集不可用，API 不可复现，成本高。
- 当前仓库适合度：低到中。不建议作为第一阶段主 idea。

## 最小可验证实验

所有实验初始状态只能是 `development` 或 `pilot`，不得产生正式 claim。

| 实验 | 目标 | 数据集 | 对照 | 固定协议 | 最小输出 |
|---|---|---|---|---|---|
| E1 | 验证 false-negative weighting 是否改善 GRACE | Cora、CiteSeer、PubMed | GRACE 原版 | split seeds 0-9；logreg_val；同 encoder | 每 seed JSON、mean±std、pair-weight ablation |
| E2 | 验证 prototype positive 是否降低类内分散 | Cora、Computers、Photo | GRACE/GCA + no prototype | prototype 数 train/val 选择或无标签准则 | accuracy + embedding class compactness 诊断 |
| E3 | 验证 feature-space augmentation 是否替代输入扰动 | Cora、CiteSeer、Amazon | GRACE 输入增强 | 相同 epochs/hidden_dim/search budget | 单视图/双视图 ablation |
| E4 | 验证 masked-contrastive hybrid 是否优于单一目标 | Cora、PubMed、Wiki-CS | GRACE、GraphMAE-style reconstruction | decoder 轻量固定；loss weight 小网格由 val 决定 | objective ablation + collapse 指标 |
| E5 | 验证弱群体是否受益 | 已有主数据集 + degree buckets | 最佳 development 方法 vs baseline | 不改变主表；只 development 诊断 | low-degree/minority-class/worst-bucket 指标 |

## 风险清单

- **协议风险**：文献中 public split 或不同 evaluator 的数字不可直接与当前 `1:1:8` 结果比较。
- **leakage 风险**：任何 label-aware weighting、prototype selection、evaluator 参数选择都不能使用 test set。
- **复杂度风险**：多目标、多模块、多超参会让方法不公平；首轮 idea 应控制为单一机制。
- **新颖性风险**：BGRL/CCA-SSG/GraphMAE/GCMAE 已覆盖很多 obvious combination；GPT 生成 idea 时必须明确与这些工作的差异。
- **可复现风险**：LLM、KAN、大型 diffusion/teacher 方案会带来环境与预算不确定性。
- **claim 风险**：pilot/development 结果只能用于筛选 idea，不支持 SOTA、robust、comprehensive 等表述。

## 是否适合当前仓库实现

| 方向 | 适合度 | 原因 | 建议 |
|---|---|---|---|
| False-negative-aware weighting | 高 | 只改 loss，直接基于 GRACE/GCA | 第一优先 |
| Prototype positive expansion | 中高 | 需要聚类/原型，但工程可控 | 第二优先 |
| Feature-space augmentation | 高 | COSTA 类机制轻量，适合现有 encoder | 第一或第二优先 |
| Masked-contrastive hybrid | 中 | 需要 decoder 和目标权重 | 作为第二波 |
| Hardness / degree-aware reweighting | 中高 | 可先做诊断再做方法 | 辅助机制 |
| Teacher-student negative-free | 中 | 易与 BGRL/SimSiam 重叠 | 需要强新颖点 |
| LLM-TAG semantic view | 低中 | 只适用于文本图，复现成本高 | 暂不做主线 |
| KAN / 大架构替换 | 低 | 预算和新颖性解释压力大 | 暂缓 |

## 给 GPT Idea Creator 的约束

1. 只生成方法 idea，不写性能 claim。
2. 每个 idea 必须声明主要解决哪个 gap：G1-G8。
3. 每个 idea 必须给出最小实现路径、最小实验和失败判据。
4. 任何 label-aware 机制只能使用 train/val；test set 只用于最终报告。
5. 默认优先兼容现有 GRACE/GCA 脚本与 frozen Logistic Regression evaluator。
6. 首轮 idea 不建议使用 LLM、KAN 或大规模外部预训练，除非它们作为 clearly optional extension。
