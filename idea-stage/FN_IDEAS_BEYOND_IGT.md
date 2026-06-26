# False-Negative GCL Ideas Beyond IGT-GCL

生成时间：2026-06-26T09:21:55Z  
子流程：`/idea-creator`  
方向：图对比学习中的假负样本问题，用于普通节点分类  
状态：idea generation only；未改代码、未跑实验、未产生性能 claim

## 输入与边界

本轮已读取并吸收以下项目文件：

- `AGENTS.md`
- `BENCHMARK_PROTOCOL.md`
- `idea-stage/IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `literature/LIT_REVIEW_GCL_FALSE_NEGATIVES.md`
- `literature/FALSE_NEGATIVE_GAP_MAP_FOR_GPT_IDEA_CREATOR.md`
- `literature/LIT_REVIEW_GCL_CORE.md`
- `literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`
- `research-wiki/gap_map.md`

补充快速核查的外部边界包括 ProGCL、KDD 2024 Node Similarity、Affinity Uncertainty HNM、DGCL-PU / PUCL、E2Neg，以及 2026 年关于 positive samples / message passing pre-alignment 的新近 arXiv 线索。外部核查只用于 closest-prior 风险标注，不构成本项目 claim。

## IGT-GCL 作为低新颖性反例

当前 IGT-GCL 的核心是 lower/upper interval pair target：高 lower bound 做 conservative positive，低 upper bound 做 reliable hard true negative，模糊区间做 abstention。fresh reviewer 给出的 novelty score 为 `5.5/10`，主要问题是它容易被看成：

- point posterior / node similarity 的双阈值版本；
- ProGCL / NodeSim / PU-GCL / uncertainty weighting 的 set-valued 包装；
- negative downweight、positive expansion、abstention 的组合消融。

因此本轮生成刻意避开以下低新颖性路径：

- 不再以 `P(y_i = y_j)`、true-negative probability 或 node-similarity posterior 作为核心贡献；
- 不再把 interval / confidence band / lower-upper bound 作为主机制；
- 不把 abstention、过滤或普通降权写成主贡献；
- 不把 “hardness = embedding similarity” 或 “similarity * reliability” 作为新方法；
- 不把 train/val label calibration 伪装成无监督 GCL。

## 设计原则

新的 idea 优先从训练目标和表示几何层面绕开 false negative，而不是继续给每个 pair 贴标签：

1. 把节点分类需要的 class-semantic subspace 与 instance-uniformity subspace 解耦。
2. 增加非平凡 positives，缓解 positive scarcity，而不是只修 denominator。
3. 把 negative selection 变成结构覆盖/冲突选择问题，而不是 pair reweighting。
4. 用 prototype / transport / masked completion / spectral decomposition 替代 instance-level all-vs-all repulsion。
5. 所有机制必须能接入当前 GRACE/GCA 基线，并遵守 frozen encoder + Logistic Regression evaluator。

## 候选 Idea 1: DSR-GCL

**名称**：DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning  
**推荐级别**：Top 1  
**主要 gap**：G2, G3, FN-G1, FN-G2, FN-G8

**机制**：

1. 将 projection head 分成两个分支：`z_sem` 和 `z_res`。
2. `z_sem` 只接收 positive-only / masked-completion / VICReg-style collapse guard，不参与 node-node negative repulsion。
3. `z_res` 继续使用 InfoNCE 或小规模 negative objective，负责 instance uniformity、局部去冗余和避免全局 collapse。
4. 下游 Logistic Regression 主评估优先使用 `z_sem`，并把 `concat(z_sem, stopgrad(z_res))` 作为诊断版本。

**避开 IGT 的点**：

不判断任何 pair 是 positive / false negative / true negative，也不做 posterior、interval、abstention 或 sample reweighting。核心转向是：把 false-negative repulsion 从分类语义子空间里移走。

**Closest prior risk**：

BGRL、CCA-SSG、Graph Barlow Twins、VICReg 已覆盖 negative-free / redundancy reduction；SPGCL 提醒 message passing 会让 positive similarity 变得平凡；Decoupled SSL / representation scattering 可能覆盖部分“解耦表示”叙事。必须把贡献收缩为“semantic branch 不承受 node-node repulsion，residual branch 独立承担 uniformity”的节点分类目标对齐。

**仓库落地方式**：

基于 GRACE/GCA encoder 不变，只替换 projection head 与 loss：

- 新增 semantic projector 和 residual projector；
- semantic loss 使用 cross-view positive alignment + variance/covariance guard；
- residual loss 使用现有 GRACE InfoNCE，但 residual 分支不直接进入主 downstream 表征；
- 记录分支 variance、rank、alignment/uniformity、LogReg on `z_sem` / `z_res` / concat。

**必须消融**：

- vanilla GRACE/GCA；
- BGRL/CCA-style negative-free branch only；
- InfoNCE residual branch only；
- single projection head with same parameter count；
- `z_sem` receives negatives；
- `z_res` detached vs not detached；
- downstream 使用 `z_sem`、`z_res`、concat；
- matched hidden dimension / projection dimension 控制参数量。

**Kill rule**：

若 `z_sem` collapse、`z_sem` 单独不比 BGRL/CCA-style baseline 更有诊断优势，或 full DSR 与同参数 negative-free baseline 无差异，则 kill 主 idea。若提升只来自更大 projection head，则 kill。

## 候选 Idea 2: CNG-GCL

**名称**：CNG-GCL / Conflict-Negative Graph Selection for GCL  
**推荐级别**：Top 2  
**主要 gap**：FN-G1, FN-G3, FN-G4, FN-G8

**机制**：

1. 对每个 anchor 建一个 anchor-local candidate negative pool。
2. 在候选负样本之间建立 conflict graph：如果两个候选在结构、特征、EMA 表示或邻域上高度耦合，则它们可能来自同一语义区域，不能同时作为有效负样本集合。
3. negative set 不是所有节点，也不是按权重降权，而是求一个小规模、多样、低拓扑耦合的 approximate independent set / coverage set。
4. 只用被选中的少量 negatives 进入 denominator，且记录 selected negatives 的拓扑耦合、degree/class bucket 和 label-only offline purity 诊断。

**避开 IGT 的点**：

不为 anchor-pair 输出 posterior 或 interval，也不对每个 pair 做连续权重；核心是组合选择：哪些 negatives 共同构成一个低冲突、高覆盖的训练环境。

**Closest prior risk**：

E2Neg 已经挑战“大量 negatives 必要”并主张少量高质量、非拓扑耦合 negatives；B2-Sampling、ProGCL、cluster-refined negative sampling 也触及负采样质量。新颖性必须落在 anchor-local conflict graph + independent-set coverage，而不是“小负样本集合”本身。

**仓库落地方式**：

基于 GRACE/GCA loss 侧添加 sampler：

- 预计算或动态维护 top-k candidate negatives；
- 构造候选间 conflict score，例如共享邻居、PPR overlap、feature cosine、EMA rank overlap；
- 使用 greedy maximal independent set / submodular coverage 选取每个 anchor 的 `m` 个 negatives；
- InfoNCE 仍保留原公式，但 denominator 只使用 selected set。

**必须消融**：

- random `m` negatives；
- hardest `m` negatives；
- E2Neg-style representative negatives；
- ProGCL-style true-negative weighting；
- no conflict edges，只做 diversity；
- conflict score 单信号 vs 多信号；
- selected set size `m` sensitivity。

**Kill rule**：

若 random small negatives 或 E2Neg-style representative selection 匹配 full CNG，说明 conflict graph 不是核心，kill。若 selected negatives 在 label-only offline 诊断中并不比 hard negatives 更少 false negative，则 kill hard-negative 叙事。

## 候选 Idea 3: MPC-GCL

**名称**：MPC-GCL / Masked Positive Completion for Graph Contrastive Learning  
**推荐级别**：强备选  
**主要 gap**：G3, G4, FN-G2, FN-G3

**机制**：

1. 对 anchor 节点构造多个 masked context views：mask feature 子集、局部边、邻居特征或 ego-context。
2. 用轻量 decoder 从上下文补全 anchor 的表示或特征残差，得到多个 completed positives。
3. 训练时主要对齐原始 anchor 表示与 completed positives，而不是依赖其他节点作为 negatives。
4. 为防止 trivial reconstruction，加入 variance/covariance guard 或小规模 prototype-level contrast，不使用全节点 denominator。

**避开 IGT 的点**：

不处理“这个 pair 是不是假负样本”的判别问题，而是把正样本从两个随机增强视图扩展为多个语义补全视图，直接缓解 positive scarcity 和 InfoNCE 目标错位。

**Closest prior risk**：

GraphMAE、GraphMAE2、MaskGAE、GCMAE、CORE 已覆盖 masked graph modeling / contrastive reconstruction；SPGCL 新近强调 positive samples 比 negative refinement 更关键。MPC 的风险是被视作 GraphMAE + GRACE 拼接，必须保持机制简洁并证明 completed positives 是核心。

**仓库落地方式**：

- 在 GRACE/GCA encoder 后增加轻量 decoder；
- 使用 masked feature / masked ego-context 产生 completed views；
- loss = completed-positive alignment + reconstruction auxiliary + collapse guard；
- downstream 仍用 frozen encoder + Logistic Regression。

**必须消融**：

- GRACE/GCA；
- GraphMAE-style reconstruction only；
- positive alignment only without decoder；
- random extra positives；
- no mask diversity；
- no covariance/variance guard；
- decoder capacity matched control。

**Kill rule**：

若收益完全来自 reconstruction 或 decoder 容量，而 completed-positive alignment 无独立贡献，则 kill。若 completed positives 与普通 feature masking 无差异，则降级为 GraphMAE/augmentation ablation。

## 候选 Idea 4: ROT-GCL

**名称**：ROT-GCL / Regularized Optimal-Transport Assignment GCL  
**推荐级别**：备选  
**主要 gap**：G3, FN-G2, FN-G5

**机制**：

1. 用 prototypes / codebook 表示跨视图语义锚点。
2. 对两个增强视图的节点表示做 entropy-regularized optimal transport assignment。
3. 训练 swapped assignment prediction，而不是把其他节点全部推远。
4. 加入 capacity constraint、degree-aware quota 或 entropy floor，避免高 degree / 大类节点吞掉所有原型。

**避开 IGT 的点**：

没有 pair-level posterior、interval 或 false-negative filtering；节点可以共享 prototype，不会因为同类节点是“其他节点”就被直接推远。

**Closest prior risk**：

SwAV、DINO、PCL、LR-GCL、AFGRL 和 prototype GCL 都是强先验。ROT 的新意必须是 node classification 下的 balanced transport + false-negative-safe replacement for instance discrimination，而不是普通 clustering。

**仓库落地方式**：

- 添加 prototype matrix；
- 每个 epoch 或 batch 用 Sinkhorn / balanced assignment；
- loss 为跨视图 assignment prediction；
- 保留 GRACE/GCA augmentations，先不改 encoder。

**必须消融**：

- no capacity balance；
- k-means pseudo-label objective；
- prototype contrast without OT；
- BGRL / CCA-SSG；
- prototype 数敏感性；
- degree-aware quota vs uniform quota。

**Kill rule**：

若结果高度依赖 prototype 数或 capacity 超参，或 minority / low-degree 诊断显示被原型吞并，则 kill。若普通 SwAV-style assignment 匹配 full ROT，则新意不足。

## 候选 Idea 5: GO-GCL

**名称**：GO-GCL / Gradient-Orthogonalized Graph Contrastive Learning  
**推荐级别**：高风险备选  
**主要 gap**：G2, G3, FN-G1

**机制**：

1. 对 InfoNCE 中 negative denominator 产生的 repulsive gradient 做显式分解。
2. 用 positive alignment、EMA teacher 或 masked context 估计 anchor 的 semantic tangent direction。
3. 将 negative gradient 中与 semantic tangent 对齐的分量投影掉，只保留 residual/uniformity 方向的 repulsion。
4. 目标不是减少某些样本权重，而是改变 false-negative repulsion 影响表示空间的方向。

**避开 IGT 的点**：

不估计 pair 是否假负，也不做过滤/降权/区间；所有 negatives 仍可参与，但其梯度不能破坏正样本语义方向。

**Closest prior risk**：

可能被 reviewer 归入 gradient surgery、SAM、representation scattering 或 simple gradient clipping。必须用 ablation 证明它不是降低 denominator 强度，而是正交化语义破坏分量。

**仓库落地方式**：

- 在 loss 侧或 autograd hook 中获得 positive gradient basis；
- 对 negative gradient 做投影；
- 记录 projected gradient mass、semantic/residual gradient angle；
- 初版只在小图 full-batch GRACE 上做，避免复杂 mini-batch。

**必须消融**：

- ordinary negative downweight；
- gradient clipping；
- random projection basis；
- positive basis stopgrad vs no stopgrad；
- no projection；
- projection only after warmup；
- gradient-angle diagnostics。

**Kill rule**：

若 ordinary downweight / clipping 匹配 full GO，kill。若计算成本过高、训练不稳定，或 semantic tangent 无法稳定估计，则转为 diagnostic paper，不作为主方法。

## 候选 Idea 6: SHELL-GCL

**名称**：SHELL-GCL / Semantic-Shell Counterfactual Negatives  
**推荐级别**：高风险备选  
**主要 gap**：FN-G1, FN-G6

**机制**：

1. 不从真实节点中采 hard negatives，而是在 anchor 周围生成 semantic shell negatives。
2. shell negative 沿 residual / non-reconstructable / non-semantic 方向扰动，避免把真实同类节点作为 hard negative 推远。
3. 用 masked decoder 或 EMA teacher 检查扰动仍在数据流形附近，但不保留 anchor 的可补全语义。
4. denominator 使用 synthetic shell negatives 与少量真实 easy negatives。

**避开 IGT 的点**：

直接改变 negative 来源，不判断真实 pair 是否 false negative，也不做 sample weighting。

**Closest prior risk**：

Counterfactual hard negatives for GCL、Hard Negative Mixing、AdCo、Khan-GCL 都是近邻。SHELL 必须比“embedding mixup hard negative”更有图语义约束，否则新颖性弱。

**仓库落地方式**：

- 在 embedding 或 hidden feature 空间生成 perturbation；
- perturbation 方向来自 residual branch、feature mask residual 或 low-confidence decoder residual；
- 增加 manifold regularizer 和 radius schedule；
- 不引入 KAN 或大生成器。

**必须消融**：

- real hard negatives；
- random Gaussian perturbation；
- Hard Negative Mixing-style interpolation；
- no manifold regularizer；
- shell radius sweep；
- residual direction vs semantic direction。

**Kill rule**：

若 synthetic negatives 离流形太远、只等价于噪声正则，或需要大量半监督标签判断语义差异，则 kill。

## 候选 Idea 7: ENV-GCL

**名称**：ENV-GCL / Environment-Invariant Positive Learning for GCL  
**推荐级别**：备选  
**主要 gap**：G1, G2, FN-G4

**机制**：

1. 把不同 augmentation / propagation / mask policy 看成环境：feature-only、structure-only、diffusion、ego-mask、drop-edge 等。
2. semantic representation 只学习跨环境稳定的 positive signal。
3. environment-specific residual branch 吸收环境噪声和实例差异。
4. 使用 IRM / variance / covariance 风格约束，让线性可分的语义在多环境中保持一致。

**避开 IGT 的点**：

不对节点 pair 做假负判别，而是把 false negative/hard negative 混淆归因于单环境下的 spurious similarity，并通过跨环境不变性避免把它写入主语义表征。

**Closest prior risk**：

GCA、GraphCL、CSGCL、IRM、GroupDRO、RGCL 都触及环境/增强/鲁棒性。风险是方法变成多增强拼盘，必须把环境不变性写成单一核心。

**仓库落地方式**：

- 复用现有 augmentations；
- 增加多环境 batch；
- semantic branch 做 invariant alignment；
- residual branch 做 environment prediction 或 residual decorrelation 诊断。

**必须消融**：

- single environment；
- random environment labels；
- no residual branch；
- no invariant penalty；
- GRACE/GCA same augmentations without ENV objective；
- environment 数量敏感性。

**Kill rule**：

若环境划分不稳定、只靠更多 augmentations 提升，或 invariant penalty 与普通 covariance regularization 无差异，则 kill。

## 候选 Idea 8: FREQ-GCL

**名称**：FREQ-GCL / Frequency-Diversified Positive Graph Contrast  
**推荐级别**：备选  
**主要 gap**：G2, G3, FN-G2, FN-G4

**机制**：

1. 将节点特征经过 graph propagation 后分解为低频同配分量、中频边界分量和高频属性残差。
2. positives 不再只是同节点两增强视图，而是同节点跨频带的互补语义视图。
3. negative repulsion 只作用于高频/残差或 prototype level，避免低频同类结构被 instance discrimination 推散。
4. 针对 heterophily，可允许中高频分量承担更多分类信号。

**避开 IGT 的点**：

不进行 pair-level false-negative 判断；通过频带解耦让 positive signal 更非平凡，也让 node-node negatives 不直接破坏低频类别语义。

**Closest prior risk**：

SPGCL / separate propagation、HomoGCL、spectral GNN、Decoupled SSL 都是近邻。FREQ 的差异要落在 false-negative-safe contrast placement：哪个频带对齐，哪个频带允许 repulsion。

**仓库落地方式**：

- 添加简单 propagation decomposition：`X_low = S^k X`，`X_high = X - X_low`，中频可用 residual stack；
- projection head 分频带；
- loss = cross-band positive alignment + residual/prototype contrast；
- 记录频带 energy 与下游分支表现。

**必须消融**：

- no frequency split；
- low-only / high-only / mid-only；
- standard propagation positives；
- SPGCL-style separate propagation baseline；
- residual contrast off；
- heterophily fixed split diagnostic。

**Kill rule**：

若频带分解不稳定、只在强同配图有效，或 SPGCL-style baseline 匹配 full FREQ，则 kill 主线。

## 候选 Idea 9: MIB-GCL

**名称**：MIB-GCL / Mutual-Information Budgeted Graph Contrast  
**推荐级别**：理论/诊断备选  
**主要 gap**：G3, FN-G1, FN-G8

**机制**：

1. 将普通 InfoNCE 视为过度保留 node identity 的目标，而节点分类只需要 class-relevant information。
2. 对 representation 施加 instance-identity MI budget：保留跨视图、跨环境稳定信息，限制局部邻域内可区分每个节点身份的能力。
3. uniformity 不再由“推开所有其他节点”承担，而由 covariance/rank/entropy 约束承担。
4. 通过 offline 诊断观察 class compactness、node identity probing、degree bucket 表现。

**避开 IGT 的点**：

完全不构造 pair target。它把问题重定义为 InfoNCE 过度实例判别与节点分类目标错位。

**Closest prior risk**：

InfoMin、VICReg/Barlow Twins、negative-free SSL、representation scattering、SPGCL 都相关。风险是理论叙事强、方法实现弱。

**仓库落地方式**：

- 新增 node-identity probe 作为 diagnostic-only；
- loss 使用 positive alignment + covariance/rank + local identity adversarial/dropout；
- 初版不使用标签，标签只做 offline 诊断。

**必须消融**：

- covariance/rank only；
- adversarial identity budget off；
- local vs global identity budget；
- BGRL/CCA-SSG；
- identity probe accuracy 与 downstream relation。

**Kill rule**：

若 identity budget 与普通 dropout/covariance regularization 无差异，或 node identity probe 无法解释下游变化，则 kill。

## 候选 Idea 10: LCC-GCL

**名称**：LCC-GCL / Label-Calibrated Contrastive Constraints for Honest Semi-supervised GCL  
**推荐级别**：半监督备选，不建议主推为无监督 GCL  
**主要 gap**：FN-G7, FN-G8

**机制**：

1. 明确使用 `1:1:8` 中 train/val 标签，方法诚实标注为 semi-supervised auxiliary。
2. 只在有标签节点上学习一个小型 constraint layer：哪些结构/特征信号更可能保留类别语义。
3. 将 constraint layer 用于选择 augmentation policy 或 prototype capacity，而不是直接给所有 unlabeled pair 打 posterior。
4. val 只选择 constraint 超参；test 完全隔离。

**避开 IGT 的点**：

不做无标签 pair posterior / interval，也不声称自监督。它把少标签信息作为合法校准器，用于增强/原型约束而非 pair reweight。

**Closest prior risk**：

SupCon、NDSCL、semi-supervised GCL、PUCL、train/val calibrated debias 都很接近。新颖性较弱，但论文叙事诚实，适合作为 upper-bound 或 auxiliary variant。

**仓库落地方式**：

- 读取保存的 train/val split；
- 学习轻量 constraint scorer；
- 不允许使用 test label；
- 与无监督版本分表报告。

**必须消融**：

- unsupervised counterpart；
- train-only calibration；
- train+val calibration；
- random labels sanity check；
- direct SupCon baseline；
- constraint used for augmentation vs prototype vs loss。

**Kill rule**：

若无标签版失败、只有 label-calibrated 版有效，则不能作为无监督 GCL 主线。若 random-label sanity check 也有效，说明只是正则或容量效应，kill。

## Top 2 推荐

### Top 1: DSR-GCL

推荐原因：它从根上绕开 IGT 的低新颖性陷阱，不再判定 pair 语义，而是把 false-negative repulsion 从节点分类语义子空间中隔离出去。它能直接回应 “InfoNCE 与 node classification 目标错位” 这个更高层问题，同时工程上只需改 projection head 和 loss，适合当前 GRACE/GCA 仓库。

主要风险：容易被认为是 BGRL/CCA/VICReg 的分支组合。因此第一版必须把“semantic branch 不承受 negatives、residual branch 独立承担 uniformity”作为唯一核心，并严格做同参数消融。

### Top 2: CNG-GCL

推荐原因：它仍保留 InfoNCE 框架，落地轻量，但不走 IGT 的 posterior/interval/reweighting 路线。它把 hard-negative mining 改写为 anchor-local conflict graph selection，可直接对 false negative 与 hard negative 混淆给出结构化解释。

主要风险：E2Neg 已经非常接近“小规模高质量 negatives”的方向。CNG 必须证明 conflict graph 的组合结构比 random small negatives、E2Neg representative negatives 和 ProGCL-style weighting 更有诊断价值，否则应快速 kill。

## 当前阶段决策

**Decision: GO_FOR_TOP2_REFINEMENT.**

本轮只完成 idea generation 与文档同步。未实现代码、未跑 smoke/pilot/development/formal 实验、未产生性能 claim。下一步若继续，应只对 DSR-GCL 和 CNG-GCL 做 `/research-refine` 或 novelty-check；其余候选进入 backlog，不应直接实现。
