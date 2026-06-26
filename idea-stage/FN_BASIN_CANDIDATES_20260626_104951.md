# 图对比学习假负样本新候选：Basin / Budget 方向

生成时间：2026-06-26T10:49:51Z  
子流程：`/idea-discovery`  
方向：普通节点分类场景下的图对比学习假负样本问题  
状态：candidate brief for fresh reviewer；未实现代码、未跑新实验、未产生性能 claim

## 本轮边界

本轮目标是重新寻找假负样本方向中更可能在 2026 年形成论文、且工程上更可能看到提升的方案。必须遵守项目协议：

- 主任务是节点分类。
- 主线是图对比学习。
- 主评估为 frozen encoder + Logistic Regression。
- 默认主 split 是 stratified random `1:1:8`，formal 前必须固定 seeds 0-9。
- smoke / pilot / development 结果不能写成 SOTA、robust 或 comprehensive claim。

本轮明确不继续以下旧线：

- BCE/CER-GCL：已由用户要求放弃并归档。
- IGT-GCL：fresh reviewer 给出低新颖性风险，容易被视为 posterior / interval / abstention / pair reweighting 包装。
- DSR-GCL：Cora seed=0 fix-audit-smoke 中 A9 fixed DSR-full `69.05`，明显低于 GRACE `84.78`、A4b `81.96`、A5b/A5c 控制；当前不适合作为追求性能提升的主线。

## 新近文献边界

本轮结合已有 `literature/LIT_REVIEW_GCL_FALSE_NEGATIVES.md` 与 2026-06-26 新检索：

| 论文 | 年份 | 关键点 | 对本轮约束 |
|---|---:|---|---|
| ProGCL: Rethinking Hard Negative Mining in GCL | 2022 / ICML | 图上最相似 negatives 往往是假负，需估计 true-negative 概率 | 不能直接把 similarity 当 hardness |
| AUGCL: Affinity Uncertainty-based HNM | 2024 / TNNLS | 用 collective affinity uncertainty 进入 GCL loss，形成 adaptive-margin 解释 | uncertainty-weighted negative 已是强 prior |
| Enhancing GCL with Node Similarity | 2024 / KDD | 从理想目标出发，同时处理 all positive / no false-negative | pair similarity soft target 已是强 prior |
| E2Neg: Does GCL Need a Large Number of Negative Samples? | 2025 / AAAI | 用少量有效 negatives / cluster-center 风格采样即可提升，反对大 denominator | 小 negative set 本身不新，必须比 E2Neg 多一层机制 |
| Khan-GCL | 2025 / arXiv / AAAI 2026 附近 | 生成 semantically meaningful hard negatives，但换 KAN encoder 且偏图级任务 | 不宜首轮做重架构或生成器 |
| SPGCL: Revisiting Positive Samples in GCL | 2026 / arXiv | message passing 造成 positive pre-alignment；按 Dirichlet energy 分离传播并做 energy-guided positive sampling | 仅修 negatives 不够，positive signal 与 propagation 也要被考虑 |
| On the Similarities of Embeddings in CL | 2025 / arXiv | 分析 negative-pair similarity variance 与 batch/full-batch 设置 | denominator 几何和相似度分布可作为方法/诊断切口 |

## 设计原则

1. 优先做 loss / sampler 插件，不换 backbone，不引入 KAN/LLM/大生成器。
2. 目标不是“猜 pair 标签”，而是减少 false-negative repulsion 的总质量和重复计数。
3. 必须保留 sufficient negative pressure，避免过滤过多导致 uniformity 下降。
4. 必须能与 GRACE/GCA 现有训练脚本快速接入，并可在 Cora seed=0 先做 smoke。
5. 必须直接和 E2Neg、ProGCL、AUGCL、NodeSim 做 closest-prior 对照。

## 候选 Idea 1：BOND-GCL

**名称**：BOND-GCL / Basin-capped Objective for Negative Debiasing in Graph Contrastive Learning  
**推荐级别**：Top 1  
**核心问题**：InfoNCE denominator 把每个“其他节点”都当成独立负样本；在同配或语义簇密集图中，同一语义盆地里的大量节点会对 anchor 产生重复排斥，其中很多是 false negatives 或 near-positives。  

**方法做什么**：

1. 对每个 anchor，用稳定的低成本信号构建若干 anchor-local basins：例如 PPR/邻域 overlap、feature cosine、EMA embedding kNN 的交集或小型聚类。
2. denominator 不再对每个 basin 内所有节点无限累加 negative mass，而是对每个 basin 的总 `exp(sim/tau)` 施加上限。
3. 被 cap 掉的 overflow mass 不直接删除，而是重分配给 low-affinity / high-residual-similarity 的 boundary representatives，保持 denominator scale 与 uniformity。
4. 训练目标仍是 GRACE/GCA 式 InfoNCE，只替换 denominator 聚合方式；encoder、augmentation、evaluator 不变。

**核心假设**：节点级 false negatives 的主要伤害不是单个错判 pair，而是同一语义盆地在 denominator 中被重复计数并持续推开；对 basin-level repulsive mass 设预算，比 pair-level posterior 更稳、更少调参，也比只用少量 cluster centers 更保留边界 negative pressure。

**与 closest prior 的差异**：

- vs ProGCL/AUGCL：不是给每个 anchor-negative pair 输出 true-negative 概率或 uncertainty 权重，而是约束 basin-level total repulsion。
- vs NodeSim：不是直接把相似节点变成 soft target，而是只控制 denominator 中重复排斥质量。
- vs E2Neg：不是只用少量 cluster centers；BOND 保留 full denominator 的信息，但给每个 basin 设置 mass budget，并将 overflow 转移到 boundary representatives。
- vs SPGCL：不主打 positive pre-alignment；但可把低能量/稳定信号仅用于 basin 构建，不改 message passing 主干。

**最小实现**：

- 输入：两视图 projection `z1/z2`、原始 graph `edge_index`、node features、EMA embedding。
- Basin 构建：每 `K` epochs 更新一次 anchor-local candidate graph；默认只取每个 anchor 的 top-`q` stable neighbors 和 random negatives。
- Loss 替换：`denom_i = pos_i + sum_b min(sum_{j in b} exp(s_ij/tau), cap_i,b) + overflow_to_boundary_i`。
- 超参：`q`, `num_basins`, `cap_ratio`, `boundary_m`，优先少量离散网格并由 val 决定。

**预期最小 smoke**：

- Cora seed=0, 200-500 epochs development smoke。
- 对照：GRACE 原版、random small negatives、hardest small negatives、E2Neg-style center negatives、ProGCL-style pair reweight。
- 成功信号：BOND 在 Cora seed=0 至少接近或超过 GRACE，同时 false-negative offline diagnostic 降低；如果只能靠大幅缩小 denominator 才提升，则不算机制成功。

**主要风险**：

- basin 构建可能退化为普通聚类或 NodeSim。
- cap_ratio 调参若过宽，可能被 reviewer 认为 search space 不公平。
- 在 Amazon/Photo 上 feature/PPR basin 可能错，需做同配依赖诊断。

## 候选 Idea 2：CNG-GCL v2

**名称**：CNG-GCL / Conflict-Negative Graph Selection with Basin Diagnostics  
**推荐级别**：Top 2 backup  
**方法做什么**：沿用已有 CNG 的 anchor-local conflict graph，但把目标从“选 independent set”收缩为“选低冲突、覆盖多 basin 的 small denominator”。  

**区别与风险**：

- 仍然很接近 E2Neg，需要证明 conflict graph 比 random small negatives / cluster centers / diversity-only 有独立贡献。
- 比 BOND 更工程轻，但论文新颖性可能偏弱。

## 候选 Idea 3：PEARL-GCL

**名称**：PEARL-GCL / Positive-Energy Alignment with Reliable Local Negatives  
**推荐级别**：备选  
**方法做什么**：借鉴 SPGCL 的 positive pre-alignment 发现，用高 Dirichlet-energy feature 产生 informative positives，用低 false-negative-risk basin 产生 local negatives；主贡献是把 positive informativeness 和 negative reliability 作为两个独立 gate。  

**风险**：非常容易被 SPGCL + ProGCL/AUGCL 夹击，新颖性需审查。

## 候选 Idea 4：RUNE-GCL

**名称**：RUNE-GCL / Repulsion Uniformity with Negative-mass Equalization  
**推荐级别**：备选  
**方法做什么**：不构造 basins，只对每个 anchor 的 negative similarity distribution 做 variance/effective-number regularization，避免少数高相似疑似假负支配 denominator。  

**风险**：可能退化为 generic similarity regularization，与 2025 embedding similarity theory 更接近，GCL 特色不足。

## 候选 Idea 5：SAFE-GCL

**名称**：SAFE-GCL / Split Agreement for False-negative Exposure  
**推荐级别**：诊断向备选  
**方法做什么**：对每个 anchor 比较 feature-only view、structure-only view、EMA view 对 negative 的一致性；只有跨视图都不同意“同语义”的节点才进入 hard negative pool。  

**风险**：本质仍是 conservative filter / reweighting，可能新颖性不足。

## 候选 Idea 6：LITE-SHELL

**名称**：LITE-SHELL / Lightweight Boundary Synthetic Negatives  
**推荐级别**：高风险备选  
**方法做什么**：不使用真实高相似节点做 hard negatives，而在 feature/embedding residual space 构造轻量 boundary virtual negatives。  

**风险**：CGC、Hard Negative Mixing、Khan-GCL 太接近；首轮不推荐。

## 初步排序

| Rank | Idea | 新颖性预估 | 工程难度 | 预期提升概率 | 首轮建议 |
|---:|---|---|---|---|---|
| 1 | BOND-GCL | 中高 | 中 | 中高 | 进入 fresh reviewer |
| 2 | CNG-GCL v2 | 中 | 低中 | 中 | backup |
| 3 | PEARL-GCL | 中 | 中 | 中 | 需查新 |
| 4 | RUNE-GCL | 中低 | 低 | 中 | 作为 ablation/diagnostic |
| 5 | SAFE-GCL | 中低 | 低 | 中低 | 作为 baseline |
| 6 | LITE-SHELL | 中 | 高 | 不确定 | 暂不主推 |

## 给 reviewer 的审查问题

1. BOND-GCL 是否真的区别于 ProGCL/AUGCL/NodeSim/E2Neg，还是 basin-level wording 的 pair weighting？
2. 如果要让 BOND 更像 2026 paper，应该保留哪些机制，删除哪些复杂度？
3. 在当前项目协议下，BOND 的最小 smoke / pilot 应该如何设计，才能快速判断是否值得投入？
4. 如果 BOND 不成立，Top backup 应该是 CNG v2、PEARL 还是重新生成？

