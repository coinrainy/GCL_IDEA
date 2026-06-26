# BOND-GCL 文献边界与 Gap 收束

生成时间：2026-06-26T10:49:51Z  
子流程：`/idea-discovery` Phase 1  
方向：图对比学习假负样本，用于普通节点分类  
状态：literature landscape；不包含实验结果，不产生性能 claim

## 结论

本轮 false-negative GCL 方向不建议继续 DSR-GCL，也不建议把普通 pair reweighting、interval / abstention、或 small negative sampling 包装成新方法。最有希望的新切口是：**把 InfoNCE denominator 中同一语义盆地的重复排斥质量作为可观测 failure mode 来建模**。

这一路线暂命名为 **BOND-GCL / Basin-capped Objective for Negative Debiasing in Graph Contrastive Learning**。它不是直接估计每个 pair 的 true-negative probability，而是给 anchor-local basin 的总 negative mass 设置预算，并用反事实对照证明机制不是普通 pair reweighting 或 E2Neg-style small-negative sampling。

## 必须面对的 closest prior

| 论文 | 年份 / 来源 | 与 BOND 的关系 | 对 BOND 的约束 |
|---|---|---|---|
| ProGCL: Rethinking Hard Negative Mining in GCL | ICML 2022 | 指出图上 hard negatives 容易是假负，并做 true-negative probability / mix | BOND 不能变成 coarser true-negative pair reweighting |
| AUGCL: Affinity Uncertainty-based HNM in GCL | arXiv 2023 / TNNLS 2024 | 用 affinity uncertainty 调节 hard negative 权重 | BOND 必须避免把 basin uncertainty 写成另一个 pair margin |
| Enhancing GCL with Node Similarity | KDD 2024 | 从 ideal objective 处理 all-positive / no-false-negative 问题 | BOND 不应直接把 node similarity soft target 当主贡献 |
| E2Neg: Effective and Efficient Negative Sampling | AAAI 2025 | 证明少量有效 negatives 可提升 | BOND 必须保留 full denominator 的 basin aggregation 语义，而不是只选 cluster centers |
| Khan-GCL | 2025 / AAAI 2026 附近 | 生成语义 hard negatives，并替换为 KAN encoder | BOND 首轮不走重架构或生成式 hard negatives |
| SPGCL: Revisiting Positive Samples in GCL | arXiv 2026 | 指出 message passing 导致 positive pre-alignment，并用 Dirichlet energy 修复 positive signal | BOND 需要承认 positive signal 风险，但主贡献不要被 SPGCL 吸走 |

## 本轮关键 gap

### Gap 1：pair false-negative risk 之外，还有 basin-level repeated repulsion

现有 ProGCL/AUGCL/NodeSim 更容易落在 pair-level 解释：某个 anchor-negative pair 是真负、假负、不确定，或者应被软化。BOND 的潜在新意是把 denominator 看成一个 **negative mass distribution**：如果一个语义盆地里有大量近似同类节点，它们即使单个权重不极端，总和也会形成过量排斥。

### Gap 2：small negative sampling 可能有效，但无法解释 full-denominator over-counting

E2Neg 已经覆盖“少量有效 negatives”这一强 prior。BOND 若只保留 basin representatives，会很容易被视为 E2Neg 变体。因此 BOND 必须形式化为 full-denominator aggregation operator：所有节点仍被观察和分组，但梯度质量按 basin 聚合与限幅。

### Gap 3：overflow / boundary mining 不能吞掉主贡献

将被 cap 掉的 mass 转给边界代表有助于维持 denominator scale，但 reviewer 指出它可能成为真正的性能来源。因此 overflow 必须是可关闭的模块，主 claim 先落在 `cap-only`。

### Gap 4：BOND 必须有反事实实验防止“只是 reweighting”

必须加入：

- same pair weights, shuffled basins；
- same cap histogram, random basin assignment；
- BOND-cap only vs pair-weight proxy；
- BOND-cap only vs E2Neg-style centers。

这些实验若不支持 BOND，方向应快速 PIVOT。

## 阶段决策

Decision: `REVISE_TO_BOND_GCL`.

BOND-GCL 可以进入 `/research-refine-pipeline` 风格的 proposal 和 experiment planning，但不能直接进入 full pilot 或 formal。下一步只允许 smoke / pilot implementation，且必须先验证：

1. loss 数值稳定；
2. denominator effective number 不崩；
3. offline false-negative mass exposure 下降；
4. random-basin / pair-weight controls 明显弱于 BOND。

