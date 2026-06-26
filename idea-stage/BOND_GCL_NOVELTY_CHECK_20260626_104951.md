# BOND-GCL Novelty Check Report

生成时间：2026-06-26T10:49:51Z  
Reviewer: fresh `gcl_scientific_reviewer`  
Trace: `.aris/traces/novelty-check/2026-06-26_run04/`  
阶段：deep novelty / method review；无实验 claim

## Proposed Method

BOND-GCL 将节点级 GCL 的假负样本问题重述为 **denominator 中语义盆地重复排斥质量过量**。它对 anchor-local basin 的 total negative mass 施加预算，而不是估计每个 pair 的真假负概率；overflow / boundary replacement 只作为可关闭模块。

## Core Claims

1. **Basin repeated repulsion 是独立 failure mode**  
   Novelty: MEDIUM-HIGH。Closest: ProGCL / AUGCL / NodeSim。  
   风险：如果最终数学等价于 basin 内共享 pair weight，则会被视为粗粒度 reweighting。

2. **Basin-level mass aggregation differs from small negative sampling**  
   Novelty: MEDIUM。Closest: E2Neg。  
   风险：若实现只保留 representatives，就会退化为 E2Neg-style center sampling。

3. **Cap-only mechanism can reduce false-negative exposure without destroying uniformity**  
   Novelty: MEDIUM。Closest: DCL / ProGCL / similarity regularization。  
   风险：提升可能来自缩小 denominator 或调参。

4. **Random-basin / same-weight counterfactuals can isolate the mechanism**  
   Novelty: MEDIUM。  
   这是 BOND 进入 pilot 的必要实验设计，不是已有证据。

## Reviewer Verdict

Overall verdict: `REVISE`.

Novelty score for BOND-GCL: `6.0/10`。  
Confidence: `0.74`。

Reviewer 认为 BOND 是本轮最值得 refine 的候选，但当前版本不能直接 GO。必须将方法压缩成单一机制：`basin-level negative mass aggregation`，删除复杂多信号 basin 和绑定式 overflow 模块。

## Closest Prior Work

| Paper | Overlap | BOND Difference | Risk |
|---|---|---|---|
| ProGCL | false-negative-aware negative treatment | 不估计 pair true-negative probability，而限制 basin total mass | 若 cap 等价 pair weight，则差异不足 |
| AUGCL | uncertainty-aware hard negative modulation | basin budget，而非 pair uncertainty margin | basin 若依赖 affinity/uncertainty，差异变薄 |
| NodeSim | similarity-informed soft target | 不把相似节点直接变 positive，只控制重复排斥 | similarity basin 会被看作 NodeSim loss variant |
| E2Neg | small/effective negative selection | full denominator 可计算但按 basin 聚合 | representative-only 实现会退化 |
| SPGCL | energy-aware positive sampling | BOND 主处理 denominator | 需承认 positive pre-alignment，但不扩张主贡献 |

## Required Revision

- 保留 denominator 中语义盆地重复计数的 failure mode。
- 保留 basin-level mass budget，并形式化为 aggregation operator。
- 删除复杂多信号 basin；首版只保留一个主构建器和一个结构无关对照。
- overflow redistribution 改为 optional ablation，不能绑定主机制。
- 明确 full-denominator basin aggregation 与 E2Neg small-negative set 的差异。
- 加入 shuffled basins / same cap histogram / same pair weight 反事实实验。

## Recommendation

让 BOND-GCL 进入 refinement 和 smoke planning，但状态只能是 `REVISE_TO_SMOKE_PLANNING`。若 Cora seed=0 smoke 显示 BOND-cap-only 不能超过 random basin 或 pair-weight proxy，应立即 PIVOT，不进入多数据集 pilot。

