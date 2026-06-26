# SIVA-GCL Novelty Check Report

生成时间：2026-06-26T11:16:23Z  
Reviewer: fresh `gcl_scientific_reviewer`  
Trace: `.aris/traces/novelty-check/2026-06-26_run05/`  
阶段：deep novelty / method review；无实验 claim

## Proposed Method

SIVA-GCL 修正版不再以 virtual hard negative 为主贡献，而是收缩为 **SIVA-positive core**：用 masked-context semantic stability critic 约束 node-local ego intervention search，主动生成“尽可能不同但语义稳定”的 positive view，让 positive pair 在 message passing 后仍有有效学习信号，从源头减少对真实节点 hard negatives 的依赖。

## Reviewer Verdict

Overall verdict: `REVISE`。  
Novelty score: `6.7/10`。  
Confidence: `0.70`。

Reviewer 认为 SIVA 比 BOND 更有论文形态，但当前不能 full GO。必须删除模块拼接感，将 virtual-negative 降级为 late ablation，主线只保留：

```text
semantic stability critic + intervention-positive search
```

## Closest Prior Work

| Prior | Overlap | Revised SIVA Difference | Risk |
|---|---|---|---|
| CGC | counterfactual hard negatives | SIVA 主贡献不再是 hard negative，而是 intervention positive | virtual-negative 若主打会被覆盖 |
| BalanceGCL | semantic positives + balanced hard negatives | SIVA 是 node-local、无标签、anti-prealignment positive | full SIVA 三模块像节点版 BalanceGCL |
| RGCL | rationale-aware semantic view | SIVA 最大化非语义干预，而不是保留 rationale | semantic-preserving view 已拥挤 |
| ACGA | adversarial positive/hard negative generation | SIVA 不主打 adversarial generator | 生成正负图组合风险高 |
| DoG | synthetic graph structures | SIVA 只生成 training views，不扩增图 | diffusion 版本会撞 DoG |
| SPGCL | positive pre-alignment | SIVA 直接生成 anti-prealignment positives | 必须提供 positive signal metrics |
| GraphMAE | masked reconstruction | critic 只是约束器，不是收益主来源 | critic-only 若有效则退化 |

## Required Revision

- 保留：critic-constrained maximal semantic-preserving intervention positive。
- 删除/降级：virtual-negative 不作为主贡献，只作后期 ablation。
- 改写问题：从“生成 hard negatives”改为“让 positives 在 message passing 后仍然 informative”。
- 增加机制指标：post-GNN positive similarity、positive gradient norm、view distance/stability curves。
- 增加强对照：GraphMAE-only、critic-shuffled、random-stable-intervention。

## Recommendation

让 SIVA-positive core 进入 smoke planning。BOND 不再作为主线，只保留为 loss-side baseline/ablation。当前不允许 formal 或性能 claim。

