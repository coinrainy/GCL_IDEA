# WILLOW-GCL Novelty Check

日期：2026-06-26  
Reviewer：fresh `gcl_scientific_reviewer`  
Verdict：`REVISE`  
Novelty：`7.0/10`  
Confidence：`0.68`

## Reviewer Summary

WILLOW 比 SIVA 更接近 2026 paper 主 idea，但还不是 `GO`。它必须收缩为：

```text
latent context-target predictive certificate
        +
certified intervention positive search for GCL
```

单独的 world model 会撞 Graph-JEPA / Predict-Cluster-Refine；单独的 intervention search 会撞 SIVA / GCA / RGCL / ACGA。只有把 latent target prediction error 用作 semantic certificate，再去搜索 hard positive views，才构成目前最强的差异点。

## Score Comparison

| Idea | Novelty | Confidence | Role |
|---|---:|---:|---|
| WILLOW-GCL | 7.0 | 0.68 | active mainline candidate |
| SIVA-GCL-positive-core | 6.6-6.7 | 0.70-0.72 | mandatory control |
| BOND-GCL | 5.8-6.0 | 0.74-0.76 | archived loss-side baseline |

## Blocking Weakness

`cert_error <= epsilon` 目前没有证明等价于节点分类语义保持。它可能只学习 degree、homophily、feature smoothness 或 teacher shortcut。必须用 label-agreement diagnostic、certificate-shuffled control、matched random hard positive control 验证。

## Required Revisions

1. 不泛称 world model；主文使用 **latent ego target-prediction certificate**。
2. 定义 context/target ego sampling、EMA target、collapse prevention、epsilon 选择。
3. 删除 virtual negative；主线只做 certified hard positive。
4. 固定 intervention search space 和 search budget。
5. 强制加入 Graph-JEPA-only、SIVA reconstruction-critic、certificate-shuffled controls。
6. 叙事聚焦 GCL positive signal / false-negative exposure。

## Decision

`REVISE_TO_WILLOW_SMOKE_PLANNING`。本报告不包含实验结果，不支持 formal 或性能 claim。
