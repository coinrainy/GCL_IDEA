# CAST Certificate Pilot-A Results

- 时间：2026-06-26T14:49:10Z
- Skill：`/experiment-bridge`
- 阶段：Pilot-A / diagnostic，不是 formal。
- 协议：stratified random `1:1:8` split；frozen GRACE encoder；Logistic Regression evaluator。
- 范围：Cora seeds 0-2、CiteSeer seeds 0-2、PubMed seed 0。
- 未完成项：PubMed seeds 1-2 未启动；原因是 Cora/CiteSeer 三 seed 与 PubMed seed0 均未支持继续扩大。

## Aggregate Results

### Cora (3 seeds)

| ID | Variant | Acc mean | Acc std | Agreement | kNN ov | PPR ov | CAST ov | pcorr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | GRACE | 82.76 | 1.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1101 |
| C1 | embedding_knn | 84.23 | 1.16 | 0.8138 | 1.0000 | 0.3974 | 0.2803 | 0.1101 |
| C2 | cast_proxy | 83.79 | 2.14 | 0.7512 | 0.2803 | 0.2001 | 1.0000 | 0.1101 |
| C3 | cast_candidate_pool_only | 83.79 | 2.14 | 0.7512 | 0.2803 | 0.2001 | 0.9996 | 0.1101 |
| C6 | ppr_diffusion_positive | 83.55 | 1.39 | 0.7549 | 0.4286 | 1.0000 | 0.2158 | 0.1101 |
| C4 | latent_target_certificate | 83.16 | 2.43 | 0.7492 | 0.3137 | 0.2072 | 0.1743 | 0.1101 |
| C5 | certificate_plus_cast_score | 82.73 | 2.94 | 0.7774 | 0.3431 | 0.2349 | 0.7418 | 0.1101 |

### CiteSeer (3 seeds)

| ID | Variant | Acc mean | Acc std | Agreement | kNN ov | PPR ov | CAST ov | pcorr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | GRACE | 71.25 | 0.60 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0575 |
| C1 | embedding_knn | 71.85 | 1.10 | 0.6530 | 1.0000 | 0.3432 | 0.3851 | 0.0575 |
| C2 | cast_proxy | 71.99 | 0.36 | 0.6499 | 0.3851 | 0.2705 | 1.0000 | 0.0575 |
| C3 | cast_candidate_pool_only | 71.99 | 0.36 | 0.6499 | 0.3851 | 0.2705 | 0.9998 | 0.0575 |
| C6 | ppr_diffusion_positive | 71.39 | 0.51 | 0.6823 | 0.4893 | 1.0000 | 0.3857 | 0.0575 |
| C4 | latent_target_certificate | 71.60 | 1.05 | 0.6258 | 0.2590 | 0.1458 | 0.1754 | 0.0575 |
| C5 | certificate_plus_cast_score | 71.77 | 0.09 | 0.6586 | 0.4278 | 0.2837 | 0.6939 | 0.0575 |

### PubMed (seed 0 only)

| ID | Variant | Acc | Agreement | kNN ov | PPR ov | CAST ov | pcorr |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | GRACE | 85.13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0677 |
| C1 | embedding_knn | 85.67 | 0.8127 | 1.0000 | 0.1441 | 0.1867 | 0.0677 |
| C2 | cast_proxy | 85.62 | 0.8007 | 0.1867 | 0.0760 | 1.0000 | 0.0677 |
| C3 | cast_candidate_pool_only | 85.62 | 0.8007 | 0.1867 | 0.0760 | 1.0000 | 0.0677 |
| C6 | ppr_diffusion_positive | 85.10 | 0.7580 | 0.1451 | 1.0000 | 0.0765 | 0.0677 |
| C4 | latent_target_certificate | 85.22 | 0.7772 | 0.0865 | 0.0422 | 0.0885 | 0.0677 |
| C5 | certificate_plus_cast_score | 85.47 | 0.8031 | 0.1914 | 0.0775 | 0.7295 | 0.0677 |

## Diagnostic Reading

- C4 latent certificate has low overlap with kNN/PPR/CAST on all checked datasets, especially PubMed seed0 (`kNN 0.0865`, `PPR 0.0422`, `CAST 0.0885`). This supports the narrow diagnostic statement that C4 is not merely selecting the same pairs as kNN/PPR/CAST.
- That novelty diagnostic did not translate into stable accuracy improvement. C4/C5 are below the strongest mining controls on Cora, CiteSeer, and PubMed seed0.
- C5 often inherits high CAST overlap (`0.7418` on Cora, `0.6939` on CiteSeer, `0.7295` on PubMed seed0), so the hybrid variant is harder to defend as a clean certificate mechanism.

## Decision

`REVISE_OR_PIVOT_BEFORE_ANY_MORE_PILOT`

Do not run PubMed seeds 1-2, Pilot-B, formal 10-seed experiments, paper tables, or `/auto-review-loop` from this state. The next useful action is not more seeds; it is a mechanism revision that explains why low-overlap certificate positives fail to improve classification.

No SOTA, robust, comprehensive, or final performance claim is supported.
