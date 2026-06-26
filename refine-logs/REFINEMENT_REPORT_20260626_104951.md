# Refinement Report: BOND-GCL

**Problem**：false-negative repeated repulsion in graph contrastive learning for node classification  
**Date**：2026-06-26  
**Final Score**：6.0 / 10  
**Final Verdict**：REVISE

## Output Files

- Candidate brief: `idea-stage/FN_BASIN_CANDIDATES_20260626_104951.md`
- Novelty check: `idea-stage/BOND_GCL_NOVELTY_CHECK.md`
- Idea report: `idea-stage/IDEA_REPORT.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Feasibility | Validation Focus | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 6 | 7 | 8 | 6.0 | REVISE |

## Method Evolution Highlights

1. 从 pair false-negative risk 转为 basin-level repeated negative mass。
2. 将 overflow / boundary replacement 从主机制移为 optional ablation。
3. 将 novelty proof 绑定到 shuffled basins、same pair weights、E2Neg-style center controls。

## Pushback / Drift Log

| Reviewer Said | Author Response | Outcome |
|---|---|---|
| BOND 可能只是 reweighting | 加入 same pair weights / shuffled basins 反事实 | accepted |
| overflow 没有原则性 | overflow 不再绑定主机制 | accepted |
| basin construction 可能吃掉贡献 | 首版只保留一个主构建器和 random-basin 对照 | accepted |

## Remaining Weaknesses

BOND 仍没有任何实验支持。当前只能作为 smoke implementation target，不能作为论文 claim。若 cap-only 不能优于 pair-weight proxy 或 random basin，应快速 PIVOT。

