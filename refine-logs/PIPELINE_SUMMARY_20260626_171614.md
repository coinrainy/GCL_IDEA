# Pipeline Summary: ORBIT-GCL Idea Discovery

- 时间：2026-06-26T17:16:14Z
- Skill：`/idea-discovery`
- Direction：重新进行 graph contrastive learning idea discovery，要求可行性、创新性、完整论文骨架
- Final decision：`REVISE_TO_ORBIT_MINIMAL_SPEC_AND_KILL_SMOKE`

## What Changed

MEGA-GCL 已出现 Cora seed0 kill-smoke no-go 信号：MEGA full 低于 GRACE、GCMAE-style 和 matched GRACE，说明 masked evidence group prediction 不足以作为当前主线。

本轮将问题从 false-negative correction / masked evidence prediction 进一步 pivot 到 operator response learning。Top1 为 ORBIT-GCL。

## Outputs

- `literature/ORBIT_GCL_LANDSCAPE.md`
- `idea-stage/OPERATOR_RESPONSE_CANDIDATES.md`
- `idea-stage/ORBIT_GCL_NOVELTY_CHECK.md`
- `idea-stage/IDEA_CANDIDATES.md`
- `idea-stage/IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `research-wiki/ideas/orbit_gcl.md`

## Next Step

Implement only OR-M0 Cora seed0 kill-smoke:

```bash
python scripts/run_orbit_smoke.py --config configs/orbit_smoke.yaml --dataset Cora --seed 0 --run-tag or_m0_001
```

No Pilot-A/B/formal or performance claim is allowed before OR-M0 passes.
