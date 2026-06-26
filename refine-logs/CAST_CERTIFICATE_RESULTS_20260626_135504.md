# CAST Certificate Smoke Results

## Scope

- Method: `CAST latent target-prediction certificate`
- Status label: `smoke`
- Dataset: `Cora`
- Seed: `0`
- Split: project stratified random `1:1:8`
- Evaluator: frozen GRACE encoder + `LogisticRegression`, `C` selected by train/validation only
- Clean run summary JSON: `results/summary/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z_summary.json`
- Clean run summary Markdown: `results/summary/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z_summary.md`
- Raw directory: `results/raw/Cora/CAST_CERTIFICATE_SMOKE/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z/`
- Log directory: `logs/cast_certificate/cast_cert_clean_smoke_Cora_seed0_20260626T135413Z/`
- Code commit at run time: `39c2707`

Note: result JSON records `project_dirty=true` because the run itself creates untracked log files before result JSON is written. The code/config had already been committed before the clean run.

## Main Table

| ID | Variant | Test@best-val | Label agreement | FN mass after | kNN overlap | CAST overlap |
|---|---|---:|---:|---:|---:|---:|
| C0 | GRACE | 84.78 | 0.0000 | 0.2222 | 0.0000 | 0.0000 |
| C1 | embedding kNN | 85.19 | 0.8152 | 0.2182 | 1.0000 | 0.2851 |
| C2 | CAST proxy | 85.70 | 0.7548 | 0.2188 | 0.2851 | 1.0000 |
| C4 | latent target certificate | 85.93 | 0.7911 | 0.2185 | 0.4150 | 0.2109 |
| C5 | certificate + CAST score | 86.16 | 0.7886 | 0.2185 | 0.3621 | 0.7606 |

## Interpretation

- C4 and C5 both improve over CAST proxy accuracy in Cora seed=0 smoke.
- C5 improves accuracy by `+0.46` over CAST proxy while also improving sibling label agreement from `0.7548` to `0.7886`.
- C4 is more distinct from CAST (`CAST overlap 0.2109`) and still improves accuracy to `85.93`.
- kNN still has the highest label agreement, so the next pilot must verify this is not simply kNN-like positive mining.

## Decision

`GO_TO_PILOT_PLANNING_WITH_CAUTION`

This is the first positive smoke after the IRIS/CPR failures. It supports planning a narrow Pilot-A, not formal claims.

