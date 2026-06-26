# Next Direction After IRIS / CPR

## Decision

`PIVOT_REQUIRED`

The current best technical lesson is not that IRIS works. The lesson is that **response residual is useful only as a weak certificate over strong candidate relations**, while hard anti-proximity and neutral-heavy closure fail.

## Recommended Next Step

Prioritize a **real CAST certificate** before restarting idea discovery:

```text
latent target-prediction transport energy
  > feature/embedding/PPR candidate score
```

The next method should not be another linear mixture of proximity and response residual. It must learn or compute a label-free transport certificate whose score predicts offline label agreement after controlling proximity.

## Minimal Next Experiment

Only if implementing real CAST:

- Dataset: Cora only.
- Seed: 0 only.
- Split/evaluator: unchanged project `1:1:8` + frozen encoder + Logistic Regression.
- Controls: GRACE, kNN, PPR, PMGCL-lite, CAST proxy, CPR I17.
- New variant: real latent target-prediction CAST certificate.

## GO Criteria

- accuracy at least near CAST proxy (`>=85.6` on Cora seed=0 smoke);
- label agreement above CAST and competitive with kNN (`>=0.80`);
- relation overlap with kNN not excessive;
- certificate score has positive partial correlation with label agreement after controls;
- no large neutralization-only artifact.

If these are not plausible, run fresh `/idea-discovery` instead of coding more variants.

