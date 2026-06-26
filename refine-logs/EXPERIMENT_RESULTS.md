# Experiment Results: CAST Certificate Pilot-A

Fixed entry for the latest experiment result.

- Timestamped source: `refine-logs/EXPERIMENT_RESULTS_20260626_144910.md`
- Stage: Pilot-A / diagnostic, not formal.
- Decision: `REVISE_OR_PIVOT_BEFORE_ANY_MORE_PILOT`

Pilot-A checked Cora seeds 0-2, CiteSeer seeds 0-2, and PubMed seed 0. C4 latent certificate shows low overlap with kNN/PPR/CAST, so it is not simply the same mining rule, but C4/C5 do not produce stable accuracy gains over kNN or CAST proxy. PubMed seeds 1-2 were stopped by the no-go signal.

No formal, SOTA, robust, comprehensive, or final performance claim is supported.
