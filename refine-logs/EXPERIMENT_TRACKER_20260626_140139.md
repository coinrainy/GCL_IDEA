# Experiment Tracker: CAST Certificate Pilot-A

| Run ID | Milestone | Purpose | System / Variant | Dataset | Seeds | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|-------|-------|---------|----------|--------|-------|
| PA-M0-001 | M0 | sanity | C0/C1/C2/C4/C5 | CiteSeer | 0 | stratified 1:1:8 | accuracy, label agreement, FN mass | MUST | TODO | first multi-dataset check |
| PA-M1-CORA | M1 | Pilot-A main | C0/C1/C2/C3/C4/C5 | Cora | 0,1,2 | stratified 1:1:8 | accuracy, label agreement, FN mass | MUST | TODO | includes known positive smoke seed 0 rerun |
| PA-M1-CITE | M1 | Pilot-A main | C0/C1/C2/C3/C4/C5 | CiteSeer | 0,1,2 | stratified 1:1:8 | accuracy, label agreement, FN mass | MUST | TODO | must confirm Cora signal transfers |
| PA-M1-PUB | M1 | Pilot-A main | C0/C1/C2/C3/C4/C5 | PubMed | 0,1,2 | stratified 1:1:8 | accuracy, label agreement, FN mass | MUST | TODO | monitor memory/runtime |
| PA-M2-001 | M2 | novelty diagnostics | kNN/PPR/CAST/C4/C5 | Cora/CiteSeer/PubMed | 0,1,2 | same as M1 | overlap, partial correlation | MUST | TODO | proves not kNN/PPR mining |
| PA-M3-001 | M3 | decision audit | all Pilot-A outputs | all | 0,1,2 | same as M1 | mean/std, kill rules | MUST | TODO | decide GO/REVISE/PIVOT |
| PA-N1-001 | M3 | simplicity check | C4 vs C5 | Cora | 0,1,2 | stratified 1:1:8 | accuracy vs label agreement | NICE | TODO | only after M1 positive |

## Pilot-A Gates

- GO: C5 matches or beats CAST proxy on accuracy and improves label agreement on at least 2/3 datasets, with nontrivial overlap/partial-correlation diagnostics.
- REVISE: C4/C5 improves diagnostics but not accuracy, or only one dataset passes.
- PIVOT: C4/C5 fails against CAST proxy on 2/3 datasets or collapses to kNN/PPR by diagnostics.

## Forbidden

- No formal 10-seed run.
- No SOTA/performance claim.
- No additional IRIS/CPR variants during Pilot-A.

