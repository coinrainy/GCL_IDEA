# Refinement Report: CAST-GCL

## What Changed

WILLOW-GCL was demoted from active mainline to module/control because same-anchor certified view generation still risks being perceived as Graph-JEPA/PCR plus augmentation scoring. CAST keeps the latent certificate but moves the contribution to cross-node semantic transport and positive closure.

## Main Refinement

- Move from same-node hard positive generation to cross-node false-negative relation construction。
- Use transport energy rather than pair probability as the semantic relation criterion。
- Require PMGCL-lite, kNN positive, WILLOW same-node, shuffled-certificate, and random-transport controls。
- Keep all claims at pre-review/proposal level。

## Current State

Ready only for fresh review or Cora seed=0 smoke planning. No code or experiment was run.
