# Refinement Report: WILLOW-GCL

## What Changed

SIVA-GCL was demoted from active mainline to mandatory control because its semantic critic can be read as GraphMAE-like reconstruction plus intervention augmentation. WILLOW keeps the positive-view direction but replaces the reconstruction critic with a latent ego target-prediction certificate.

## Main Refinement

- Replace "world model" overclaim with "latent ego target-prediction certificate"。
- Delete virtual negatives from the mainline。
- Require fixed intervention budget and matched random controls。
- Add Graph-JEPA-only and certificate-shuffled controls as kill gates。
- Keep the paper narrative inside GCL positive signal and false-negative exposure。

## Current State

Ready only for smoke implementation planning. No code or experiment was run in this refinement.
