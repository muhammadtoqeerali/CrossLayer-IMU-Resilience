# Phase 1 Final Report

## Status

COMPLETE

## Purpose

Audit inherited implementations and reconstruct a scientifically clean
protected workload for CrossLayer experiments.

## Audited projects

The workstation audit covered:

- IMU_Reliability
- RC-RGD-IMU_publish
- multiple Protechto worktrees
- HR_LR_Fallings
- TRUST_ROBOT
- related runtime/source directories

The audit found substantial duplication across inherited worktrees.

No source was accepted solely on automated file-ranking evidence.

## Primary finding

A task-specific reconstruction of the DATE-2025 400-ms pre-impact
fall-detection CNN exists in the clean RC-RGD-IMU lineage.

This is more appropriate for the main CrossLayer protected workload than the
generic HAR CNN candidates.

## Historical source

Clean reference tree:

`RC-RGD-IMU_publish`

Reference commit:

`e1db7880e17a5632bc4ce238125923f986e85519`

## Historical checkpoint

Pinned external artifact:

`/mnt/hdd16T/protechto/checkpoints/CNN/400ms/2025-02-25_12_24_47/best-checkpoint.ckpt`

SHA-256:

`ee7c0079bfb8555bff45c3077cc24eaa4373c57729045d92a831a1d7a3ea9bb1`

Size:

782797 bytes

The checkpoint is intentionally not stored in this Git repository.

## Migrated model

ID:

`DATE2025_CNN_400MS_RECONSTRUCTED`

Contract:

- stored input: 40 x 9
- effective CNN channels: 6
- sampling rate: 100 Hz
- temporal window: 400 ms
- classes: Activity and Falling
- parameters: 63,173
- penultimate representation: 256

## Exact parity evidence

Deterministic parity vectors:

68

Maximum absolute logit difference:

0.0

Maximum absolute penultimate-feature difference:

0.0

Argmax task predictions:

identical

Historical streaming decisions:

identical

Normalized state-artifact reload:

exact task-logit and feature parity

Canonical tensor-state SHA-256:

`c98987476536320191f8875316cf8caeeb0b3f2edc51d65be7ac7d2eaea03124`

## Scientific isolation

The CrossLayer protected baseline contains no inherited:

- OOD threshold
- sensor integrity mechanism
- reliability gate
- trust state
- compute integrity mechanism
- recovery mechanism
- CrossLayer supervisor

This is essential for later causal comparisons.

## Migration decisions

ADAPT:

- DATE-2025 task model
- historical decision semantics
- ONNX/export engineering pattern
- portable embedded engineering pattern

REUSE:

- verified historical checkpoint as external candidate weights

REFERENCE_ONLY:

- historical reconstruction experiments
- generic compact 1D CNN
- DS-CNN
- compact TCN
- previous reliability runtime

REJECT for the clean baseline:

- inherited OOD logic
- inherited sensor reliability/fusion logic

## Validation repair

The first migration-validation attempt exposed a new provenance-hasher defect
for zero-dimensional integer state tensors.

The hasher was corrected by flattening contiguous tensors before a byte-wise
view.

A regression test now covers scalar integer model state.

The defect affected only the new metadata hash implementation.

It did not affect the checkpoint, model weights, architecture or inference
parity.

## Phase-1 conclusion

Phase 1 establishes that the historical pre-impact task workload can be
faithfully and independently represented inside CrossLayer-IMU-Resilience.

The candidate remains unfrozen until Phase 2.

## Next phase

Phase 2 will establish the frozen executable baseline.

It must cover:

- candidate confirmation
- environment identity
- decision semantics
- clean reference behavior
- export
- quantization
- numerical parity
- model size
- computational cost
- initial deployment feasibility

Fault-study protocols remain outside Phase 2.
