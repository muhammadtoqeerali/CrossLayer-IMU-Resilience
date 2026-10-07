# Phase 6F — CSC Metadata-Executor Binding V1

## Status

**FROZEN_METADATA_EXECUTOR_BINDING_PRE_EXECUTION**

Phase 6F freezes the qualified metadata-only implementation of the Phase-6E
combined sensor + compute (CSC) executor before any execution-capable CSC path
is introduced.

This is an implementation/governance freeze, not a scientific protocol change.

## Frozen implementation

The bound implementation is:

- `experiments/phase_06/csc_outer_executor_v1.py`
- SHA256:
  `99320b5811c5e72940bcfc779fa530df88fb8b7a2f8662711b8f56e4cf5d61fd`

Its regression test is:

- `tests/test_phase6e_csc_outer_executor_metadata_v1.py`
- SHA256:
  `5da200d643c1c9333efba7e5fa38d32aa02ba72f8c8f8aa5eeed4451c882f0c2`

## Qualification

Before this freeze, the implementation passed:

- 13 metadata-executor tests;
- 54 Phase-6D/Phase-6E dedicated tests;
- 904 repository tests with the one historical Phase5R lifecycle test
  deselected.

## Bound scientific surface

The implementation consumes the already-frozen Phase-6E plan without changing
it:

- 4,237,835 model-independent CSC pairs;
- 21,793,038 model/seed pair-members;
- pair-surface binding:
  `3b39a909ea140db75da52315abcf5efe2748fcdb2d31f0fc98ddf766b000af53`;
- combined compute-coordinate binding:
  `2117c7b07084566607ccac46db150faae81d28bef51d0dcab7c700627d8eb435`;
- temporal-workload binding:
  `5b447bca2f45d4889bcf0dc86d6170c40a1028171a9f1b0ea7a28e2884f12ba8`.

## What is implemented

The frozen metadata executor can validate frozen hashes and deterministically
derive sensor selection, R2 structural omission status, compute-stratum
assignment, R1 transient-window selection, Phase-5 compute coordinates, and
sensor/compute temporal-overlap metadata.

## What is deliberately impossible

`EXECUTION_ENABLED` is false.

`execute_shard(...)` always raises
`PHASE6E_EXECUTION_DISABLED_METADATA_ONLY`.

The implementation does not import Torch or the Phase-4H sensor operator
module. It does not load models, call `apply_fault`, execute compute faults,
materialize CSC pair files, perform model forwards, or compute metrics.

Adding any execution-capable path requires a separately qualified and frozen
implementation step. It must not silently mutate this frozen metadata-only
implementation.

## Governance

No outer, CS, CC, CSC, validation, or OnField performance outcome was used in
this implementation or freeze. No threshold, checkpoint, pair, target,
persistence mode, coordinate, or workload was selected or altered.

No commit or push is performed by this freeze.
