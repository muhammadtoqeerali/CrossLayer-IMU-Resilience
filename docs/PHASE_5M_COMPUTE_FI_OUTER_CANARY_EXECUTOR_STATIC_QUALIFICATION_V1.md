# Phase 5M — Outer Canary Executor Static Qualification v1

**Status:** QUALIFIED_STATIC_CANARY_EXECUTOR_PRE_OUTER  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5M implements the exact executor for the single Phase-5L-authorized outer
canary but does not invoke its execution path.

Only configuration and shard-validation paths are executed in this phase.

## Authorization

The executor hard-binds the frozen Phase-5L config SHA.

It accepts exactly:

`p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9`

and exactly its clean cache:

`p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b`.

Every other shard is rejected before any signal-array access or model load.

## Frozen canary estate

The executor validates:

- subject 9;
- fold 5;
- FP32;
- seed 42;
- transient one-inference persistence;
- 44 trials;
- 2,481 windows;
- 10 FP32 targets;
- 2,481 clean forwards;
- 24,810 faulted forwards;
- 27,291 total first-run forwards.

## Fault identity

For each transient fault execution the implementation reuses the frozen
Phase-5D functions for:

- canonical outer sampling payload;
- `sampling_instance_id`;
- element index derivation;
- bit-position derivation;
- transient inference index;
- parent-bound `outer_instance_id`.

The Phase-5A `fault_id` remains the mutation-specification identity.

## Clean cache

Clean inference is a separate pass.

The clean artifact stores per-window:

- parent identity;
- input SHA-256;
- clean-output SHA-256;
- exact output values;
- float32 hexadecimal values;
- softmax values;
- non-finite status.

The clean cache must be atomically committed and reusable before fault-shard
execution begins.

## Fault shard

Fault execution uses only the Phase-5K fault-only FP32 runner.

It does not call the paired runner and does not perform a clean-reference
forward inside the fault loop.

Each fault row stores:

- `outer_instance_id`;
- `sampling_instance_id`;
- Phase-5A `fault_id`;
- complete fault coordinates;
- parent identity;
- input SHA-256;
- faulted-output SHA-256;
- exact output values;
- float32 hexadecimal values;
- softmax values;
- explicit non-finite status;
- mutation metadata.

No thresholding or metrics are computed by the executor.

## Atomic/resume semantics

The executor reuses the qualified Phase-5F:

- success-marker validation;
- partial-output rejection;
- atomic temporary directory;
- final commit;
- reusable-artifact logic.

Success markers bind both the frozen Phase-5E plan SHA and the exact executor
source SHA.

## Outer-data boundary

Only `_load_trial_signal()` calls `np.load`.

That function is reachable only from `execute-canary`.

`validate-config` and `validate-shard` contain no:

- `np.load`;
- `torch.load`;
- model loader;
- model forward;
- fault runner;
- output-root creation.

No labels are opened by the executor.

## Current state

The executor is implemented and statically qualified.

The actual `execute-canary` command has not been invoked.

Therefore no outer payload, model load, forward, fault, prediction, CC result,
or CSC result has yet been generated.
