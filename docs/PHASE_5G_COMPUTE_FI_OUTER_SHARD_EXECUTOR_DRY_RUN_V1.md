# Phase 5G — Outer Shard Executor Wiring / 732-Shard Dry Run v1

**Status:** QUALIFIED_732_SHARD_DRY_RUN  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5G wires the frozen Phase-5E outer execution plan to the qualified
Phase-5F atomic/resume core and validates the complete 732-shard estate without
executing a model.

The outer execution gate remains false.

## Dry-run scope

The dry run verifies:

- all frozen dependency hashes;
- all 732 fault shards;
- all 366 clean caches;
- all 15 FP32 checkpoint artifact hashes;
- all 15 PTQ state-dict artifact hashes;
- all 15 PTQ TorchScript artifact hashes;
- exact shard/clean-cache uniqueness;
- exact model/fold/seed binding;
- exact transient/persistent identity cardinality;
- exact persistent model-window exposure;
- exact header-only regeneration of the Phase-5E plan.

The Phase-5E trial inventory, subject inventory, persistent onset binding,
clean-cache records, and all 732 shard records must replay exactly.

## Frozen dry-run totals

- Fault shards: 732.
- Clean caches: 366.
- Model artifacts hash-checked: 45.
- Transient outer-instance IDs: 19,715,760.
- Persistent outer-instance IDs: 454,248.
- Total outer-instance IDs: 20,170,008.
- Clean model-window evaluations: 1,642,980.
- Transient faulted model-window evaluations: 19,715,760.
- Persistent faulted model-window evaluations: 10,083,060.
- Total faulted model-window evaluations: 29,798,820.
- Total clean + faulted model-window evaluations: 31,441,800.

## Execution gate

The controller exposes an `execute-shard` command only to enforce the current
gate.

During Phase 5G it fails before dataset/model access because:

`outer_execution_enabled = false`

A later governed freeze is required before any outer shard may execute.

## Scientific boundary

The dry run reads `segments.npy` headers only.

It does not:

- read segment payload bytes;
- open labels;
- call `np.load`;
- call `torch.load`;
- materialize signals;
- load a model;
- execute a model;
- invoke a fault operator;
- execute a fault shard;
- read an outer prediction;
- use OnField;
- generate CC or CSC results.

Phase-5D sampling/identity and the Phase-5E plan remain unchanged.

## Next gate

Before outer execution can be enabled, the complete shard-style execution
logic—including transient window handling, persistent trial reset/onset
handling, clean-cache sharing, output records, and atomic success artifacts—
must be qualified on synthetic and training-calibration fixtures.
