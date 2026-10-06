# Phase 5E — Outer Compute-FI Execution Plan v1

**Status:** QUALIFIED_FROZEN_DERIVED_EXECUTION_PLAN  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5E converts the already-frozen Phase-5D prospective outer compute-FI
protocol into a deterministic resumable execution plan.

This step does not change the scientific sampling protocol.

## Metadata boundary

The planner reads only:

- the frozen fivefold split;
- filesystem subject/task/trial names;
- `segments.npy` NPY header bytes.

It never:

- opens `labels.npy`;
- calls `np.load`;
- creates a memmap;
- reads segment payload bytes;
- materializes a signal array;
- loads or executes a model;
- injects a fault;
- reads outer predictions.

## Bound outer inventory

The header-only inventory contains:

- 61 subjects;
- 6,309 trials;
- 273,830 windows.

All retained `segments.npy` files use NPY format v1.0 and stored dtype
little-endian float64.

The exact trial-window inventory and subject inventory are content-addressed in
the execution plan.

The persistent-onset binding digest is also reproduced exactly from the prior
header-only qualification.

## Sharding

The fault-execution shard unit is:

`subject × model_variant × checkpoint_seed × persistence`

This gives exactly:

- 61 subjects;
- 2 variants;
- 3 checkpoint seeds;
- 2 persistence modes;
- **732 fault shards**.

Every subject therefore has 12 fault shards.

Each shard is immutable and identified by a deterministic SHA-256-derived
`shard_id`.

The required resume key is `shard_id`.

A shard may be reused only when its atomic success marker and all recorded
hashes match the frozen executor, plan, and outputs.

## Clean C0 cache

Clean inference is not recomputed separately for transient and persistent
shards.

The clean cache unit is:

`subject × model_variant × checkpoint_seed`

This gives exactly **366 clean caches**.

Each clean cache is shared by its transient and persistent fault shards.

The frozen clean-forward count remains **1,642,980** model-window evaluations.

## Fault-execution cardinality

The plan preserves the frozen Phase-5D identity cardinalities:

- transient outer instances: **19,715,760**;
- persistent outer instances: **454,248**;
- total outer instances: **20,170,008**.

For transient faults, one fault identity affects exactly one model-window
evaluation.

For persistent faults, one identity affects every inference from its
deterministic onset through trial end.

Therefore persistent faulted model-window exposure is derived exactly from the
frozen per-trial window counts and frozen onset hashes and is larger than the
persistent identity count.

The exact persistent and total model-window evaluation counts are stored in the
plan and are immutable once the plan is qualified.

## Identity hierarchy

Every fault execution record must retain:

- `sampling_instance_id`;
- unchanged Phase-5A `fault_id`;
- parent-bound `outer_instance_id`.

`outer_instance_id` is the primary execution-record key.

## Execution requirements

All 732 fault shards are required.

All 366 clean caches are required.

No target, bit, element, onset, family, persistence mode, checkpoint seed, or
sampling coordinate may be changed from outer outcomes.

Thresholds remain frozen validation-selected thresholds.

Non-finite faulted outputs must be recorded and must not be silently coerced to
a class.

Phase 5 produces C0/CC only.

CSC remains Phase 6.

This remains P0 software fault injection and does not establish physical or MCU
fault equivalence.
