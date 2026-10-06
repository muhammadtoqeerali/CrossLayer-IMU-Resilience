# Phase 5K — Fault-Only Execution Qualification v1

**Status:** QUALIFIED_FAULT_ONLY_EXECUTION_EQUIVALENCE  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Motivation

The previously qualified paired execution harness computes both a clean
reference forward and a faulted forward for every requested fault execution.

That behavior is appropriate for qualification but cannot be used literally
for prospective outer execution because Phase 5E already budgets clean model
forwards separately through the frozen clean-cache estate.

Using the paired runner in outer execution would therefore exceed the frozen
clean-forward exposure count.

## Fault-only layer

Phase 5K adds isolated fault-only primitives without modifying the already
qualified paired harness.

The new layer reuses the same:

- `FaultIdentity`;
- target mappings;
- scheduled bit operator;
- mutation-record construction;
- exact one-bit validation;
- PTQ FX interpreter logic;
- qint8 state mutation logic.

It does not call a clean-reference model before the faulted execution.

## Qualification

The fault-only layer is qualified on the existing five-window
training-calibration fixture.

Fourteen cases are tested:

- two FP32 targets × two persistence modes;
- four PTQ activation/buffer targets × two persistence modes;
- one PTQ qint8 weight target × two persistence modes.

This covers all five representation families.

For every inference, the new fault-only path must match the existing paired
harness exactly in:

- faulted model output, bitwise;
- mutation record;
- fault ID;
- input SHA-256;
- active/inactive schedule.

Transient mask:

`True, True, True, True, True`

Persistent mask:

`False, False, True, True, True`

## Forward-exposure contract

The fault-only layer contains no embedded clean-reference forward.

For FP32, exactly one of two mutually-exclusive single-forward branches runs.

For PTQ activation/buffer faults, exactly one FX interpreter execution runs.

For PTQ qint8 weight faults, exactly one mutated-model forward runs.

This makes the path compatible with the separately frozen Phase-5E clean-cache
forward budget.

## Scientific boundary

Qualification uses training-calibration payloads only.

No outer-test payload is read.

No outer model forward or fault execution occurs.

No outer shard or prediction is read.

Phase-5D sampling/identity, Phase-5E plan, and Phase-5I authorization remain
unchanged.

No CC or CSC result is generated.
