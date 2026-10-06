# Phase 5H — Complete Shard-Style Execution Qualification v1

**Status:** QUALIFIED_COMPLETE_SHARD_STYLE_EXECUTION  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5H qualifies the complete transient and persistent sequence mechanics
required by the frozen 732-shard outer compute-FI plan.

Qualification uses frozen training-calibration data only.

The outer execution gate remains false.

## Qualification repair transparency

The first qualification attempt stopped before fault-shard execution because
the qualifier's `shard_id` string expression omitted one concatenation operator before the transient/persistent suffix.

Before that stop, the permitted training-calibration fixture had been read,
models had been loaded, clean forwards had been executed, and the two clean
cache artifacts had been written.

No fault shard had executed.

The qualifier was repaired by adding only the missing string-concatenation
operator. The already-frozen Phase-5H qualification configuration and all
scientific semantics remained unchanged. Failed qualification outputs were
discarded before rerunning the complete fixture matrix.

## Fixture

A frozen fold-1 training-calibration identity anchors one five-window sequence
from the same training trial.

Persistent onset is the third inference.

The required active masks are:

- transient: `True, True, True, True, True`;
- persistent: `False, False, True, True, True`.

## Coverage

Fourteen shard-style fault fixtures are executed.

FP32 covers:

- FP32 activation;
- FP32 intermediate buffer;

under transient and persistent semantics.

PTQ covers:

- FP32 activation;
- FP32 intermediate buffer;
- quint8 activation;
- quint8 intermediate buffer;
- qint8 persistent weight;

under transient and persistent semantics.

This yields seven transient and seven persistent fault fixtures.

## Transient semantics

Transient fixtures use one independent fault identity per inference.

A five-window transient fixture therefore creates five parent-bound execution
records.

Transient instances remain alternate worlds and are not recomposed into a
persistent sequence.

## Persistent semantics

Persistent fixtures use one fault identity for the trial target.

The identity is inactive before onset and active from onset through the end of
the five-window fixture.

Persistent fixture storage contains one execution-instance record and five
per-window effect records.

## PTQ qint8 weight reset

The PTQ qint8 weight session is reset after each sequence.

A separate frozen training-calibration trial is used as a post-reset probe.

The reset fault-model output must be bitwise identical to the clean PTQ model,
which qualifies the no-cross-trial-leakage requirement.

## Cross-variant coordinate reuse

For FP32 targets common to FP32 and PTQ, qualification coordinate derivation
excludes model variant.

The same element and bit coordinates are therefore required across the two
variants.

## Clean-cache sharing

Exactly two clean-cache fixture artifacts are produced:

- one FP32;
- one PTQ.

Transient and persistent fault fixtures for a given variant reference the same
clean cache.

## Atomic outputs

All fixture artifacts use the qualified Phase-5F atomic/resume contract.

Qualification produces:

- 2 clean-cache artifacts;
- 14 fault-shard artifacts;
- 42 unique parent-bound execution-instance records;
- 70 per-window effect records.

## Scientific boundary

Training-calibration payloads are used.

No outer-test payload is used.

The outer execution gate remains false.

No outer model forward or outer fault execution occurs.

No frozen outer shard is executed.

No outer prediction is read.

No OnField evidence is used.

Phase-5D sampling/identity and the Phase-5E plan remain unchanged.

No CC or CSC result is generated.
