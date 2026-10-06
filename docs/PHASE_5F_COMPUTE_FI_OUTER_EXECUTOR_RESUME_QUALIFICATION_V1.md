# Phase 5F — Compute-FI Outer Executor / Resume Core Qualification v1

**Status:** QUALIFIED_EXECUTOR_RESUME_CORE  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5F qualifies the execution-record and atomic resume machinery that the
frozen Phase-5 outer executor must use.

This step does not execute any outer-test shard.

## Qualified primitives reused

The executor core reuses the already-qualified Phase-5 components:

- `FaultIdentity`;
- exact qint8/quint8/FP32 bit operators;
- FP32 paired execution;
- PTQ activation/buffer paired execution;
- PTQ persistent-weight session;
- parent-bound `outer_instance_id`.

No new mutation semantics are introduced.

## Atomic artifact contract

Every clean cache or fault shard is written first to:

`<artifact_id>.partial`

The temporary directory must contain all declared outputs with matching SHA-256
hashes.

The temporary directory is atomically renamed to the final artifact directory.

`_SUCCESS.json` is written **after** the directory rename and is therefore the
last artifact written.

An artifact is reusable only when:

- `_SUCCESS.json` exists;
- artifact ID matches;
- artifact kind matches;
- frozen plan hash matches;
- executor implementation hash matches;
- every recorded output exists;
- every recorded output SHA-256 matches.

A final directory without a valid success marker is partial and cannot be
reused.

A temporary directory cannot be reused.

Partial output requires explicit recomputation.

## Qualification

Resume semantics were tested synthetically for:

- first atomic write;
- exact reuse;
- wrong plan hash rejection;
- wrong executor hash rejection;
- post-success output corruption rejection;
- partial final directory rejection;
- partial temporary directory rejection;
- explicit partial recomputation cleanup.

Model integration was qualified on one frozen training-calibration input using:

- FP32 activation bit flip;
- PTQ quint8 activation bit flip;
- PTQ qint8 weight bit flip.

The reconstructed eager PTQ model remained bitwise equal to the frozen
TorchScript clean reference.

All three execution records had unique parent-bound `outer_instance_id`
values.

## Scientific boundary

No outer-test payload was read.

No outer-test model forward was executed.

No outer-test fault was injected.

No one of the 732 frozen outer shards was executed.

No outer prediction was read.

No OnField data were used.

The frozen Phase-5D sampling protocol and Phase-5E execution plan were not
changed.

No C0/CC outer result and no CSC result were generated.

## Next step

The next governed step is to implement the full frozen outer shard executor on
top of this qualified core, then dry-run/hash-validate all 732 shards without
model execution before allowing the first outer shard.
