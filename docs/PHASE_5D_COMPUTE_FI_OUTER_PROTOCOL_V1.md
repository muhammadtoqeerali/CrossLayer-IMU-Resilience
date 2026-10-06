# Phase 5D — Prospective Outer Compute-FI Protocol v1

**Status:** FROZEN_PROSPECTIVE_OUTER_COMPUTE_FI_PROTOCOL  
**Evidence tier:** P0 software fault injection  
**Freeze date:** 2026-10-05

## Purpose

Phase 5D freezes the compute-fault outer-test sampling, cardinality, pairing,
temporal semantics, and reporting rules before any compute-fault outer-test
model execution.

No outer-test model output was used to define this protocol.

## Outer population

The frozen primary 300-ms fivefold population contains:

- 61 subjects;
- 6,309 trials;
- 273,830 windows;
- 264,024 Activity windows;
- 9,806 Falling windows.

Each subject appears in exactly one outer fold.

All three checkpoint seeds and all five folds remain required.

No single best, worst, representative, or deployment checkpoint is selected.

## Fault estate

The five frozen compute-FI families are:

1. qint8 persistent-weight single-bit flip;
2. quint8 activation single-bit flip;
3. quint8 intermediate-buffer single-bit flip;
4. FP32 activation single-bit flip;
5. FP32 intermediate-buffer single-bit flip.

The target estate contains 14 model-independent sampling strata.

Ten FP32 targets are common to FP32 and mixed-precision PTQ execution and use
the same sampling coordinates in both variants.

Four additional strata are PTQ-only:

- `conv_2.0.weight`;
- `conv2_quantized_input`;
- `conv2_quantized_output`;
- `post_fc2_quantized_buffer`.

Counting variant-specific execution, the estate contains 24 target strata.

## Target shapes

Target tensor shapes and element counts were frozen using clean synthetic
execution only.

Common FP32 boundaries were required to have identical per-inference shape and
element count in FP32 and PTQ.

No outer-test sample was loaded to establish target shape.

## Deterministic coordinate sampling

There is one replicate: replicate 0.

Multiplicity is exactly one.

For every eligible target and parent identity, SHA-256 deterministically
selects:

- one tensor element;
- one eligible bit.

Persistent faults also receive one deterministic onset index within the trial.

No global RNG state is used.

Checkpoint seed is excluded from coordinate sampling.

Model variant is excluded from coordinate sampling.

Therefore all three checkpoint seeds use the same logical coordinate, and
common FP32 targets use the same logical coordinate in FP32 and PTQ.

The model-specific frozen `FaultIdentity` remains distinct because it includes
checkpoint seed and model variant.

## Temporal modes

### Transient

Every outer window is a parent.

One single-bit instance is generated for every unique target stratum.

A transient fault exists only for that inference.

Different transient fault instances are alternate worlds and must **not** be
concatenated into a synthetic faulted trial.

### Persistent

Every outer trial is a parent.

One persistent instance is generated for every unique target stratum.

The onset window is SHA-256-derived.

The same logical target element and bit remain corrupted from onset through
trial end.

State must reset at trial boundaries.

## Frozen cardinality

Model-independent sampling identities:

- transient: **3,833,620**;
- persistent: **88,326**;
- total: **3,921,946**.

Variant/seed-specific frozen fault identities:

- transient: **19,715,760**;
- persistent: **454,248**;
- total: **20,170,008**.

Clean C0 window forwards across two model variants and three seeds:

- **1,642,980**.

Persistent faulted forward exposure is greater than the persistent fault-ID
count because one persistent identity can affect multiple post-onset windows.

Its exact forward count must be derived from frozen trial window counts and
frozen onset hashes in a separate deterministic execution plan **before any
outer model forward**. That derivation may not alter any sampling decision.

## Non-finite outputs

The bit operator must never sanitize, clip, repair, replace, or suppress
NaN/Inf.

Non-finite outputs are explicitly recorded.

They may not be silently coerced to Activity or Falling.

The primary transient divergence endpoint therefore treats either:

- a non-finite faulted output; or
- a finite changed decision

as divergence from clean execution.

Finite-only decision flips remain separately reported.

## Reporting

All three previously frozen operating points remain required:

- balanced;
- low false alarm;
- timely 150 ms.

Thresholds remain the frozen validation-selected thresholds.

No compute-fault threshold retuning is permitted.

Transient faults are summarized as alternate-world paired inference effects.

Persistent faults may be evaluated as coherent trial sequences.

Primary uncertainty is subject-level.

Overlapping windows are never treated as independent uncertainty units.

Checkpoint seeds are evaluated separately within subject and equal-weighted
within subject before cross-subject macro aggregation.

Every predeclared target must be reported.

Family macros equal-weight their predeclared target strata.

Transient and persistent modes are not pooled in the primary analysis.

FP32 bit results include the predeclared mantissa, exponent, and sign groups.
Quantized results include bits 0 through 7.

Outer outcomes may not select favorable targets, bits, families, metrics,
checkpoint seeds, or persistence modes.

## Phase boundary

Phase 5 outer execution will produce C0 and CC evidence only.

CSC is deferred to Phase 6.

This remains P0 software FI.

No MCU, SEU-rate, cache, register, firmware, HIL, or physical-equivalence claim
is permitted.

The next step is to generate and qualify the deterministic outer identity,
persistent-exposure, and shard plan from frozen metadata without executing a
model.

## Parent-bound execution identity technical repair

A post-freeze identity audit found that the qualified Phase-5A `fault_id`
contains the mutation specification but does not contain subject, task, trial,
window, or other outer-parent identity.

Therefore `fault_id` is retained unchanged as the frozen mutation-specification
identifier, but it is not used alone as the primary outer execution-record key.

Phase 5D now uses three explicit identity levels:

1. `sampling_instance_id` — parent-bound and model-independent;
2. `phase5a_fault_id` — model/checkpoint mutation-specification provenance;
3. `outer_instance_id` — parent-bound, model-variant-bound, and
   checkpoint-seed-bound execution identity.

`outer_instance_id` is SHA-256 over canonical compact JSON containing:

- namespace;
- `sampling_instance_id`;
- `phase5a_fault_id`;
- model variant;
- checkpoint seed.

This repair does **not** change:

- the frozen sampling payload;
- any sampled element;
- any sampled bit;
- any persistent onset;
- any target;
- any fault family;
- multiplicity;
- replicate count;
- persistence semantics;
- reporting endpoints;
- any frozen cardinality.

The existing 20,170,008-count field named `model_specific_fault_ids` is
retained for provenance but is no longer interpreted as a claim that the
Phase-5A `fault_id` itself is globally unique across outer parents.

The authoritative planned execution cardinality is also recorded explicitly as
20,170,008 parent-bound `outer_instance_id` records.

No outer data, prediction, model execution, or result was used to make this
technical identity repair.
