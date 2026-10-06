# Phase 5A — Compute-FI Operator Qualification v1

**Status:** QUALIFIED_BIT_EXACT_P0_OPERATORS  
**Date:** 2026-10-05  
**Evidence tier:** P0

## Scope

This qualification validates the bit-level compute-fault operators defined
after the frozen Phase-5A representation contract.

No task-performance inference is performed.

No dataset is accessed.

No outer-test or OnField evidence is read.

## Qualified operator semantics

The implementation qualifies:

- signed INT8 payload single-bit XOR;
- `torch.qint8` / `torch.int8` payload mutation for the persistent
  `conv_2.0.weight` tensor;
- `torch.quint8` / `torch.uint8` payload mutation for the actual v7
  quantized activation and intermediate-buffer representation;
- FP32 IEEE-754 payload single-bit XOR;
- transient one-inference scheduling;
- persistent-from-onset-to-trial-end scheduling.

The qint8/quint8 distinction is representation-critical: the frozen PTQ v7
persistent quantized weight is qint8, while its quantized activations and
buffers are quint8. Both remain the same frozen eight-bit XOR fault semantics;
they are not pooled as the same storage dtype.

## Bit-exact guarantees

Qualification requires that:

- exactly one selected payload bit changes;
- every non-selected tensor element remains bit-identical;
- applying the same bit flip twice restores the exact original payload;
- the source tensor is never modified;
- quantized scale, zero point and quantization axis remain unchanged;
- FP32 NaN or infinity outcomes are not sanitized;
- persistent state does not leak across trials.

## Synthetic coverage

The qualification exhaustively evaluates all signed INT8 values across all
eight bit positions.

It additionally evaluates every element/bit combination over both a synthetic
qint8 tensor and a synthetic quint8 tensor. The latter binds the operator to
the actual unsigned eight-bit activation/buffer payload representation used by
PTQ v7.

A fixed set of representative FP32 payload patterns is also qualified,
including zero, signed zero, finite extrema and subnormal values.

## Frozen-artifact structural qualification

The operator is additionally checked against:

- the frozen seed-42/fold-1 PTQ v7 `conv_2.0.weight` qint8 tensor;
- representative FP32 parameter tensors from the frozen seed-42/fold-1
  prospective CNN checkpoint.

These checks mutate detached tensor copies only.

The model is never executed.

## Scientific boundary

This qualification demonstrates software-level payload correctness only.

It does not establish:

- MCU memory placement;
- physical SEU rates;
- SRAM/flash equivalence;
- register/cache fault equivalence;
- firmware/HIL/physical evidence;
- CC task-performance degradation;
- CSC interaction results.

The outer compute-FI sampling plan remains unfrozen. The next step is to bind
these qualified operators into a development-safe model execution harness,
qualify clean/faulted paired inference and deterministic replay, and only then
freeze outer cardinality.
