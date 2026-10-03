# Phase 4G — Static PTQ v2: PReLU FP32

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Reason for v2

The frozen v1 static-PTQ candidate was executed first on the
seed-42/fold-1 checkpoint using only its 4096 frozen training
calibration windows.

v1 failed the predeclared numerical fidelity gates.

Subsequent calibration-only diagnostics preserved the same v1
qconfig and localized the dominant discrepancy to QuantizedPReLU.

The final operator microtest exercised all three converted PReLU
operators with controlled positive and negative values using their
actual converted input/output quantization parameters.

Observed result:

- positive branch: 25 / 25 integer-code mismatches;
- negative branch: 0 / 22 mismatches.

No alternative quantizer, validation data, outer-test data,
OnField data, or fault data was used to obtain this diagnosis.

## Single v2 change

v2 makes exactly one implementation change:

`torch.nn.PReLU` is excluded from FX quantization and remains FP32.

Equivalent QConfigMapping rule:

`get_default_qconfig_mapping("x86").set_object_type(torch.nn.PReLU, None)`

Everything else remains unchanged from v1:

- static PTQ;
- x86 backend;
- default x86 qconfig for eligible operators;
- Conv1d and Linear remain quantization targets;
- activation/weight observer policies remain unchanged;
- frozen 4096-window training-only calibration set per fold;
- all five calibration identity sets;
- all 15 frozen FP32 checkpoints;
- numerical acceptance gates.

## Precision language

v2 is **mixed-precision static PTQ by construction**.

Because PReLU intentionally remains FP32, v2 must not be described
as fully INT8 even if all Conv1d and Linear operators quantize
successfully.

## Unchanged numerical gates

Using only each fold's frozen training calibration set:

- all outputs finite;
- output shape preserved;
- FP32/PTQ argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Scientific boundary

The first v2 execution will again use seed 42 / fold 1 only as the
engineering pilot, because it is the same checkpoint used to expose
the preserved v1 implementation failure.

It is not selected from outer-test performance.

If v2 fails, that failure is preserved and any further change requires
another explicitly versioned pre-result protocol.
