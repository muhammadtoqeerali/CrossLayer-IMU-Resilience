# Phase 4G — Static PTQ v7: FP32 fc.1

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Preserved v6 result

v6 remains a failed candidate.

Its structural and serialization gates passed. All numerical gates
except the unchanged p99 absolute softmax-probability fidelity gate
passed.

No numerical threshold is changed in v7.

## v6 FC-region causal localization

The FP32 route entering the FC region remained below the frozen p99
limit.

Quantizing the flattened activation before `fc.1` alone pushed p99
above the frozen 0.05 gate.

The quantized `fc.1` operator added a smaller subsequent increment.

The later activation requantization after `fc.2` did not cause the
first breach and remains preserved in v7.

## Single v7 change

Relative to v6, v7 removes the quantization boundary before `fc.1`.

v6:

`fc.0 -> quantize -> quantized fc.1 -> dequantize -> fc.2 -> quantize -> fc.3`

v7:

`fc.0 -> FP32 fc.1 -> FP32 fc.2 -> quantize -> fc.3`

Thus `fc.0`, `fc.1`, and `fc.2` form one continuous FP32 segment.

All earlier v6 precision boundaries remain unchanged.

## Synthetic pre-result capability evidence

The synthetic structural capability probe confirmed:

- the v6 FP32 front end remains unchanged;
- `conv_2.0` remains the only quantized Conv1d;
- the post-conv2 FP32 path remains unchanged;
- `fc.0 -> fc.1` is a direct FP32 edge;
- `fc.1` is native FP32 Linear;
- `fc.1` checkpoint weight and bias are exact;
- `fc.1 -> fc.2` is a direct FP32 edge;
- no quantized Linear module remains;
- quantization begins immediately after `fc.2`;
- the `fc.0 -> fc.1` and `fc.1 -> fc.2` tensor handoffs are exact.

No real dataset sample or v7 task output was observed.

## Numerical acceptance

Unchanged:

- all outputs finite;
- exact output shape;
- argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Scientific boundary

v7 is frozen before any real v7 calibration-fidelity result.

No validation, outer-test, OnField, or fault evidence is used for v7
design.

v7 remains mixed-precision static PTQ and must not be described as
fully INT8.
