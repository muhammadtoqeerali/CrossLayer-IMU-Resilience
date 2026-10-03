# Phase 4G — Static PTQ v5: FP32 Front End

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Preserved v4 result

v4 is preserved as a failed candidate.

Its structural and serialization gates passed, but the unchanged p99
absolute softmax-probability error gate failed.

No acceptance threshold is relaxed in v5.

## Causal localization

Causal suffix replay showed that the first unacceptable final-output
distortion was already present at the first Conv+BN boundary.

A subsequent first-block decomposition separated:

1. input activation quantization;
2. folded Conv+BN weight quantization with FP32 accumulation;
3. the actual quantized Conv integer/output path.

Input activation quantization alone drove the final-output p99 error
above the frozen 0.05 gate.

Adding quantized folded weights and then the actual quantized Conv path
did not cause the initial breach.

## Failed structural mechanisms

Two narrower implementation mechanisms were tested with synthetic input
only and preserved as failures:

- excluding only `conv_1.0` from quantization;
- making only the complete `conv_1` block opaque FP32.

Both still left FX quantization upstream of the first block, so they
did not remove the diagnosed input-activation quantization.

## Successful synthetic mechanism

The viable mechanism groups

`normalizer -> permute(0,2,1) -> conv_1`

into one opaque FP32 `front_end`.

The structural capability probe demonstrated:

- raw model input enters `front_end`;
- no quantize node exists upstream of `front_end`;
- no dequantize node exists upstream of `front_end`;
- converted front-end input is bit-exact to pre-FX input;
- converted front-end output is bit-exact to pre-FX output;
- quantization begins immediately after `front_end`;
- `conv_2.0` remains quantized;
- `fc.1` remains quantized;
- `fc.4` remains FP32.

No real dataset sample or v5 task output was observed by that probe.

## Single v5 change

Relative to v4, v5 changes one precision boundary:

the first quantization point moves from before the first Conv1d to the
output of the FP32 front end.

Thus the normalizer, permutation, and complete first convolutional block
remain FP32.

All downstream v4 precision rules are retained.

## Required v5 precision placement

FP32:

- `front_end`:
  - normalizer;
  - permutation;
  - complete `conv_1`;
- all three FloatPReLU operations;
- final classifier `fc.4`.

Quantized:

- `conv_2.0`;
- `fc.1`.

## Numerical acceptance

Unchanged gates:

- all outputs finite;
- exact output shape;
- FP32/v5 argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Scientific boundary

v5 was frozen before observing any real-data v5 calibration fidelity
or task output.

The design used only preserved training-calibration engineering
diagnostics from failed v4 and synthetic structural capability tests.

No validation, outer-test, OnField, or fault evidence was used.

v5 remains mixed-precision static PTQ and must not be described as
fully INT8.
