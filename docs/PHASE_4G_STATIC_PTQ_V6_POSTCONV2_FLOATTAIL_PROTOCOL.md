# Phase 4G — Static PTQ v6: FP32 Post-Conv2 Tail

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Preserved v5 result

v5 remains a failed candidate.

Its structural and serialization gates passed, while the unchanged
p99 absolute softmax-probability fidelity gate failed.

No numerical acceptance threshold is changed in v6.

## v5 causal localization

The v5 FP32 front end contributed exactly zero final-output error.

The second quantized Conv+BN boundary remained below the frozen p99
limit.

The first breach occurred at the `conv2_pool` stage.

## Post-PReLU / MaxPool decomposition

The residual stage was decomposed into:

1. FP32 `conv_2.2` output followed by FP32 MaxPool;
2. the same `conv_2.2` output after the actual v5 activation
   quantize/dequantize boundary followed by FP32 MaxPool;
3. the actual quantized MaxPool output.

The MaxPool integer kernel was exactly equivalent to FP32 MaxPool on
the dequantized quantized input.

The p99 increase was therefore attributed to activation
requantization after `conv_2.2`, not to MaxPool computation.

## Single v6 change

Relative to v5, v6 moves only the second activation-quantization
boundary.

v5:

`conv_2.2 -> quantize -> MaxPool -> Dropout -> ...`

v6:

`conv_2.2 -> MaxPool -> Dropout -> Flatten -> quantize -> fc.1`

Thus `conv_2.3`, `conv_2.4`, and `fc.0` remain FP32.

The first v5 boundary after the opaque FP32 front end remains
unchanged, `conv_2.0` remains quantized, `fc.1` remains quantized,
the later v5 requantization after `fc.2` remains, and `fc.4` remains
FP32.

## Synthetic pre-result capability evidence

An initial structural audit incorrectly reported failure because its
global reachability check counted the desired quantizer after `fc.0`
and the unrelated later quantizer after `fc.2`.

The candidate mechanism itself was not changed.

A corrected direct-edge audit confirmed:

- `conv_2.2 -> conv_2.3` is direct FP32;
- `conv_2.3 -> conv_2.4` is direct FP32;
- `conv_2.4 -> fc.0` is direct FP32;
- each direct tensor handoff is exact;
- quantization begins immediately after `fc.0`;
- `fc.1` remains quantized;
- the later quantization after `fc.2` is preserved.

No real dataset sample or v6 task output was observed.

## Numerical acceptance

Unchanged:

- all outputs finite;
- exact output shape;
- argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Scientific boundary

v6 is frozen before any real v6 calibration-fidelity result.

No validation, outer-test, OnField, or fault evidence is used for v6
design.

v6 is mixed-precision static PTQ and must not be described as fully
INT8.
