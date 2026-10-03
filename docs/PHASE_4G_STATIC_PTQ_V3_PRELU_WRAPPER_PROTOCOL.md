# Phase 4G — Static PTQ v3: Float PReLU Wrapper

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Lineage

v1 converted successfully but failed the frozen numerical fidelity gates.
Its native QuantizedPReLU implementation was subsequently shown to violate
the expected positive PReLU branch in an operator-level microtest.

v2 attempted to keep native `nn.PReLU` in FP32 by assigning it no qconfig.
In this installed PyTorch FX runtime, `convert_fx` nevertheless entered the
native QuantizedPReLU lowering path and failed because the reference module
had `qconfig=None`.

That v2 conversion failure is preserved.

## Synthetic capability probe

Before v3 was frozen, a structural-only probe using deterministic synthetic
input evaluated implementation mechanisms without reading any dataset
window or task metric.

The opaque float-wrapper mechanism converted successfully with:

- 2 QuantizedConv1d modules;
- 2 QuantizedLinear modules;
- 0 QuantizedPReLU modules;
- 3 FloatPReLU wrappers;
- explicit dequantize/requantize boundaries around the float PReLU regions.

No real calibration, validation, outer-test, OnField, or fault sample was
used by that capability probe.

## Frozen v3 mechanism

v3 replaces each native scalar `nn.PReLU` with a mathematically equivalent
`FloatPReLU`:

`torch.where(x >= 0, x, weight * x)`

The learned slope tensor is cloned exactly from the frozen FP32 checkpoint
and exact tensor equality is required.

`FloatPReLU` is:

1. assigned `qconfig=None`; and
2. declared non-traceable through `PrepareCustomConfig`.

This forces explicit Q/DQ transitions while preserving the learned PReLU
operation in FP32.

The implementation source itself is frozen by SHA-256 in the v3 protocol
manifest.

## What remains unchanged

v3 does not change:

- CNN task weights;
- mathematical PReLU function;
- x86 backend;
- default x86 qconfig for eligible operators;
- Conv1d/Linear quantization targets;
- observer families or integer types;
- the 4096 frozen training-only calibration identities per fold;
- the 15-checkpoint FP32 estate;
- numerical fidelity thresholds.

## Numerical acceptance

The unchanged gates are:

- all outputs finite;
- exact output-shape preservation;
- FP32/PTQ argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Precision language

v3 is **mixed-precision static PTQ by construction**.

It must never be labeled fully INT8 because all three PReLU operations are
intentionally executed in FP32.

## Scientific boundary

No real-data v3 calibration fidelity or task metric was observed before this
freeze.

The next action is a single seed-42/fold-1 pilot using only that fold's
frozen 4096 training calibration windows.

If v3 fails, the failure is preserved and any subsequent change requires a
new versioned protocol.
