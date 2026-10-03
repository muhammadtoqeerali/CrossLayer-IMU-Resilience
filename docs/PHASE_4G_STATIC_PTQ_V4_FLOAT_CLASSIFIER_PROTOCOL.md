# Phase 4G — Static PTQ v4: FP32 Final Classifier

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

## Preserved predecessor result

v3 is preserved as a failed candidate.

Its structural gate passed and serialization passed, but the unchanged
numerical fidelity gate failed because the p99 absolute softmax
probability error exceeded 0.05.

No v3 acceptance threshold is being relaxed.

## v3 final-classifier diagnosis

The final v3 classifier `fc.4` was a QuantizedLinear with output scale
approximately 0.2999878824.

Because this is a two-logit classifier with a common output scale, its
softmax probability is restricted to the lattice

`sigmoid((fall_code - activity_code) * output_scale)`.

The 25 worst recorded v3 calibration probabilities matched this lattice
to within 1e-6.

The largest adjacent probability step on the lattice was approximately
0.07444, already larger than the frozen p99 probability-error limit of
0.05.

This establishes a direct representation-level mechanism for the v3
probability tail.

## Synthetic structural capability

Before v4 was frozen, a synthetic-only structural probe tested one
mechanism: leave `fc.4` in FP32 while retaining the rest of v3.

That probe converted successfully with:

- two quantized Conv1d layers;
- one quantized Linear (`fc.1`);
- three FP32 FloatPReLU wrappers;
- one FP32 final Linear (`fc.4`);
- a dequantization boundary before `fc.4`;
- direct FP32 output after `fc.4`.

No real dataset sample or task metric was evaluated by that capability
probe.

## Single v4 change

v4 makes exactly one precision-placement change from v3:

`fc.4` is excluded from quantization and executes in FP32.

The implementation rule is:

`build_v3_qconfig_mapping().set_module_name("fc.4", None)`

Everything else remains unchanged.

## Unchanged quantized regions

v4 still requires:

- `conv_1.0` QuantizedConv1d;
- `conv_2.0` QuantizedConv1d;
- `fc.1` QuantizedLinear.

The three PReLU operations remain the same hash-frozen FloatPReLU
implementation introduced in v3.

## Numerical acceptance

The numerical gates are unchanged:

- all outputs finite;
- exact output shape;
- FP32/v4 argmax agreement >= 0.99;
- mean absolute softmax probability error <= 0.01;
- p99 absolute softmax probability error <= 0.05.

## Precision language

v4 is mixed-precision static PTQ.

It must not be called fully INT8 because:

- all three PReLUs execute in FP32; and
- the final classifier `fc.4` executes in FP32.

## Scientific boundary

v4 was frozen before observing any real-data v4 calibration fidelity.

Its design uses only:

- preserved v3 training-calibration engineering evidence; and
- a synthetic structural capability probe.

No validation, outer-test, OnField, or fault evidence was used to
design v4.

If the seed-42/fold-1 v4 pilot fails, that failure is preserved and any
further change requires another explicitly versioned pre-result
protocol.
