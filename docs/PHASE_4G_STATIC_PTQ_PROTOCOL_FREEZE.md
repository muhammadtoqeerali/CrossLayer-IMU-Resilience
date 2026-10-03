# Phase 4G — Static PTQ Protocol Freeze

**Status:** FROZEN_PRE_RESULT
**Date:** 2026-10-02

This document freezes the prospective 300-ms static post-training
quantization protocol **before any quantized output is observed**.

## Calibration closure

The frozen Phase-3 calibration identity artifact contains five
fold-specific sets of 4096 windows.

An independent Phase-4G replay gate verified all 20,480 fold-window
instances against the same primary 300-ms dataset used for Phase 4E.

For every fold:

- calibration subjects exactly equal the frozen training subjects;
- validation-subject overlap is zero;
- outer-test-subject overlap is zero;
- OnField is not used;
- all selection SHA-256 identities reconstruct exactly;
- all selected windows resolve to `segments.npy`;
- all selected tensors are finite and shape `(30, 9)`;
- labels replay exactly.

## Frozen realization

Exactly one v1 quantization realization is declared:

`torch_fx_x86_default_qconfig_v1`

Method:

- static PTQ;
- PyTorch FX;
- backend `x86`;
- `get_default_qconfig_mapping("x86")`;
- activation dtype `quint8`;
- weight dtype `qint8`;
- activation observer `torch.ao.quantization.observer.HistogramObserver`;
- weight observer `torch.ao.quantization.observer.PerChannelMinMaxObserver`.

There is no post-result calibrator/backend/granularity sweep in v1.

## Numerical fidelity gates

Using only the corresponding fold's frozen 4096 training-only
calibration windows, the quantized checkpoint must satisfy:

- all outputs finite;
- output shape exactly preserved;
- FP32/quantized argmax agreement >= 0.99;
- mean absolute softmax-probability error <= 0.01;
- 99th-percentile absolute softmax-probability error <= 0.05.

These are predeclared engineering fidelity gates, not physical or
clinical safety thresholds.

## Operator coverage and precision language

Every Conv1d and Linear operator must be audited after conversion.

PReLU coverage must also be audited explicitly.

A model may be described as **fully INT8** only if the converted graph
shows no internal FP32 arithmetic island or internal dequantize/requantize
boundary across the model body, apart from explicitly permitted model
input/output boundaries.

If weighted operators are quantized but PReLU or another arithmetic
operator remains FP32, the result is retained only as a
**mixed-precision static PTQ realization**.

No MCU execution or hardware-performance claim follows from this phase.

## Leakage boundary

Quantizer choice and quantization parameters may not use:

- validation predictions;
- outer-test predictions;
- OnField;
- sensor-fault data;
- compute-fault data;
- combined-fault data;
- future physical-fault evidence.

If this v1 realization fails, the failure is preserved. Any alternative
observer, backend, format, weight granularity, or QAT route requires a
new versioned protocol rather than silent tuning.
