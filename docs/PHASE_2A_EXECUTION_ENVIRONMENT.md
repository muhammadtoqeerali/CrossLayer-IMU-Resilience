# Phase 2A Execution Environment Audit

## Status

PASS

## Baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

Freeze status:

`NOT_FROZEN`

Canonical tensor-state SHA-256:

`c98987476536320191f8875316cf8caeeb0b3f2edc51d65be7ac7d2eaea03124`

Parameters:

63,173

## Storage

FP32 parameter bytes:

252,692

FP32 parameter MiB:

0.240986

Buffers are reported separately from trainable parameters.

## Structural compute

Conv1D + Linear MAC estimate for one 400-ms window:

147,712

This definition excludes preprocessing, normalization, activations, pooling
and softmax.

## Deterministic decision audit

Reference vectors:

68

Argmax versus historical streaming decisions that differ:

0

This difference must be resolved before decision-semantic freeze.

No rule will be selected using later fault-test outcomes.

## Export capability

ONNX installed:

True

ONNX Runtime installed:

True

ONNX Runtime quantization module installed:

True

## PyTorch quantization capability

torch.ao.quantization:

True

FX quantization:

True

Supported engines:

`['qnnpack', 'none', 'onednn', 'x86', 'fbgemm']`

Current engine:

`x86`

These are capability observations only.

They do not prove that the protected model has been successfully quantized.

## Host reference timing

Scope:

development workstation CPU reference only

Mean:

0.096152 ms

Median:

0.094656 ms

P95:

0.101810 ms

P99:

0.111378 ms

These values are not MCU evidence.

## Still unfrozen

- primary task decision semantics
- ONNX deployment artifact
- INT8 deployment artifact
- quantization acceptance criteria
- dataset-level clean metrics
- MCU target
