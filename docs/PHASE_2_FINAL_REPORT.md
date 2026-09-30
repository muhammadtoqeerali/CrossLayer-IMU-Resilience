# Phase 2 Final Report

## Status

COMPLETE

## Frozen protected reference

`DATE2025_CNN_400MS_RECONSTRUCTED`

Freeze scope:

`FP32_REFERENCE_BASELINE`

## Model identity

Parameters:

63,173

Input:

`[1, 40, 9]`

Effective CNN channels:

6

Output:

`[1, 2]`

Sampling:

100 Hz

Window:

400 ms

FP32 parameter storage:

252,692 bytes

Conv/Linear MACs:

147,712

Canonical tensor-state SHA-256:

`c98987476536320191f8875316cf8caeeb0b3f2edc51d65be7ac7d2eaea03124`

## Primary task decision

The primary decision is frozen to the recovered historical streaming rule:

`Falling iff P(Falling) > 0.9`

The comparison is strict.

Argmax is retained only as a secondary diagnostic.

This decision was frozen for historical fidelity before fault-study outcomes
exist.

## FP32 ONNX reference

SHA-256:

`f3a55933bab5ed63225a1647a8b78aa909c4bb5a81405b06f235ba1b3bf6de1f`

Bytes:

255,191

Opset:

13

Deployment batch:

1

Operator inventory:

`{'Concat': 1, 'Constant': 4, 'Conv': 2, 'Div': 3, 'Flatten': 1, 'Gemm': 2, 'MaxPool': 2, 'PRelu': 3, 'Split': 1, 'Transpose': 1}`

## FP32 parity

Structured deterministic vectors:

68

Numerical stress vectors:

256

Elementwise tolerance violations:

0

Structured maximum absolute difference:

`7.748603820800781e-07`

Stress maximum absolute difference:

`6.103515625e-05`

Maximum relative difference:

`3.713452542797313e-06`

Argmax decision differences:

0

Historical decision differences:

0

Repeated export byte-identical:

True

Repeated runtime outputs bit-identical:

True

## Quantization protocol

Primary method:

static post-training quantization

Protocol status:

frozen

Final calibrated INT8 status:

`DEFERRED_UNTIL_DATA_PROTOCOL_FREEZE`

Reason:

Activation calibration requires representative clean windows from a frozen
development/calibration partition.

No final-test or fault-study evidence may be used to select quantizer
parameters.

QAT cannot be silently substituted if PTQ is unsuitable because that would
change the recovered historical task weights.

## Dynamic quantization smoke

Status:

`FAILED_NONBLOCKING`

Scientific role:

toolchain smoke only

Deployment candidate:

False

Recorded nonblocking error:

`NotImplemented("[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/conv_1/conv_1.0/Conv_quant'")`

A failed dynamic smoke test does not block the static-PTQ protocol.

## Phase dependency correction

The original conceptual plan placed quantization before the formal data split.

Implementation showed that activation calibration depends on the frozen clean
calibration partition.

The corrected execution order is:

1. frozen FP32 task reference
2. frozen quantization protocol
3. frozen dataset/split/calibration protocol
4. calibrated static INT8 realization
5. fault characterization

## Scientific boundary

Phase 2 used no final task test results to choose:

- architecture
- primary decision rule
- ONNX representation
- FP32 parity policy
- quantization method

No sensor-, compute- or combined-fault outcome influenced these choices.

## Next phase

Phase 3 freezes:

- dataset provenance
- subject/event/sequence independence units
- partitions
- calibration subset
- timing semantics
- labels
- window generation
- leakage safeguards
