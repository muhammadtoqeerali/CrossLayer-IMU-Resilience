# Phase 2B FP32 ONNX Validation

## Status

PASS

## Protected baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

Freeze status:

`NOT_FROZEN`

Canonical tensor-state SHA-256:

`c98987476536320191f8875316cf8caeeb0b3f2edc51d65be7ac7d2eaea03124`

## Deployment tensor contract

Input:

`imu_window`

Shape:

`[1, 40, 9]`

Output:

`logits`

Shape:

`[1, 2]`

Deployment batch:

1

The fixed batch-1 contract is intentional because the target workload is
single-window embedded inference.

## Export

Format:

ONNX

Precision:

FP32

Opset:

13

Artifact SHA-256:

`f3a55933bab5ed63225a1647a8b78aa909c4bb5a81405b06f235ba1b3bf6de1f`

Artifact bytes:

255,191

The ONNX binary is generated locally and is excluded from Git.

## ONNX graph

Node count:

20

Initializer count:

11

Operator counts:

`{'Concat': 1, 'Constant': 4, 'Conv': 2, 'Div': 3, 'Flatten': 1, 'Gemm': 2, 'MaxPool': 2, 'PRelu': 3, 'Split': 1, 'Transpose': 1}`

## Numerical parity

Structured vectors:

68

Seeded numerical stress vectors:

256

Total vectors:

324

ONNX parity execution mode:

`fixed_batch_1_serial`

Each vector was run individually through the fixed deployment graph.

Maximum absolute PyTorch/ONNX difference:

6.103515625e-05

Mean absolute difference:

9.670209692558274e-06

Maximum relative difference:

3.713452542797313e-06

Absolute tolerance:

1e-05

Relative tolerance:

1e-05

Argmax-decision parity:

True

Historical-decision parity:

True

## Tracing warnings

PyTorch emitted tracing warnings for fixed shape checks and constant
normalization tensors during ONNX export.

These warnings are consistent with this export contract.

The graph is intentionally fixed to `[1,40,9]`.

The preprocessing divisors are intentionally constant.

Graph validity and numerical parity are independently checked after export.

## Scientific boundary

The numerical stress vectors are synthetic parity inputs only.

They are not:

- training data
- calibration data
- validation data
- test data
- task-performance evidence

Final INT8 calibration remains deferred until the formal data/calibration
protocol is frozen.
