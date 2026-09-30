# Phase 2C Quantization Support Audit

## Status

PASS

## Baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

The baseline remains unfrozen.

## Installed ONNX Runtime quantization capability

ONNX Runtime version: `1.22.0`

Available calibration methods:

- `Distribution`
- `Entropy`
- `MinMax`
- `Percentile`

Available quantization formats:

- `QDQ`
- `QOperator`

Available integer types:

- `QFLOAT8E4M3FN`
- `QInt16`
- `QInt4`
- `QInt8`
- `QUInt16`
- `QUInt4`
- `QUInt8`

## Protected graph operator support

| Operator | Count | QLinear registry | Integer registry | QDQ registry |
|---|---:|---:|---:|---:|
| `Concat` | 1 | True | False | False |
| `Constant` | 4 | False | False | False |
| `Conv` | 2 | True | True | True |
| `Div` | 3 | False | False | False |
| `Flatten` | 1 | False | False | False |
| `Gemm` | 2 | True | False | True |
| `MaxPool` | 2 | True | False | True |
| `PRelu` | 3 | False | False | False |
| `Split` | 1 | True | False | True |
| `Transpose` | 1 | True | True | True |

Registry membership is toolchain capability evidence only.

It does not prove that the final deployment backend will execute every quantized operator efficiently.

## Dynamic-quantization smoke test

Status: `FAILED_NONBLOCKING`

Role: engineering toolchain smoke only.

It is not the final INT8 deployment candidate.

## Primary Phase-2 quantization direction

Static post-training quantization is the primary candidate.

Final activation calibration is deferred until a representative clean calibration partition is frozen.

This preserves the recovered historical task weights and avoids leakage from final-test or fault-study data.

QAT is not silently substituted if PTQ is inadequate because QAT changes the recovered historical task weights.

## Data boundary

No project dataset sample was used in this audit.
