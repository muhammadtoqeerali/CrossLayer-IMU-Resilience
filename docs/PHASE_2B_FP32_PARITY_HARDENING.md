# Phase 2B FP32 ONNX Parity Hardening

## Status

PASS

## Why this audit was added

The initial parity run observed a maximum absolute PyTorch/ONNX difference of:

`6.103515625e-05`

This value is larger than the absolute tolerance component alone.

The combined parity criterion is:

`abs(error) <= 1e-5 + 1e-5 * abs(reference)`

The hardening audit checks that criterion element by element rather than
reporting only a global maximum absolute error.

## Results

Elementwise tolerance violations:

0

Structured-vector maximum absolute difference:

`7.748603820800781e-07`

Seeded numerical-stress maximum absolute difference:

`6.103515625e-05`

Maximum relative difference:

`3.713452542797313e-06`

Argmax-decision differences:

0

Historical-decision differences:

0

## Worst absolute-error element

Partition:

`seeded_random_stress`

Sample:

122

Class:

0

PyTorch logit:

`73.2707290649414`

ONNX logit:

`73.27066802978516`

Absolute difference:

`6.103515625e-05`

Relative difference:

`8.330086984642548e-07`

Allowed combined tolerance:

`0.0007427072850987315`

## Repeat-export audit

Byte-identical repeated export:

True

Graph and initializer identity:

True

Runtime outputs bit-identical:

True

## Candidate FP32 parity policy

For the structured deterministic vectors:

`max absolute error <= 1e-5`

For the seeded numerical stress vectors:

`abs(error) <= 1e-5 + 1e-5 * abs(reference)`

For all vectors:

- zero elementwise criterion violations
- zero argmax-decision differences
- zero historical-decision differences

The seeded stress vectors are numerical representation checks only.

They are not task-performance or quantization-calibration evidence.
