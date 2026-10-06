# Phase 5AF-R1 — Post-Boundary Nonfinite Softmax Adapter Repair V1

## Status

**QUALIFIED_POST_BOUNDARY_NONFINITE_ADAPTER_REPAIR**

## Context

The authorized Phase-5AF production execution crossed the one-way prospective
outcome boundary and then failed before completing any shard.

The failure was caused by an implementation compatibility gap.

The frozen Phase-5M and Phase-5R producers serialize non-finite floating
values as one-key JSON dictionaries:

- `{"nonfinite":"nan"}`
- `{"nonfinite":"+inf"}`
- `{"nonfinite":"-inf"}`

The accepted-estate representation audit scanned all 31,441,800 accepted
records and observed 61,752 dict-valued softmax elements.

All observed dict-valued softmax elements:

- occurred in `faulted_softmax_values`;
- had exactly the key `nonfinite`;
- carried the token `nan`.

No unexpected dictionary shape was observed.

## Scientific contract

No scientific rule changes.

Phase-5Z already requires non-finite stored values to be preserved without
imputation and evaluated with the exact historical threshold comparator.

Therefore decoding `{"nonfinite":"nan"}` to IEEE NaN restores the producer's
stored value.

It does not impute, clip, replace, retune, or reinterpret the value.

Under the frozen comparator, `NaN >= threshold` is false.

## Repair

The historical `cc_outcome_analyzer_v1.py` is preserved byte-identical.

A versioned `cc_outcome_analyzer_v2.py` is created.

V2 adds only an exact serialized-float decoder and routes softmax elements
through it before constructing the two-element NumPy vector.

Finite JSON numbers preserve V1 behavior.

Unknown dictionary shapes and unknown non-finite tokens abort.

## Qualification

Synthetic qualification proves:

- finite V1/V2 probability equivalence;
- NaN token restoration;
- positive-infinity token restoration;
- negative-infinity token restoration;
- rejection of unknown token values;
- rejection of unexpected dictionary shapes;
- frozen NaN threshold behavior;
- NaN scenario reconstruction;
- unchanged historical event-metric semantics for already-decoded NaNs.

## Boundary

The repair qualification does not resume Phase 5AF.

It applies no accepted threshold and computes no accepted CC metric.

No scientific protocol, metric, aggregation, uncertainty, stratum, checkpoint,
fault membership, CSC rule, or OnField rule changes.

A repaired final executor variant must next be qualified against this exact V2
analyzer and then receive an explicit post-boundary continuation
authorization before Phase-5AF resumes.
