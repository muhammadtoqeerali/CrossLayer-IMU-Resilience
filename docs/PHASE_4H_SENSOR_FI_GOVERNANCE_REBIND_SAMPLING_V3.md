# Phase 4H — Governance Rebind to Sampling v3

**Status:** sampling-v3 lineage rebind

## Trigger

Sampling v3 supersedes v2 for future execution because frame loss must
have a causal predecessor frame.

V3 changes only the minimum eligible frame-loss onset from sample 0 to
sample 1.

## Aggregation

The qualified aggregation v1 configuration and implementation are
preserved byte-for-byte.

No aggregation mathematics depends on the exact frame-loss onset index.

A new lineage manifest therefore binds the unchanged qualified
aggregation protocol to qualified sampling v3.

No change is made to:

- subject-level inference;
- nested replicate handling;
- variant macros;
- seed averaging;
- fold handling;
- metrics;
- paired degradation;
- bootstrap;
- OnField exclusion.

## Reporting

Reporting v1 explicitly required the gate:

`qualified_sampling_v2_hash_matches`

Reporting v2 replaces only that lineage key with:

`qualified_sampling_v3_hash_matches`

All performance interpretation, CI direction rules, model requirements,
fault-family/severity requirements, missingness handling, held-out
governance, claim boundaries, and OnField rules remain unchanged.

No arbitrary practical robustness margin is introduced.

## Remaining item

After this rebind, exactly one pre-operator governance item remains:

**executable jitter semantics**

The historical preprocessing path windows sensor rows at fixed 100 Hz
and does not use source timestamps to construct windows. Therefore a
timestamp-only mutation would be ineffective. Jitter must be assigned
an explicit software resampling/value semantics before the operator
engine is implemented.

No model prediction or sensor-fault execution is part of this rebind.
