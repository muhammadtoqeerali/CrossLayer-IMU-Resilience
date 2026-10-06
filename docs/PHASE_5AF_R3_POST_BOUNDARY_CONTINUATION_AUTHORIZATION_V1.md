# Phase 5AF-R3 — Post-Boundary Continuation Authorization V1

## Status

**FROZEN_POST_BOUNDARY_CC_CONTINUATION_AUTHORIZATION**

## Context

Phase 5AF crossed the one-way accepted-outcome boundary under the original
Phase-5AE activation.

Before any shard completed, execution stopped because the original V1 analysis
consumer could not decode the already-frozen Phase-5M/5R JSON representation
for non-finite floating values.

Phase 5AF-R1 qualified the minimal non-finite decoding repair.

Phase 5AF-R2 qualified the full V2 execution chain:

`cc_outcome_analyzer_v2.py`
→ `cc_outcome_io_runner_v2.py`
→ `cc_outcome_executor_v2.py`.

Phase 5AF-R3 authorizes continuation with exactly that chain.

## Original run remains immutable

The original `execution_start.json` is hash-bound by R3 and may not be
modified.

It records that:

- the one-way boundary was crossed under the original Phase-5AE activation;
- the original V1 executor initiated the run;
- scientific rules became immutable after first payload access.

There were zero completed shards before the repair.

## Nature of the repair

The continuation replaces only the implementation chain needed to restore the
producer's already-frozen serialized IEEE values.

It does not change scientific semantics.

The exact repaired behavior is:

- ordinary JSON numbers remain ordinary floating values;
- `{"nonfinite":"nan"}` restores IEEE NaN;
- `{"nonfinite":"+inf"}` restores IEEE positive infinity;
- `{"nonfinite":"-inf"}` restores IEEE negative infinity;
- malformed or unknown token dictionaries abort.

No imputation or clipping is permitted.

## Authorized continuation

R3 authorizes the exact V2 chain to continue the already-started Phase-5AF run
using:

- the same accepted outer estate;
- the same outer dataset;
- the same risk/timing index;
- the same Phase-5E shard membership and order;
- the same Phase-5Z thresholds, metrics, event semantics, aggregation, and
  uncertainty rules.

The V1 executor is not authorized to resume production after R3.

## Resume provenance

Continuation must create a separate continuation-start marker.

That marker must bind:

- the R3 authorization hash;
- the exact V2 executor hash;
- the original immutable `execution_start.json` hash.

Any later shard reuse must bind the same R3 authorization and V2 executor.

Partial or unverified results may not be reused.

## Scientific invariants

The repair does not authorize changes to:

- thresholds;
- comparator semantics;
- metrics;
- aggregation;
- uncertainty;
- strata;
- checkpoint membership;
- fault membership;
- transient/persistent membership;
- FP32/PTQ separation.

Only C0 and CC remain authorized.

CSC remains excluded.

OnField remains excluded.

No model forward or new fault execution is authorized.

## R3 boundary

R3 itself performs no shard resume.

It opens no accepted prediction payload, loads no outer label array, applies
no threshold, and computes no CC metric.

The next step is the governed continuation of Phase 5AF with the exact
R3-bound V2 chain.
