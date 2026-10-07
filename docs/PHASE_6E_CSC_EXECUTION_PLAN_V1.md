# Phase 6E — Exact CSC Execution Plan V1

## Status

**FROZEN_PRE_EXECUTION_CSC_PLAN**

This artifact freezes the exact prospective outer-test execution surface for
combined sensor + compute faults (CSC). It does not execute any sensor fault,
compute fault, model forward, threshold comparison, or performance metric.

## Frozen lineage

The plan depends on the byte-frozen Phase 6D pairing protocol, Phase 6D-R1
temporal clarification, Phase 6D-R2 structural-omission amendment, the frozen
Phase 4H sensor sampler/executor metadata, and the frozen Phase 5D/5E compute
sampling and outer-plan contracts.

Any upstream hash mismatch is an execution abort.

## Exact post-R2 pair surface

- Original Phase-6D nominal pairs: 4,239,939.
- R2 structurally ineligible source-trial groups: 2,104.
- Retained source-trial CSC pairs: 130,385.
- Stored-window CSC pairs: 4,107,450.
- Exact model-independent CSC pairs: 4,237,835.
- Subject x sensor-family shards: 732.
- Zero-pair shards: 0.

Structural omissions are reported as missingness. They are not imputed,
replaced, relocated, resampled, or rebalanced.

## Compute strata and model variants

The 14 frozen Phase-5D targets x 2 persistence modes define 28 compute strata.
All 28 are nonzero.

Ten targets are eligible for both FP32 and PTQ-v7. Four targets are PTQ-v7
only. The resulting variant-expanded pair counts are:

- FP32: 3,026,511 pairs.
- PTQ-v7: 4,237,835 pairs.
- Combined variant-expanded: 7,264,346 pairs.

Across checkpoint seeds 42, 123, and 2025 this becomes exactly 21,793,038
pair-members. Pair-members are not model-window-forward counts.

## Exact temporal workload

The frozen A11 derivation binds:

- C0 clean-cache window cardinality: 1,642,980.
- Sensor-reference member windows: 35,167,107.
- Compute-faulted member windows: 411,540,372.
- Simultaneous CSC-overlap member windows: 19,926,021.
- Sensor/compute union member windows: 426,781,458.

The sum of sensor-reference and compute-faulted accounting is 446,707,479.
That sum is bookkeeping only and is not a required executor architecture.

## Transient semantics

For a source-trial sensor instance paired with transient compute FI, Phase
6D-R1 selects exactly one sensor-exposed evaluation window using its frozen
SHA256 rule. The Phase-5 transient compute sampling parent is that exact window.

For a stored-window sensor instance paired with transient compute FI, the
compute window is the same frozen trial-local stored-window index. No additional
hash selection is permitted.

Therefore every retained transient CSC pair has exactly one simultaneous
sensor+compute overlap window.

## Persistent semantics

The Phase-5 persistent compute identity remains trial-bound. Its exact
SHA256-derived onset is frozen by Phase-5 sampling, and its compute-active
window set is the onset-to-trial-end suffix.

Temporal CSC overlap is the intersection of that suffix with the frozen
sensor-exposed set.

There are 1,018,215 persistent pairs with zero simultaneous overlap,
corresponding to 5,234,355 model/seed pair-members:

- source-trial sensor parents: 13,275 pairs / 68,277 pair-members;
- stored-window sensor parents: 1,004,940 pairs / 5,166,078 pair-members.

These pairs remain in the execution plan. Zero overlap must be recorded and
must not trigger filtering, replacement, temporal relocation, resampling, or
rebalancing.

## Freeze bindings

The execution implementation must reproduce all frozen bindings before any
CSC model forward, including:

- pair-surface binding:
  `3b39a909ea140db75da52315abcf5efe2748fcdb2d31f0fc98ddf766b000af53`
- combined compute-coordinate binding:
  `2117c7b07084566607ccac46db150faae81d28bef51d0dcab7c700627d8eb435`
- temporal-workload binding:
  `5b447bca2f45d4889bcf0dc86d6170c40a1028171a9f1b0ea7a28e2884f12ba8`

The source/stored selection and assignment digests are also stored in the JSON
plan and are mandatory integrity checks.

## Governance

This freeze uses geometry, frozen identities, frozen sampling coordinates, and
frozen model-eligibility metadata only. It does not use CS, CC, CSC, validation,
OnField, or other held-out performance outcomes.

No threshold, checkpoint, sensor family, sensor severity, compute target,
persistence mode, fault coordinate, pair, or shard is selected using outer
performance feedback.

No MCU-equivalence or physical-realism claim is made.

## Next

Implement and qualify the Phase-6E CSC executor against this frozen plan.
No CSC model forward is authorized by this freeze command itself.
