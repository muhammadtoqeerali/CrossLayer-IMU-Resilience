# Phase 6D-R1 — CSC Pairing Clarification V1

## Status

**FROZEN_PRE_PLAN_CSC_PAIRING_CLARIFICATION**

This is a narrow pre-plan clarification to the already frozen Phase-6D CSC
pairing protocol.

It does **not** change the scientific fault surface, family set, severity set,
compute target set, persistence modes, model variants, checkpoints, thresholds,
or outer population.

No CSC pairs had been materialized and no CSC model forward had occurred before
this clarification.

## Why this clarification exists

Phase 6D already required a source-trial sensor fault paired with a transient
compute fault to select exactly one sensor-exposed evaluation window by SHA256.

The scientific rule was frozen, but the exact JSON serialization and digest
byte-to-index rule were not machine-explicit.

Because the chosen evaluation window determines the exact Phase-5 compute
sampling coordinate, this detail is frozen here before Phase-6E plan
derivation.

## Exact historical window geometry

The frozen Phase-4H rewindow path uses a 30-sample half-open Python slice:

`sequence[start:start+30]`

Therefore the stored historical window-end value is an **exclusive stop**.

For historical end `e`:

- window start = `e - 30`
- window stop exclusive = `e`
- inclusive source support = `[e-30, e-1]`

## Exact sensor active support

For finite-duration sensor faults:

- start = `onset_sample`
- stop exclusive = `onset_sample + duration_samples`
- inclusive support =
  `[onset_sample, onset_sample + duration_samples - 1]`

For until-end sensor faults:

- start = `onset_sample`
- stop exclusive = `parent_length`
- inclusive support = `[onset_sample, parent_length - 1]`

## Sensor-exposed evaluation window

A window is sensor-exposed exactly when:

`window_start < fault_stop_exclusive and fault_start < window_stop_exclusive`

The exposed-window set is ordered by ascending trial-local stored
`window_index`.

No model outputs or performance outcomes are required.

## Source-trial + transient-compute selection

Namespace:

`crosslayer-phase6d-r1-source-trial-transient-window-v1`

Canonical payload fields:

1. `namespace`
2. `sensor_replay_id`
3. `target_name`

Serialization is UTF-8 JSON with:

- `sort_keys=True`
- `separators=(',', ':')`
- `ensure_ascii=False`

The SHA256 digest is computed over those exact bytes.

The first eight digest bytes are interpreted as an unsigned big-endian integer.

The selected position is:

`digest_u64 % len(ordered_exposed_window_indices)`

The selected compute window is the corresponding entry of the ordered exposed
set.

A zero-length exposed set aborts Phase-6E plan derivation with no fallback,
resampling, or replacement.

## Stored-window + transient-compute selection

The compute window is exactly the same trial-local window as the frozen
sensor-parent window.

No hash selection is used.

## Persistent compute overlap

The frozen Phase-5 persistent compute fault is active from its exact
`persistent_onset_index` through the final trial window.

For stored-window sensor faults, the sensor-exposed set is the singleton parent
window.

For source-trial sensor faults, it is the complete geometry-derived exposed
window set.

The temporal-overlap set is the intersection of sensor-exposed windows and
persistent-compute-active windows.

The plan records both:

- `temporal_overlap = len(overlap_set) > 0`
- `temporal_overlap_window_count = len(overlap_set)`

Pairs with zero overlap are retained. They are never filtered or resampled.

## Governance

This clarification was frozen before:

- CSC pair materialization;
- CSC model execution;
- CSC metric computation;
- any CSC result interpretation.

No CS or CC performance outcome was used.

No threshold, checkpoint, sensor family, severity, compute target, bit,
persistence mode, model variant, or pair was selected using held-out results.

No OnField, MCU-equivalence, or physical-realism claim is made.

## Next

Phase 6E must derive and freeze the exact CSC pair inventory and execution
workload using both the original Phase-6D protocol and this Phase-6D-R1
clarification.
