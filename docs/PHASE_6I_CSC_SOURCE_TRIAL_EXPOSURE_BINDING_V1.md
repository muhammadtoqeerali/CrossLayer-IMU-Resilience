# Phase 6I — CSC Source-Trial Exposure Helper Binding v1

## Status

`FROZEN_METADATA_ONLY_SOURCE_TRIAL_EXPOSURE_HELPER_BINDING`

Phase 6I freezes the qualified metadata-only source-trial exposure helper.

This freeze does not authorize CSC execution and does not change the
scientific surface.

## Frozen helper

- `experiments/phase_06/csc_source_trial_exposure_v1.py`
- SHA256:
  `f4db0c1dfecd5992deed7947665c5a8afc8d60db28f97457ac7ad11bc24b0aeb`

Qualification test:

- `tests/test_phase6i_csc_source_trial_exposure_v1.py`
- SHA256:
  `8335c3c9965d1471ed4b9bbd2f021c15fa198da5125d0ca0402fa6d3c81f4624`

## Why this helper exists

The pre-implementation probes established that the frozen repository had the
R2 source-trial selector and consumers of exposure indices/counts, but no
committed producer of `exposure_count_by_fault_id`.

The exact producer semantics were nevertheless already prospectively fixed by
the frozen Phase-4H sampling/operator contracts and Phase-6D-R1/R2 geometry.

This helper materializes only that missing metadata transformation.

## Active-support semantics

Finite sequence episodes:

- drift
- stuck channel
- dropout
- frame loss

use the half-open support:

`[onset_sample, onset_sample + duration_samples)`

Full-trial sequence families:

- jitter
- delay
- orientation

use:

`[0, source_length)`

Their onset is zero and their duration is null.

## Retained-window exposure

Each stored evaluation window is:

`[historical_window_end - 30, historical_window_end)`

A window is sensor-exposed exactly when:

`window_start < fault_stop_exclusive AND fault_start < window_stop_exclusive`

The resulting indices are stored evaluation `window_index` values in ascending
order.

## R2 handoff

`source_trial_exposure_census(...)` produces:

- `sensor_exposed_window_indices_by_fault_id`
- `exposure_count_by_fault_id`

The frozen Phase-6E metadata executor then consumes the count map through
`select_source_trial_candidate(...)`.

Eligibility remains:

`sensor_exposed_window_count >= 1`

If no candidate is observable, the frozen status remains:

`STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR`

There is no fallback, resampling, replacement, temporal relocation, or
rebalancing.

## Execution boundary

The helper is metadata-only.

It does not import Torch or NumPy, read outer arrays, apply sensor faults,
load models, execute compute faults, perform model forwards, materialize pair
files, or compute performance metrics.

## Qualification entering the freeze

Immediately before this freeze:

- dedicated Phase-6I exposure-helper tests: 14 passed;
- full Phase-6 regression set: 116 passed;
- full repository fleet: 966 passed;
- historical Phase-5R lifecycle test: 1 deselected.

No validation, OnField, or outer performance outcome informed the helper.

## Next boundary

An execution-capable CSC runtime may now be implemented as a separate new
file, but it must remain unauthorized.

Before any CSC outer model forward, a later gate must freeze the runtime bytes,
its qualification tests, all 732 shard identities, exact per-shard record
cardinalities, and the validated Phase-5 clean-cache estate.
