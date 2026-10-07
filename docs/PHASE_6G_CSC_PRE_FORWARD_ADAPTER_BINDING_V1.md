# Phase 6G — CSC Pre-Forward Adapter Binding v1

## Status

`FROZEN_PRE_FORWARD_ADAPTER_BINDING_EXECUTION_DISABLED`

Phase 6G freezes the already-qualified pre-forward CSC adapter before any
callable combined sensor+compute execution runtime is introduced.

This is a governance and byte-binding freeze. It is not an execution gate.

## Frozen adapter

- `experiments/phase_06/csc_execution_adapter_v1.py`
- SHA256:
  `2d8d218b0d2214e099dab28b769eeb9fb90d6fa6420b50351b21e54cc4bd1fb5`

Its qualification test is:

- `tests/test_phase6g_csc_execution_adapter_pre_forward_v1.py`
- SHA256:
  `7a341af41a2e5e5986dc522620870a25615d368b01a1acba26c6f28d85e77cb8`

The frozen Phase-6E metadata executor remains byte-identical:

- `experiments/phase_06/csc_outer_executor_v1.py`
- SHA256:
  `99320b5811c5e72940bcfc779fa530df88fb8b7a2f8662711b8f56e4cf5d61fd`

## Execution boundary

The Phase-6G adapter remains execution-disabled.

It does not import Torch, NumPy, the sensor operator, or the Phase-5 outer
executor. It does not read outer arrays, load a checkpoint or PTQ model,
apply a sensor fault, execute a compute fault, perform a model forward, or
compute CSC metrics.

`execute_request(...)` remains a hard stop with:

`PHASE6G_EXECUTION_DISABLED_PRE_FORWARD_ADAPTER`

## Frozen composition semantics

The prospective executable implementation must preserve all of the following:

1. Sensor corruption precedes Phase-5 window-tensor conversion.
2. Source-trial sensor faults are applied to the source sequence and then
   rewindowed using the frozen Phase-4H v2 route.
3. Stored-window sensor faults remain on their frozen selected parent window.
4. Transient compute execution uses the one frozen R1-selected compute window.
5. Persistent compute execution uses the exact frozen Phase-5 onset-to-end
   suffix.
6. The frozen Phase-5 `FaultIdentity` and compute coordinates are preserved.
7. PTQ-v7 clean-state construction/restoration semantics are preserved.
8. The qint8 persistent-weight route restores clean PTQ state in `finally`.
9. Clean/reference forwarding is outside the compute-fault loop.
10. Persistent zero-overlap CSC pairs remain retained; they are not filtered,
    resampled, relocated, or rebalanced.

## Phase-4H source-helper binding

The frozen Phase-4H v2 runner owns `rewindow_sequence` locally.

Its source-loading/path helpers are explicit v2 exports from the frozen v1
runner:

- `load_source_trial = V1.load_source_trial`
- `source_trial_path = V1.source_trial_path`

The v1 owner file is:

- `experiments/phase_04/sensor_fi_devcal_runner.py`
- SHA256:
  `bee7b43423723d5ccdbf237d6dbcf70633d9db489d2aca35d1aca26c5100d398`

## Qualification entering the freeze

Immediately before this freeze:

- Phase-6G adapter suite: 14 passed.
- Phase-6D through Phase-6G suite: 77 passed.
- Full repository fleet: 927 passed, 1 historical Phase-5R lifecycle test
  deselected.

No performance outcome, validation result, OnField result, threshold selection,
checkpoint selection, resampling, or rebalancing informed the adapter.

## Next boundary

Any executable CSC runtime must be implemented separately.

Neither the frozen metadata executor nor this Phase-6G adapter may be modified
to enable execution. A later runtime must receive an explicit prospective
authorization/freeze before any CSC outer model forward occurs.
