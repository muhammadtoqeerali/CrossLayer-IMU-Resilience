# Phase 5R — Full-Fleet Outer Compute-FI Executor Qualification v1

**Status:** QUALIFIED_FULL_FLEET_EXECUTOR_PRE_OUTER  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5R implements and qualifies the production executor required by the
frozen Phase-5P full-fleet completion gate.

Qualification is restricted to static analysis, frozen metadata replay, and
real training-calibration payloads.

No additional outer shard executes in Phase 5R.

## Frozen executor

The production executor is:

`experiments/phase_05/compute_fi_outer_fleet_executor_v1.py`

It is cryptographically bound to the Phase-5P completion gate and frozen
Phase-5E execution plan.

The executor accepts the exact Phase-5E estate only.

The already accepted Phase-5N canary is reuse-only and is not rerun by default.

## Four production execution classes

The executor supports the complete Phase-5E execution matrix:

- FP32 transient;
- FP32 persistent;
- PTQ-v7 transient;
- PTQ-v7 persistent.

The frozen plan contains 183 shards in each class.

The scientific target inventory remains:

- 10 FP32 targets;
- 14 PTQ-v7 targets.

## Deterministic identity

The production executor reuses the frozen Phase-5D sampler for:

- canonical parent-bound sampling payload;
- `sampling_instance_id`;
- deterministic element index;
- deterministic bit position;
- persistent onset;
- Phase-5A mutation-specification `fault_id`;
- parent/variant/seed-bound `outer_instance_id`.

Transient identities are window-parent-bound.

Persistent identities are trial-parent-bound.

Before execution, persistent per-target onset-binding digests and expected
post-onset forward counts are rederived and checked against Phase-5E.

## Fault-only execution

Production fault execution uses only qualified Phase-5K interfaces:

- `run_fp32_fault_only`;
- `run_ptq_activation_buffer_fault_only`;
- `PTQWeightFaultOnlySession.run_fault_only`.

No paired execution runner is used.

No embedded clean-reference forward occurs inside the production fault loop.

## Persistent execution

For persistent activation and buffer faults, one trial-target identity is
reused from its frozen onset through trial end.

For persistent PTQ qint8 weight faults, the same frozen element and bit are
reapplied from clean model state at each post-onset inference.

The PTQ weight session is reset to clean state after every sequence, preventing
cross-trial persistent-state leakage.

## Clean-cache and atomic semantics

A clean cache is shared by its transient and persistent sibling shards.

The accepted Phase-5N canary clean/fault artifacts are validated against the
Phase-5O frozen artifact hashes and reused.

New artifacts use the qualified Phase-5F atomic/resume contract.

## Static qualification

Static source auditing established:

- validation paths contain no `np.load`;
- validation paths contain no model loader;
- validation paths contain no fault execution;
- `_load_outer_trial_signal()` is the only outer-array loader;
- no paired fault runner is delegated to;
- no `labels.npy` access is implemented.

## Training-calibration qualification

Qualification uses the frozen fold-1 training-only calibration identity.

Five real 300 ms windows were selected from its first frozen lexicographic
calibration member:

- subject 9;
- task 1;
- trial 1.

Six production orchestration routes were executed:

1. FP32 transient activation;
2. FP32 persistent buffer;
3. PTQ transient quantized activation;
4. PTQ persistent quantized buffer;
5. PTQ transient qint8 weight;
6. PTQ persistent qint8 weight.

This produced exactly 30 fault-only training-calibration sequence executions.

The transient active mask was:

`[True, True, True, True, True]`

The persistent active mask with onset index 2 was:

`[False, False, True, True, True]`

PTQ eager clean inference matched the frozen TorchScript reference bitwise.

PTQ qint8 weight state reset passed after both transient and persistent
qualification routes.

Source artifacts remained unchanged.

The accepted outer canary artifacts remained unchanged.

## Qualification schema repair

The first qualification attempt stopped before executing any qualification
case because the qualifier referenced a non-existent Phase-5K
`scientific_boundary` key:

`onfield_payload_used`

A read-only schema probe established that the actual frozen Phase-5K key is:

`onfield_used`

The qualifier assertion alone was repaired to use the actual Phase-5K key.

The Phase-5R result schema intentionally retains its own field
`onfield_payload_used = false`.

No executor logic, fault protocol, qualification case matrix, partition,
sampling, identity, target, persistence, authorization, or outer estate changed
during this repair.

## Scientific boundary

Qualification used training-calibration payloads only.

It used no validation payload.

It read no additional outer-test payload.

It used no OnField payload.

It applied no threshold.

It computed no metric.

It performed no prediction-outcome selection.

It changed no sampling, identities, targets, bits, onset, persistence,
Phase-5E plan, or Phase-5P gate.

The outer execution estate remains exactly:

- 1 executed fault shard;
- 1 executed clean cache.

Full-fleet execution has not started.

No aggregate CC result exists.

No CSC result exists.

## Next boundary

Before any of the 731 remaining fault shards execute, a final execution
activation must freeze and bind:

- the Phase-5P gate SHA;
- the exact Phase-5R executor SHA;
- the Phase-5R qualification result SHA;
- the Phase-5R qualification manifest SHA;
- the unchanged Phase-5E plan SHA.

That activation must not itself execute another outer shard.
