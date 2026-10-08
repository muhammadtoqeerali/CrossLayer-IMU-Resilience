# Phase 6J — CSC Execution Runtime Binding v1

## Status

`FROZEN_PRE_AUTHORIZATION_CSC_EXECUTION_RUNTIME_BINDING`

Phase 6J freezes the separately implemented CSC execution runtime and its
synthetic/pre-authorization qualification suite.

This freeze does not authorize real CSC outer execution and makes no scientific
change.

## Frozen runtime

- `experiments/phase_06/csc_execution_runtime_v1.py`
- SHA256:
  `e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438`

Qualification test:

- `tests/test_phase6j_csc_execution_runtime_pre_authorization_v1.py`
- SHA256:
  `27828bb9681e1436c5e49fe1b7f13bf27a9e96c5d598e437c1bfc1d575d0de59`

## Execution boundary

The public production entrypoint is `execute_shard(...)`.

It remains prospectively blocked by:

`EXECUTION_AUTHORIZED = False`

and the hard-stop status:

`PHASE6J_OUTER_EXECUTION_NOT_AUTHORIZED`

Even a syntactically plausible future gate cannot currently reach production
outer execution. Heavy execution modules are imported only below the
authorization check, and the authorized production body remains explicitly
unreleased as:

`PHASE6J_AUTHORIZED_EXECUTION_BODY_NOT_YET_RELEASED`

Synthetic injected hooks are the only execution-style qualification surface in
this phase.

## Frozen runtime contracts

The runtime now binds the Phase-6H architecture without changing its scientific
surface.

It implements:

- deterministic subject × sensor-family shard identity;
- canonical FP32 then PTQ-v7 model-member ordering;
- checkpoint-seed order 42, 123, 2025;
- canonical pair ordering and ascending compute-window order;
- exact Phase-5 clean-cache resolution by
  fold × subject × model variant × checkpoint seed;
- `ABORT_SHARD_NO_PHASE6_RECOMPUTE` for missing, ambiguous, or invalid clean
  cache state;
- zero Phase-6 C0 recomputation;
- direct Phase-6I source-trial exposure-helper binding;
- stored-window sensor conditioning;
- source-trial sensor faulting before historical Phase-4H-v2 rewindowing;
- sensor-reference generation before compute-fault execution;
- exact Phase-5 `_window_tensor`, `execute_fault_sequence`, and
  `execution_active_mask` bindings;
- per-compute-window reference switching between Phase-6 sensor reference and
  frozen Phase-5 clean reference;
- retained zero-overlap pair behavior;
- one model bundle per shard × model variant × checkpoint-seed stream;
- Phase-5-owned PTQ persistent-weight restoration in the already-pinned
  `execute_fault_sequence` `finally` block;
- Phase-6-local atomic output commit;
- success-marker creation before atomic directory replacement;
- no reuse of the Phase-5 `commit_atomic_artifact` helper;
- whole-shard resume/recompute semantics with no row-level resume.

## Output contract

A completed runtime shard uses:

- `pair_members.jsonl`
- `sensor_reference.jsonl`
- `csc_fault.jsonl`
- `metadata.json`
- `_SUCCESS.json`

The shard is the atomic and concurrency unit.

## Qualification entering this freeze

Immediately before creation of the freeze artifacts:

- dedicated Phase-6J synthetic/pre-authorization tests: 22 passed;
- full Phase-6 regression set: 148 passed;
- full repository fleet: 998 passed;
- historical Phase-5R lifecycle test: 1 deselected;
- strict Phase-6H/runtime semantic audit: PASS;
- pinned Phase-5 PTQ reset-delegation audit: PASS.

The runtime and test hashes remained byte-stable throughout this qualification.

## Scientific and governance boundary

During Phase 6J qualification:

- real CSC execution: FALSE;
- outer arrays read: FALSE;
- model loaded: FALSE;
- real sensor fault execution: FALSE;
- real compute fault execution: FALSE;
- model forward: FALSE;
- performance metrics generated: FALSE;
- validation outcomes used for tuning: FALSE;
- outer-test outcomes used for tuning: FALSE;
- OnField outcomes used for tuning: FALSE;
- threshold retuning: FALSE;
- checkpoint reselection: FALSE;
- resampling/replacement/relocation/rebalancing: FALSE.

## Next boundary

Phase 6J is not an execution authorization.

Before any real CSC outer model forward, a separate prospective gate must bind:

1. this exact runtime SHA;
2. this exact qualification-test SHA;
3. the frozen Phase-6H and Phase-6I artifacts;
4. all other pinned upstream hashes;
5. all 732 runtime shard identities;
6. exact per-shard pair-member counts;
7. exact per-shard sensor-reference counts;
8. exact per-shard CSC-fault-window counts;
9. a successful validation of the complete frozen Phase-5 clean-cache estate;
10. the output schema and atomic/resume contract;
11. an explicit guarantee that no held-out performance outcome informed the
    authorization decision.

Only a later prospective phase may release an authorized production execution
body.
