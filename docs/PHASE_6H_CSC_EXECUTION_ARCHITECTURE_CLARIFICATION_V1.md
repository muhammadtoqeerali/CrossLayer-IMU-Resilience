# Phase 6H — CSC Execution Architecture Clarification v1

## Status

`FROZEN_PRE_IMPLEMENTATION_EXECUTION_ARCHITECTURE_CLARIFICATION`

This clarification resolves implementation architecture that was not
machine-defined by Phase 6E–6G. It makes no change to the frozen scientific
surface and does not authorize execution.

## Runtime work unit

The execution work unit is one **subject × sensor-family** shard. There are
exactly 732 such shards. Each shard contains all three severities, all retained
selected parents for that subject/family, all assigned compute strata, and all
eligible model-variant/checkpoint-seed members.

The shard is also the unit of concurrency and atomic persistence.

## Reference ownership

Phase 6 performs **zero new C0 clean forwards**. Clean references come from the
already-frozen Phase-5 clean-cache estate. A matching Phase-5 clean cache must
be uniquely resolved and hash-valid for the same subject, fold, model variant,
and checkpoint seed. Missing or invalid clean caches abort the Phase-6 shard;
Phase 6 does not regenerate them.

Sensor-only references are owned by Phase 6 because Phase-4H did not persist a
raw per-window reference estate suitable for CSC joining. One logical sensor
reference cache is defined per selected sensor instance × model member.

Only the exact frozen sensor-exposed retained-window indices receive
sensor-reference rows.

For every compute-faulted window:

- if the sensor fault is active at that retained window, the no-compute-fault
  reference is the Phase-6 sensor-reference output;
- otherwise the reference is the frozen Phase-5 clean output for the same
  retained window and model member.

## Composition ordering

Sensor corruption is applied before compute fault injection.

For source-trial families, the source is corrupted with the frozen Phase-4H
operator and then rewindowed with the frozen Phase-4H v2 route.

For stored-window families, only the frozen selected stored parent window is
sensor-corrupted.

Every compute-active window is then converted using the frozen Phase-5 input
conversion and passed through the frozen Phase-5 compute-fault route. This is
true even when the sensor is not exposed at that particular compute-active
window.

## Model lifecycle

Within one shard, processing order is:

1. model variant: FP32, then PTQ-v7;
2. checkpoint seed: 42, 123, 2025;
3. canonical selected-pair order;
4. compute-active window index ascending.

One model bundle is loaded per shard × variant × seed stream, used
sequentially, then released. No mutable model state is shared concurrently.

PTQ persistent-weight execution retains the frozen clean-state restoration in
`finally`. Any exception aborts the whole shard and a retry reloads the frozen
model bundle.

## Atomic persistence and resume

Each subject-family shard produces one artifact directory:

`{output_root}/shards/{shard_id}`

Execution occurs in:

`{output_root}/shards/{shard_id}.partial`

The shard writes:

- `pair_members.jsonl`
- `sensor_reference.jsonl`
- `csc_fault.jsonl`
- `metadata.json`
- `_SUCCESS.json`

All JSONL is canonical UTF-8 JSON, one object per line, uncompressed.

`_SUCCESS.json` is created inside the temporary directory only after every
declared file is complete and hashed. The temporary directory is then
atomically renamed/replaced as the final artifact.

There is no row-level resume. Invalid final or partial artifacts abort unless
`--recompute-partial` is explicitly supplied, in which case the entire shard
is recomputed.

## Output roles

`pair_members.jsonl` has exactly one row per executed pair member and binds the
frozen sensor instance, compute coordinate, model member, reference caches,
window sets, overlap, and record cardinalities.

`sensor_reference.jsonl` contains raw model outputs for exact sensor-exposed
retained windows before compute fault injection.

`csc_fault.jsonl` contains every compute-faulted window output plus mutation
audit, sensor-activity status, and whether its appropriate no-compute-fault
reference is the Phase-6 sensor reference or the Phase-5 clean cache.

The global frozen cardinalities are:

- pair-member records: 21,793,038
- sensor-reference records: 35,167,107
- CSC fault-window records: 411,540,372
- zero-overlap pair-member records: 5,234,355

Per-shard cardinalities are not invented here. They must be derived
metadata-only and frozen in the later pre-execution authorization manifest.

## Zero-overlap persistent pairs

Zero-overlap pairs remain fully retained.

Their pair-member row records zero temporal overlap and an empty simultaneous
overlap list. Their unchanged persistent compute suffix still executes.
Every CSC fault row has sensor-active=false and uses the matching Phase-5 clean
reference. Their sensor-exposed reference rows are still emitted for the
sensor-exposed windows outside the compute suffix.

No filtering, resampling, relocation, replacement, or rebalancing is allowed.

## Remaining execution gate

This clarification does not authorize a model forward.

The execution-capable runtime must be a new file. It must not modify the frozen
Phase-6E metadata executor or the frozen Phase-6G adapter.

Before any CSC outer forward, a later gate must freeze:

- the execution-runtime SHA;
- its qualification-test SHA;
- all 732 shard identities;
- exact per-shard pair-member/reference/fault-window counts;
- validation of the frozen Phase-5 clean-cache estate;
- all upstream hashes.

No outer performance outcome may inform that gate.
