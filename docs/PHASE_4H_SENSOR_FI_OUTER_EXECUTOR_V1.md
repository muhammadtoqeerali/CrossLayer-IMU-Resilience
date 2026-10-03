# Phase 4H — Outer Shard Executor v1

**Status:** FROZEN_PRE_OUTER_EXECUTION_EXECUTOR

This executor is bound to the qualified outer execution v2 historical
truth contract and its 793 immutable subject/block shards.

The implementation was frozen before any outer model load, outer model
forward, or outer sensor-fault operation.

## Modes

`dry-run` is metadata/interface only. It validates hashes, artifact
presence, threshold matrix, subject/trial/window inventories, source
paths, historical truth counts, sampling condition-slot schemas, and
shard arithmetic. It does not load weights, call model forward, call
`apply_fault`, or compute outer performance.

`execute-shard` performs one immutable subject/block shard.

## Raw-input boundary

The executor feeds historical raw 30x9 windows to the frozen model
callable. No external normalization is applied. The CNN/PTQ model owns
the IMU normalization internally. Fault injection therefore remains at
the frozen pre-normalizer raw-unit layer.

## Same-fault causal comparison

For each CS condition slot, parent-specific sampling-v3 metadata is
created once. The resulting exact fault instances are then reused across
all three checkpoint seeds and both FP32/PTQ model variants.

Operating points reuse the same model probabilities and therefore do not
multiply model forward count.

## Sequence families

Sequence faults are applied to the source-faithful oriented trial.
Rewindowing uses the qualified Phase-3 route:

- Activity: `15 * activity_local_index`
- Falling: `fall_start_frame + 15 * falling_local_index`

Physical lead-time uses the separate zero-based
`fall_start_position`/`impact_position` contract.

## Historical exception

`KFALL_106_T27_R05` remains all-Activity in retained labels but is a
risk-linked true Falling event. No relabeling occurs. The historical
valid-trigger rule makes a valid Falling trigger impossible for this
trial, so it is a frozen missed/FN event with lead time NA.

## Compact output

Each shard writes:

- `subject_condition_metrics.jsonl`
- `coverage.json`
- `_SUCCESS.json` last

Raw per-window probabilities are not persisted.

Each CS condition row includes a SHA-256 digest over all parent-specific
`fault_id|replay_id` identities for that subject condition slot.

## Resume

Final shard directories are immutable. A completed shard is reusable
only when `_SUCCESS.json` and all output/config/implementation/plan
hashes match. Partial shards are never aggregated.
