# Phase 4H — Instance Sampling v2: fault_id Namespace Repair

**Status:** FROZEN_PRE_SANITY_RESULT

## Parent v1 result

Instance-sampling v1 failed its metadata-only sanity qualification.

The failure was isolated to one gate: human-readable `fault_id`
collisions across folds when the same source-trial parent appeared in
multiple fold-specific training-calibration sets.

The canonical `replay_id` was already collision-free.

All other v1 gates passed.

## Single v2 change

V2 changes only the `fault_id` namespace.

V1:

`p4h-{family}-{level}-{variant}-r{replicate}-{parent_hash}`

V2:

`p4h-{partition}-f{fold}-{family}-{level}-{variant}-r{replicate}-{parent_hash}`

No model identity is included.

## Unchanged from v1

V2 does not change:

- fault families;
- severity values;
- target channels;
- onset rules;
- duration rules;
- persistence rules;
- stochastic replicate counts;
- seed namespace;
- seed derivation;
- source partitions;
- replay canonicalization;
- OnField prohibition;
- same-instance causal replay across backbones.

Because `fault_id` is part of the canonical instance payload, v2
replay hashes will differ from v1. That is an identifier consequence of
the namespace repair, not a change in sampled fault location/severity.

No sensor values or model predictions are used in this repair.
