# Phase 4H — Instance Sampling v2 Qualification

**Status:** QUALIFIED_METADATA_REPLAY
**Evidence tier:** P0
**Physical-realism claim:** false

## Result

The v2 fault-instance sampling/replay protocol passed every predeclared
metadata-only sanity gate in all five folds.

- folds: 5
- passes: 5
- failures: 0

No sensor values were mutated and no model was loaded.

## V1 lineage

V1 remains preserved as a failed pre-result protocol.

Its only failing gate was cross-fold human-readable `fault_id`
uniqueness for folds 2, 3, and 4.

The v1 canonical `replay_id` was already collision-free.

## V2 causal change

V2 made one change only:

`fault_id` now includes partition and fold.

No severity, targeting, onset, duration, persistence, replication,
seed namespace, seed derivation, partition rule, or replay
canonicalization changed.

## Qualified properties

V2 qualification supports the following metadata claims:

- deterministic replay generation;
- exact 153 stored-window instances per parent;
- exact 138 sequence instances per parent;
- unique replay IDs;
- unique fold/partition-aware fault IDs;
- valid temporal bounds;
- exact stochastic replicate structure;
- early/middle/late onset-stratum coverage for dropout and frame loss;
- model-independent causal instance identity;
- rejection of OnField fault generation.

## Not yet a robustness result

This qualification does not establish model robustness, outer-test
robustness, physical sensor realism, HIL behavior, MCU behavior, or
OnField robustness.

## Remaining pre-execution freezes

Before any sensor-fault/model execution:

1. freeze evaluation aggregation and uncertainty rules;
2. freeze robustness acceptance/reporting criteria.

After those are frozen, the Phase-4H FI engine may execute without
further protocol tuning.
