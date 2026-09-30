# Phase 3E Checkpoint and Falling-Simulator Lineage

## Status

PASS

## Baseline policy

400 ms remains the primary frozen CrossLayer workload.

300 ms is retained as a temporal-resolution sensitivity comparator.

## Checkpoint discovery

Fold checkpoint files: 1527

Complete 300-ms FP32 five-fold runs: 266

Complete 400-ms FP32 five-fold runs: 3

## Falling Simulator


## KFold subject enumeration

`SORTED_LEXICOGRAPHIC_DIRECTORY_NAMES`

Important:

Historical fold reconstruction is deterministic because subject directory names are explicitly sorted before KFold.

Exact reconstruction still requires the correct dataset root, augmentation-exclusion list and KFold/validation parameters.

## Next decision

Bind complete checkpoint families to their dataset roots and fold memberships before selecting any fold family for CrossLayer confirmation.

The recent Falling Simulator is preferred as the dataset-lineage candidate if its UniVR/KFall trial keys reconcile exactly with the protected historical trial space.
