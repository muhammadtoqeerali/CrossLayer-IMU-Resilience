# Phase 3B Historical Split Recovery

## Status

PASS

## Why a new random split is inappropriate

The primary protected model is already trained.

A newly randomized subject split could place subjects seen during historical
model training into a nominal new test partition.

That would compromise the interpretation of held-out performance.

Phase 3 therefore recovers the historical subject partition before freezing
the CrossLayer data protocol.

## Recovered source

Path:

`/mnt/hdd16T/ToqeerHomeBackup/toqeer/IMU_Reliability/data/manifests/date2025_cnn400_inferred_split_v1.json`

SHA-256:

`3ed3e1c8a70e581d7b37416f1fb19a08efa8787eae55f524152bb264e8db2305`

Candidate copies found:

1

All recovered copies agree:

True

## Verified historical protected tree

Path:

`/mnt/hdd16T/protechto/data/back/UniVrFall_KFall/segments/400ms_50ov_npseg_filt_binary`

Trials:

6,193

Windows:

1,187,326

Subjects including augmentation-only:

69

## Partition reconstruction

| Historical partition | CrossLayer candidate role | Trials | Windows |
|---|---|---:|---:|
| train | development | 4,628 | 510,479 |
| validation | calibration | 447 | 89,868 |
| test | held-out confirmation | 1,116 | 366,507 |
| subjects 999/1000 | augmentation-only | 2 | 220,472 |

Unassigned trials:

0

## Leakage audit

Train/validation subject overlap:

`[]`

Train/test subject overlap:

`[]`

Validation/test subject overlap:

`[]`

Augmentation-only subjects inside the protected split:

`[]`

## Freeze state

The recovered partition is now independently verified.

It is still marked as a Phase-3 candidate until label, event, timing and raw
lineage semantics are audited.

No new random split has been generated.
