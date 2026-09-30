# Phase 3J Primary 300-ms Five-Fold Freeze

## Status

FIVEFOLD MEMBERSHIP FROZEN

## Dataset

- Root: `/mnt/hdd16T/protechto/data/UniVrFall_KFall_NoOF/segments/300ms_50ov_npseg_filt_binary`
- Window: 300 ms
- Sampling: 100 Hz
- Samples/window: 30
- Overlap: 50 percent
- Stride: 150 ms
- UniVRFall subjects: 29
- KFall subjects: 32
- Total primary subjects: 61

## Fold algorithm

- Subject order: lexicographic storage directory names
- KFold: 5
- Shuffle: true
- KFold random state: 42
- Validation source: outer non-test indices only
- Validation fraction: 0.2
- Validation random state: 42

## Fold sizes

- Fold 1: train=38, validation=10, outer-test=13
- Fold 2: train=39, validation=10, outer-test=12
- Fold 3: train=39, validation=10, outer-test=12
- Fold 4: train=39, validation=10, outer-test=12
- Fold 5: train=39, validation=10, outer-test=12

## Leakage guarantees

- Train, validation and outer-test subjects are disjoint in every fold.
- Every one of the 61 subjects appears in outer test exactly once.
- OnField is absent from all five primary folds.
- OnField 999 and 1000 are unusable everywhere.

## Partition roles

Training is the only permitted source for later INT8 calibration.

Validation may be used for model/protection parameter selection before outer-test evaluation.

Outer test is evaluation-only and cannot influence model, fault, quantization or protection decisions.

Retained OnField 1001-1010 remain an untouched Activity-only external evaluation cohort.

## Remaining Phase-3 work

The next task is event-time lineage. UniVRFall and KFall fall-onset/impact annotations must be mapped to the exact processed 300-ms trial identities.

Only then will causal window timestamp and pre-impact lead-time semantics be frozen.
