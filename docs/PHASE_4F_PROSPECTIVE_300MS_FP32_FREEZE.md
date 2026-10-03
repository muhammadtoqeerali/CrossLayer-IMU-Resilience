# Phase 4F — Prospective 300-ms FP32 Baseline Freeze

**Status:** FROZEN
**Date:** 2026-10-02
**Provenance tier:** P0 clean baseline

## Frozen primary workload

The prospective primary workload is the fresh CNN trained on the clean
UniVR + KFall primary population using:

- 300-ms windows
- 100 Hz sampling
- 30 samples per window
- 50% overlap
- 15-sample / 150-ms stride
- no augmentation
- seeds 42, 123, and 2025
- five frozen folds
- 15 retained seed/fold checkpoints

The 400-ms configurations remain historical/reference/sensitivity evidence
only. The 95%-overlap lineage remains excluded from the primary protocol.

## Partition and selection audit

Formal closure verified that:

- all 15 seed/fold checkpoints exist;
- fold membership is invariant across seeds;
- train, validation, and outer-test subjects are disjoint within every fold;
- the five Phase-4E fold memberships exactly match the frozen Phase-3
  manifest;
- all 61 primary subjects appear in outer test exactly once;
- all 45 operating thresholds come from validation selection;
- validation and outer-test predictions reuse those validation-selected
  threshold/consecutive-trigger settings;
- outer-test evidence is evaluation-only;
- no best seed/fold is selected from outer-test performance;
- the complete 3-seed x 5-fold estate is retained.

## OnField and exclusions

OnField is not part of this prospective training/tuning baseline.
Rejected cases 999 and 1000 are absent.

Valid OnField cases remain reserved for later independent Activity-only
external evaluation and do not participate in model training, threshold
selection, INT8 calibration, or fault/protection tuning.

## Source provenance

Implementation workspace:

`/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori`

Git commit:

`1453e2644d2b27a0356efa4523a97ce95b3c0b5f`

Dirty workspace at freeze:

`True` (38 status entries)

Because the implementation workspace is dirty, the Git commit is not used
as the sole source identity. The authoritative freeze manifest records:

- exact hashes for critical source files;
- a hash inventory of relevant tracked and untracked source/config files;
- aggregate source-inventory SHA-256;
- exact checkpoint SHA-256 identities;
- hashes for all primary clean evidence artifacts;
- frozen fold/risk-manifest identities;
- software environment versions.

Critical source hashes:

- `models/CNN.py`: `5e1ddac378c80929068b6fb8f62fff0d351bacbf047297829bd0531ad9c43eb2`
- `train.py`: `da4e46b583ebaa5374528d7f74426e68bc5ded3459b71d7516fb59c9648363ec`

Source inventory SHA-256:

`31b3f5fb9ee8fbc4a892eb15b888910df0aa9301d529893371fc539fb4db3097`

## Checkpoint policy

No single deployment checkpoint is selected in Phase 4F.

The authoritative baseline is the full 15-checkpoint,
3-seed x 5-fold prospective estate. Any future need for a single
deployment checkpoint must use a deterministic rule based only on allowed
validation/calibration information and must be frozen before fault-study
outcomes are examined.

## Next subphase

Phase 4G performs static PTQ / INT8 realization.

Quantization calibration must use only the already frozen Phase-3
training-partition calibration identities. Validation, outer test,
OnField, and fault-injected samples may not determine quantization
parameters.

No resilience hypothesis is confirmed by Phase 4F.
