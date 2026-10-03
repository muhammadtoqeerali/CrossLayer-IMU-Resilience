# Phase 4I OnField External 300-ms Root Binding v1

Status: **QUALIFIED_CANONICAL_ROOT_BINDING_PRE_EXTERNAL_EVALUATION**

The Phase-4 external Activity-only evaluation is bound to:

`/mnt/hdd16T/protechto/data/OnField/segments/300ms_50ov_npseg_filt_binary`

This is the canonical Phase-3 audited root and contains the governed retained
storage IDs 1001 through 1010 exactly.

The backup/workstation tree contains 1109 and 1110 instead of 1009 and 1010.
A read-only byte-level comparison established:

- workstation 1109 == canonical 1009
- workstation 1110 == canonical 1010

The comparison includes relative file paths, file sizes, and SHA-256 hashes
of both `segments.npy` and `labels.npy`. No renaming or data mutation is
performed. Phase 4I uses the canonical root directly.

The retained external cohort is fixed at 10 subjects, 16 trials, 1,023,337
Activity windows, and zero Falling windows. Storage IDs 999 and 1000 remain
permanently excluded.

OnField cannot be used for training, validation, calibration, threshold
selection, fault-parameter selection, model selection, or operating-point
selection.
