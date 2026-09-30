# Phase 3I OnField External-Validation Audit

## Status

PASS

## Final prospective OnField cohort

The prospective CrossLayer study retains exactly ten OnField storage
identities:

- 1001
- 1002
- 1003
- 1004
- 1005
- 1006
- 1007
- 1008
- 1009
- 1010

These ten retained cases contain Activity data only.

They contain no Falling windows.

At 300 ms and 50 percent overlap the retained cohort contains:

- 10 retained cases
- 16 recordings/trials
- 1,023,337 Activity windows
- 0 Falling windows

## Rejected historical cases

Storage IDs 999 and 1000 are permanently excluded from the prospective
CrossLayer study because of known data-quality issues.

They are retained in documentation only as historical provenance.

They must not be used for:

- training
- augmentation
- validation
- testing
- INT8 calibration
- fault-severity selection
- monitor tuning
- supervisor tuning
- recovery tuning
- any later CrossLayer experiment

## Primary benchmark

The primary subject-independent benchmark remains:

- UniVRFall: 29 subjects
- KFall: 32 subjects
- total: 61 subjects

OnField is not mixed into these five folds.

## OnField scientific role

The ten retained OnField cases provide an independent Activity-only
field-domain evaluation cohort.

This cohort can measure:

- clean false alarms
- field-domain generalization on Activity
- fault-induced false alarms
- protection false triggers
- Activity-side graceful degradation

This cohort cannot measure:

- fall sensitivity
- fall recall
- missed-fall rate
- fall-detection lead time
- fall recovery success

Those fall-positive outcomes must be evaluated using the annotated UniVRFall
and KFall fall trials.

## Overall retained study identities

The study therefore contains 71 retained dataset-qualified subject identities:

- 61 primary UniVRFall + KFall subjects
- 10 independent OnField Activity subjects
