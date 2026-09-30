# Phase 3C Repaired Dataset and Five-Fold Audit

## Status

PASS

## Official UniVRFall specification

- 29 laboratory participants
- 10 construction workers
- 39 participants total
- 100-Hz IMU
- 21 fall types
- 573 fall events
- video-synchronized fall-onset frame
- video-synchronized fall-impact frame
- subject-independent 5-fold cross-validation documented

## Official KFall specification

- 32 participants
- 5,075 motion files
- 2,729 ADL motions
- 2,346 fall motions
- 21 ADL types
- 15 fall types
- 100-Hz IMU
- fall-onset and fall-impact annotations

## Historical checkpoint split

The recovered 47/6/14 split remains the provenance split for the
single frozen historical checkpoint.

It is not replaced post hoc by the five-fold protocol.

## Five-fold protocol

Candidate fold-related files found: 6963

Explicit train/validation/test fold records detected: 36

Five-fold evaluation requires fold-specific model training.

A single pretrained historical checkpoint cannot be reused as five
independent cross-validation models.

## Timing correction

Physical pre-impact timing is no longer treated as globally unavailable.

Both UniVRFall and KFall provide onset/impact frame annotations.

It becomes publishable physical lead-time evidence only after those
annotations are mapped unambiguously to the protected historical trials
and the exact window timing semantics are verified.
