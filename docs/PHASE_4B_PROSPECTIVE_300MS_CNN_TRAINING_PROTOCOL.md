# Phase 4B — Prospective 300-ms CNN Training Protocol

## Purpose

Freeze the prospective FP32 CNN baseline before any model training,
prediction, INT8 calibration, or fault injection.

## Primary dataset

- 300-ms windows
- 50% overlap
- 100 Hz sampling
- 30 samples per window
- 15-sample / 150-ms stride
- 61 primary subjects
- 29 UniVR subjects
- 32 KFall subjects

## OnField boundary

The retained OnField cohort 1001-1010 is external validation only.

OnField subjects are not members of the primary five-fold training,
validation, or outer-test population.

IDs 999 and 1000 are permanently excluded.

## Model

CNN.

No historical checkpoint is reused as the primary prospective model.

## Fold authority

The frozen Phase-3 five-fold manifest is authoritative.

Folds must not be regenerated during training.

## Training boundary

The outer test subjects are unavailable for:

- checkpoint selection
- early stopping
- hyperparameter selection
- threshold selection
- calibration
- architecture selection

OnField data are also unavailable for all of the above.

## Historical checkpoints

Historical checkpoint performance must not be used to choose the
prospective baseline.

## Quantization

INT8 calibration is deferred until the FP32 baseline is frozen.

Calibration data must come from the permitted training-side population.

## Fault injection

No fault injection occurs before the FP32 baseline is frozen.

## 400-ms sensitivity

400-ms / 50%-overlap evaluation remains a separate sensitivity condition.
It does not replace the 300-ms primary protocol.

## Execution status

This document freezes the protocol candidate only.

No model training occurred during Phase 4B protocol construction.
