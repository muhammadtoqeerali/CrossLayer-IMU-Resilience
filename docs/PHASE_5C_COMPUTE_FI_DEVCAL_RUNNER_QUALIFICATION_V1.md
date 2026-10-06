# Phase 5C — Real Training-Calibration Paired Runner Qualification v1

**Status:** QUALIFIED_REAL_TRAINING_CALIBRATION_PAIRED_RUNNER  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5C qualifies the compute-fault execution harness on real model inputs
without touching validation, outer-test, or OnField evidence.

The 105-case qualification protocol was frozen before real-window execution.

## Input partition

Only the frozen training-only PTQ calibration identity configuration is used.

Each fold contributes its first lexicographically ordered frozen calibration
identity. The same fold identity is reused across seeds 42, 123, and 2025.

The calibration manifest's classification label is retained as its native
textual metadata (`Activity` or `Falling`). It is not used for qualification
selection or any task-performance gate.

## Model estate

All 15 prospective 300-ms FP32 checkpoints are retained:

- seeds 42, 123, 2025;
- folds 1 through 5.

All corresponding mixed-precision PTQ-v7 artifacts are retained.

No best, worst, representative, median, or deployment member is selected.

## Frozen qualification matrix

The matrix contains exactly 105 paired cases:

- 30 FP32-variant paired cases;
- 75 PTQ-v7 paired cases.

FP32 variant families:

- FP32 activation single-bit flip;
- FP32 buffer single-bit flip.

PTQ-v7 families:

- qint8 persistent-weight single-bit flip;
- quint8 activation single-bit flip;
- quint8 buffer single-bit flip;
- FP32 activation single-bit flip;
- FP32 buffer single-bit flip.

Every qualification case uses:

- element index 0;
- bit position 0;
- inference index 0;
- transient-one-inference persistence;
- multiplicity 1;
- replicate index 0.

These are qualification smoke coordinates only. They are not the prospective
outer compute-FI sampling plan.

## Representation binding

Persistent PTQ weight faults use `torch.qint8` / `torch.int8`.

Quantized activation and buffer faults use
`torch.quint8` / `torch.uint8`.

Floating activation and buffer faults use exact IEEE-754 binary32 XOR
mutation.

## Qualification gates

All 15 reconstructed eager PTQ models must match their corresponding frozen
TorchScript artifacts bit-for-bit on their selected real calibration window.

All 105 fault identities must be unique.

Every active fault must alter exactly the selected payload element and bit.

Source FP32 checkpoints, PTQ state dictionaries, and PTQ TorchScript artifacts
must remain hash-identical.

## Serialization repair

The first execution attempt completed model execution but stopped while
serializing the selected calibration identity because the manifest label
`Activity` was incorrectly cast to `int`.

The repair changed only that result-metadata serialization from integer
conversion to preservation of the manifest string.

The frozen configuration, 105-case matrix, partitions, model estate, fault
coordinates, fault families, targets, and acceptance gates did not change.

The exact same frozen 105-case qualification was then rerun.

## Scientific boundary

Real faulted forwards are restricted to training-calibration inputs.

No raw logits or probabilities are persisted.

No accuracy, recall, specificity, threshold, timing, or other task-performance
metric participates in qualification or protocol selection.

Validation, outer-test, and OnField partitions are prohibited.

No CC outer-test result or CSC result is generated.

No physical or MCU fault equivalence is claimed.

The prospective outer compute-FI sampling/cardinality plan remains unfrozen.

The next governed step is to freeze that outer compute-FI protocol before any
CC outer-test execution.
