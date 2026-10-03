# Phase 4H — Development/Calibration Execution Runner v1

**Status:** FROZEN_PRE_QUALIFICATION_RESULT

The runner integrates the already-qualified Phase-4H sensor-fault
protocol with the frozen 300 ms FP32 and PTQ-v7 model estates.

## Qualification scope

Runner qualification is restricted to the frozen
`training_calibration` partition.

Validation, outer test, and OnField are rejected in qualification mode.

The qualification does execute sensor faults and model forward passes,
but it does **not** use the resulting probabilities as a scientific
performance signal.

No accuracy, recall, specificity, F1, AUROC, AUPRC, robustness delta, or
model ranking is an acceptance gate.

## Model pairing

For each seed/fold pair:

- the frozen FP32 checkpoint is hash-verified and strictly reconstructed;
- the paired qualified PTQ-v7 TorchScript artifact is hash-verified;
- the exact same clean or corrupted 30×9 tensor is supplied to both.

Fault sampling remains model-independent.

## Stored-window families

Bias, scale factor, noise, clipping, and axis loss operate directly on
the frozen stored filtered 30×9 calibration parent.

Fold-local q99 and robust-sigma vectors come only from the frozen
training-calibration severity audit.

## Sequence families

Drift, stuck channel, dropout, frame loss, jitter, delay, and orientation
operate on the source-faithful oriented trial.

The faulted sequence is then converted back into model windows using the
historical 5 Hz per-window filter, 30-sample window, and 15-sample
stride.

For qualification, one deterministic activity-only calibration-
represented trial per fold is used. Its clean source reconstruction must
match the stored filtered windows exactly before any sequence-fault
model forward is accepted.

## Thresholds

The 45 validation-selected threshold rows are loaded only to validate
the frozen artifact schema.

They are not applied, modified, or reselected during runner
qualification.

## Outer-test boundary

Runner v1 does not expose outer-test execution from its CLI.

After qualification, a separate final outer-test execution manifest
must freeze the exact full execution matrix and sharding plan before
held-out robustness is run.
