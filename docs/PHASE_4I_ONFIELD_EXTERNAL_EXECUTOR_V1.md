# Phase 4I OnField External Executor v1

Status: **QUALIFIED_PRE_ONFIELD_INFERENCE**

The executor is a thin execution layer for the already-frozen Phase-4I
Activity-only external-evaluation protocol.

Before qualification it performed no OnField model forward pass and read no
OnField performance result.

## Reused frozen semantics

The executor reuses:

- the qualified Phase-4H FP32/PTQ model-loading path;
- the same 15 FP32 seed×fold checkpoints;
- the corresponding 15 qualified PTQ v7 TorchScript members;
- the frozen 45-row validation-selected threshold matrix;
- the historical `trigger_episodes` rule.

Probabilities from one checkpoint/trial forward pass are thresholded three
ways for the balanced, low-false-alarm, and timely-150ms operating points.

## External metric construction

For every trial, checkpoint, model variant, and operating point the executor
records:

- Activity windows;
- true-negative windows;
- false-positive windows;
- Activity specificity;
- false-trigger episodes;
- Activity duration;
- false triggers per Activity hour.

Trigger state resets at every trial boundary.

For subjects with multiple trials, Activity specificity is computed from
summed TN/FP denominators, and false-trigger rate is computed from summed
episodes and summed Activity duration. Trials are therefore not assigned
arbitrary equal inferential weight.

## Model-estate aggregation

Every checkpoint-specific result remains available.

Each subject summary uses:

1. equal mean over folds 1–5 within each seed;
2. equal mean over seeds 42, 123, and 2025.

No checkpoint is selected or performance-weighted.

## Uncertainty

Only the ten external subjects are bootstrap units.

The cohort report uses deterministic 10,000-replicate subject bootstrap
intervals after the full 15-checkpoint subject aggregation.

Overlapping windows and trials are not treated as independent inferential
samples.

## Output boundary

The executor never stores raw probability streams.

It does not generate fall recall, event recall, lead time, recovery, balanced
accuracy, an FP32/PTQ equivalence test, or a binary global robustness label.
