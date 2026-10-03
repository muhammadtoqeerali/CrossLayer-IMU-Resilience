# Phase 4H — Sensor FI Severity Protocol v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0 software fault injection
**Physical-realism claim:** false

## Purpose

This protocol freezes a three-level software-stress ladder before
model robustness is measured.

The levels are named L1, L2, and L3 only to indicate increasing
software corruption. They are not estimates of real fault frequency,
hardware failure probability, or physical sensor realism.

## Leakage control

Any signal-distribution reference is materialized independently within
each outer fold from that fold's frozen 4096-window training-only
calibration identities.

No cross-fold pooled data-derived magnitude is used. This matters
because every subject is an outer-test subject in one fold.

No model predictions, validation outcomes, outer-test outcomes, or
OnField outcomes entered severity selection.

## Frozen levels

### Bias

Signed fractions of fold/channel q99 absolute signal:

- L1: 0.05
- L2: 0.10
- L3: 0.20

### Drift

Linear endpoint excursion over a nominal 100-sample / 1-second fault:

- L1: 0.05 × fold/channel q99
- L2: 0.10 × fold/channel q99
- L3: 0.20 × fold/channel q99

Both signs are retained.

### Scale factor

Absolute gain delta around one:

- L1: 0.05
- L2: 0.10
- L3: 0.20

Both `1-delta` and `1+delta` variants are retained.

### Additive noise

Gaussian sigma relative to fold/channel robust sigma
`MAD × 1.4826`:

- L1: 0.50
- L2: 1.00
- L3: 2.00

### Clipping/saturation

Symmetric threshold relative to fold/channel q99 absolute signal:

- L1: 1.00 × q99
- L2: 0.75 × q99
- L3: 0.50 × q99

Lower thresholds represent higher software stress.

### Stuck channel

Contiguous hold duration:

- L1: 5 samples / 50 ms
- L2: 15 samples / 150 ms
- L3: 30 samples / 300 ms

### Axis loss

Zeroed effective-channel count:

- L1: one channel;
- L2: one complete sensor triad;
- L3: all six effective accelerometer/gyroscope channels.

### Dropout

Contiguous zero-fill duration:

- L1: 5 samples / 50 ms
- L2: 15 samples / 150 ms
- L3: 30 samples / 300 ms

### Frame loss

Complete lost-frame count:

- L1: 1 frame / 10 ms
- L2: 3 frames / 30 ms
- L3: 5 frames / 50 ms

Lost complete frames are reconstructed causally by forwarding the last
available complete frame before normal preprocessing continues.

### Jitter

Maximum absolute timestamp displacement:

- L1: 1 ms
- L2: 2 ms
- L3: 4 ms

The maximum remains below half the nominal 10 ms sample period.

### Delay

Causal sample delay:

- L1: 1 sample / 10 ms
- L2: 5 samples / 50 ms
- L3: 10 samples / 100 ms

### Orientation

Software rotation of both accelerometer and gyroscope triads by the
same proper 3D rotation:

- L1: 5 degrees
- L2: 15 degrees
- L3: 30 degrees

Rotations about x, y, and z remain separate variants. Euler channels
are not used to implement this fault.

## Predeclared sanity gates

Before any model robustness measurement, this frozen ladder must pass
input-space-only sanity checks:

- all fold-local reference scales are finite and positive;
- materialized value magnitudes are finite;
- L1/L2/L3 ordering is monotonic;
- scale-factor gains remain positive;
- clipping thresholds decrease monotonically;
- all temporal levels fit the shortest parent trial represented by
  each fold's frozen training-calibration identities;
- jitter L3 remains below half a sample period;
- orientation matrices are orthonormal with determinant one;
- axis-loss channel counts are exactly 1, 3, and 6.

These gates use no model predictions.

## Still not frozen

This severity protocol does not yet freeze:

- fault onset sampling;
- stochastic instance count per parent sequence;
- the fault seed namespace;
- evaluation aggregation;
- robustness acceptance criteria.

Those items must be frozen before comparative robustness execution.
