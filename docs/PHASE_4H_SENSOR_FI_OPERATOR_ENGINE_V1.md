# Phase 4H — Complete Sensor-FI Operator Engine v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0

## Scope

This engine implements all 12 frozen sensor-fault families:

Window-value layer:

- bias
- scale factor
- Gaussian noise
- clipping/saturation
- axis loss

Sequence layer:

- drift
- stuck channel
- dropout
- frame loss
- jitter
- delay
- orientation

## Reference-scale rule

Bias, drift, and clipping consume the frozen fold/channel q99 absolute
reference.

Noise consumes the frozen fold/channel robust-sigma reference.

The engine never estimates these references from the parent being
faulted. They must be supplied by the caller from the already-frozen
fold-local training-calibration severity materialization.

## Causal temporal operators

Stuck-channel holds the last valid sample immediately before fault
onset.

Frame loss forward-fills the last complete effective-channel frame
before onset. Sampling v3 guarantees onset >= 1.

Delay shifts all six effective channels causally and fills the prefix
with the first available frame.

## Jitter

The operator delegates to the separately frozen and qualified jitter
semantics: bounded deterministic frame-level software time warp with
piecewise-linear resampling.

## Orientation

The same proper right-handed 3D rotation is applied to the
accelerometer and gyroscope triads.

Euler channels are unchanged.

## Safety and scientific boundary

The engine:

- works only on in-memory copies;
- never writes dataset files;
- never loads a model;
- never selects thresholds;
- rejects OnField fault execution;
- makes no P2/P3 physical-realism claim.

Operator qualification is performed before any model robustness result.
