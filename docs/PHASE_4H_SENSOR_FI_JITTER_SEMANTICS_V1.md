# Phase 4H — Executable Jitter Semantics v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0
**Physical-realism claim:** false

## Problem resolved

The historical preprocessing pipeline constructs 300 ms windows from
ordered rows at a nominal 100 Hz cadence. It does not consume source
timestamps when constructing model inputs.

Therefore changing a timestamp field alone would not change the model
input and would be an ineffective jitter injection.

## Frozen P0 semantics

For frame `i`:

`t_i = i × 10 ms`

Draw exactly one deterministic displacement per frame:

`epsilon_i ~ Uniform(-J, +J)`

where the frozen severity levels are:

- L1: `J = 1 ms`
- L2: `J = 2 ms`
- L3: `J = 4 ms`

The same `epsilon_i` applies to all six effective accelerometer and
gyroscope channels in that frame.

Define:

`q_i = clip(t_i + epsilon_i, 0, t_last)`

For each effective signal channel, the jittered value is the
piecewise-linear interpolation of the original source sequence at
`q_i`.

The output remains an ordered N×9 row sequence. The historical 5 Hz
per-window filtering, 30-sample windows, 15-sample stride, and
IMUNormalizer are then applied unchanged.

## Ordering

The largest frozen jitter bound is 4 ms.

Because:

`4 ms < 10 ms / 2`

adjacent displaced query times remain strictly ordered. The operator
does not reorder or duplicate frames.

## Boundaries

When a displaced query would fall outside the available trial support,
it is clipped to the first or last nominal source time. This is
equivalent to endpoint hold beyond source support.

## Channels

Channels 0–5 are time-warped.

Euler channels 6–8 are preserved exactly.

FrameCounter is metadata and is not modified.

## Claim boundary

This is a deterministic P0 software timing-stress model.

It is not evidence for:

- physical IMU clock-jitter realism;
- packet-network jitter realism;
- hardware timestamp behavior;
- HIL timing behavior;
- MCU timing behavior.

No model outcome, validation outcome, outer-test outcome, or OnField
outcome was used to define these semantics.
