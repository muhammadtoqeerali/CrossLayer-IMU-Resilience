# Phase 4H — Fault-Instance Sampling and Replay Protocol v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0
**Physical-realism claim:** false

## Purpose

This protocol freezes fault-instance targeting, placement, replication,
seeding, and replay identity before any model robustness experiment.

## Stable seed namespace

All instance seeds are derived from:

`crosslayer-phase4h-sensor-fi-instance-v1`

using SHA256 over partition, fold, parent kind, parent identity, family,
severity level, variant, and replicate index.

Model identity is deliberately excluded.

Therefore the same fault instance can be replayed on FP32, mixed-precision
PTQ, and later same-backbone protected variants.

## Parent units

Stored-window faults use a parent identity:

`subject | task | trial | window`

Sequence faults use:

`subject | task | trial`

Window faults occupy all 30 samples.

## Target policy

Bias, drift, scale-factor, noise, clipping, stuck-channel, and dropout
are evaluated as single-channel variants over each of the six effective
accelerometer/gyroscope channels.

Frame loss, jitter, delay, and orientation operate on all six effective
channels together.

Axis loss retains the severity-protocol variants:

- L1: each one-channel loss;
- L2: complete accelerometer or gyroscope triad loss;
- L3: all six effective channels.

Euler channels are not targeted.

## Replication

Noise, dropout, frame loss, and jitter have exactly three stochastic
replicates per parent / severity / target variant.

Those replicates are nested within the same parent. They are not
independent subjects, events, or sequences for uncertainty analysis.

Other families have one instance per deterministic variant.

## Temporal placement

Drift and stuck-channel episodes are centered in the source trial.

Dropout and frame-loss replicates cover early, middle, and late eligible
onset strata. Within each stratum, onset is selected deterministically
from the instance seed.

Jitter, delay, and orientation persist from sample zero through the end
of the source trial.

## Instance counts

For one stored parent window, the fully enumerated v1 protocol creates
153 metadata instances.

For one source trial, it creates 138 metadata instances.

These counts define the protocol; they do not imply that overlapping
windows are statistically independent.

## Partition boundaries

The sampling protocol is designed and sanity-checked only with frozen
training-calibration identities.

Validation, outer-test, and OnField outcomes cannot tune it.

After all remaining pre-robustness rules are frozen, the same generator
may instantiate outer-test faults for final evaluation without changing
the protocol.

OnField fault generation is prohibited.

## Still not frozen

Two pre-robustness components remain:

1. evaluation aggregation protocol;
2. robustness acceptance/reporting criteria.

No model robustness experiment should execute before both are frozen.
