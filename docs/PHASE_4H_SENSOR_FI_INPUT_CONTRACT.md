# Phase 4H — Sensor FI Input and Injection-Layer Contract

**Status:** FROZEN_PRE_SEVERITY
**Evidence tier:** P0 software fault injection
**Date:** 2026-10-02

## What is frozen

The primary stored model input is a 300 ms, 100 Hz, 30-sample,
9-channel filtered window with 15-sample stride.

Stored channel order is:

`AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ`.

Repository provenance supports the software-unit convention:

- accelerometer: mg;
- gyroscope: mdps;
- Euler: degrees where available.

Only channels 0–5 are effective learned inputs. `IMUNormalizer`
discards Euler channels before learned convolution.

## Exact provenance checks

KFall raw-to-oriented data exactly reproduce the repository's current
reflection, rotation, unit conversion, and Euler-zeroing transform.

A deterministic KFall frozen 300 ms window reproduces exactly from its
oriented source trial after the historical 5 Hz filter.

For UniVR subject 10 / task 10 / trial 1, all 31 stored frozen windows
reproduce exactly and uniquely from the oriented source trial at
starts 0, 15, 30, ..., 450.

## Injection layers

Value faults that do not require temporal continuity are injected into
the stored filtered 30x9 window before `IMUNormalizer`:

- bias;
- scale factor;
- noise;
- clipping/saturation;
- axis loss.

Faults requiring sequence state or timing are injected into the
source-faithful oriented trial before filtering and windowing:

- drift;
- stuck channel;
- dropout;
- frame loss;
- jitter;
- delay;
- orientation.

The orientation family acts on accelerometer/gyroscope axes; mutating
discarded Euler channels is not the primary orientation-fault
definition.

## Replay and leakage rules

Every fault instance records its family, layer, target channels,
onset, duration/persistence, explicit severity metadata, provenance,
seed, parent sequence, partition, and deterministic replay ID.

Splitting precedes stochastic fault injection.

Clean and corrupted variants retain the same parent sequence and
partition.

Same-backbone causal comparisons reuse the same fault instance.

## Severity boundary

No severity values are frozen by this contract.

No default magnitude, severity grid, clipping limit, delay, dropout
rate, noise level, drift rate, or orientation magnitude is introduced
here.

Software magnitudes may be expressed in documented software units
such as mg, mdps, samples, or milliseconds, but this does not make
them physically realistic.

Physical realism remains unsupported until appropriate P2/P3 evidence
exists.

Validation, outer-test, OnField, and fault-result outcomes are not
used to select severity values in this freeze.
