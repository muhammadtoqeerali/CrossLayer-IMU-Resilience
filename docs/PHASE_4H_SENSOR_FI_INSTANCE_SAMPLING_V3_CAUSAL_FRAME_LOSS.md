# Phase 4H — Instance Sampling v3: Causal Frame-Loss Repair

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0

## Why v3 exists

The frozen frame-loss severity semantics require a lost complete frame
to be reconstructed by **causal forward-fill from the last available
complete frame**.

An implementation-readiness audit found that sampling v2 permitted
frame-loss onset at sample 0.

At sample 0 there is no prior complete frame. Therefore those instances
cannot satisfy the frozen causal reconstruction rule.

This was discovered before any sensor fault was executed against a
model and before any robustness result existed.

## Single change from v2

For `frame_loss` only:

- v2 earliest eligible onset: `0`;
- v3 earliest eligible onset: `1`.

The eligible onset set is therefore:

`1 ... parent_length - duration`

inclusive, before the same deterministic three-stratum seeded sampling
procedure is applied.

## Unchanged

V3 does not change:

- fault families;
- severity values;
- frame-loss durations;
- target channels;
- stochastic replicate count;
- early/middle/late stratum count;
- seed namespace;
- seed payload;
- partition policy;
- physical-realism claim.

Dropout continues to permit onset 0 because its frozen replacement
semantics do not require a prior complete frame.

## Consequence for replay identity

Frame-loss seeds remain unchanged.

Some frame-loss onsets change because the causal eligible set changes.
Their canonical replay IDs consequently change.

Non-frame-loss instances must remain exactly identical to v2.

## Downstream governance

The previously qualified aggregation and reporting scientific rules do
not need scientific changes, but their governance lineage currently
references sampling v2.

Therefore they must be rebound to qualified sampling v3 before any
fault/model execution.

Jitter implementation semantics also remain to be frozen because the
historical preprocessing pipeline consumes row order rather than source
timestamps.
