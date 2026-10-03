# Phase 4H — Instance Sampling v3 Qualification

**Status:** QUALIFIED_METADATA_REPLAY_CAUSAL_FRAME_LOSS

Sampling v3 was qualified exhaustively over the frozen fold-local
training-calibration trial identities.

The audit confirmed the previously identified v2 issue:

- v2 frame-loss instances with onset sample 0: **142**.

Under v3:

- frame-loss instances with onset sample 0: **0**;
- every frame-loss instance has a prior complete frame;
- frame-loss seeds are unchanged;
- frame-loss human-readable fault IDs are unchanged;
- all non-frame-loss sampling metadata is unchanged from v2;
- temporal bounds and input-contract validation pass.

No sensor values were mutated and no model was loaded.

The downstream aggregation/reporting scientific rules remain valid but
must be rebound from sampling v2 to qualified sampling v3 before fault
execution.

Executable jitter semantics also remain to be frozen before the full
operator engine is implemented.
