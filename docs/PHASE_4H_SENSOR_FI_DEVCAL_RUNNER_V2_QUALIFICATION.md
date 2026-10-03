# Phase 4H — Runner v2 Falling-Route Qualification

**Status:** QUALIFIED_PHASE3_FALLING_ROUTE_TRAINING_CALIBRATION

Runner v1 remains preserved.

Runner v2 makes one scientific repair: Falling windows use the frozen
Phase-3 historical slice origin

`fall_start_frame + 15 * falling_local_index`

rather than treating the curated zero-based event position as the
historical Python-array slice index.

## Qualification evidence

All 2,269 training-calibration-represented Falling parents reconstructed
their stored windows exactly.

- KFall: 1,813 / 1,813 exact with annotation-frame slicing.
- UniVR: 456 / 456 exact with annotation-frame slicing.
- Generic zero-based-position slicing reproduced only the 456 UniVR
  parents, confirming why that interpretation is invalid for KFall.

The sole historical exception `KFALL_106_T27_R05` remains an
Activity-labelled whole-trial fallback and does not enter the Falling
branch.

All seven sequence FI families were additionally exercised through the
corrected Falling route in every fold with seed-42 paired FP32/PTQ
models.

No performance metric, threshold, probability delta, validation,
outer-test, or OnField outcome was used for qualification.

The next step is a separately frozen final outer-test execution
manifest.
