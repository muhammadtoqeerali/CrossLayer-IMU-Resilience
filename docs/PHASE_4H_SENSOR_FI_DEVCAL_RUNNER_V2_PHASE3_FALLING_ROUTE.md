# Phase 4H — Dev/Cal Runner v2: Phase-3 Falling Route

**Status:** FROZEN_PRE_QUALIFICATION_RESULT

Runner v1 is preserved unchanged.

## Single scientific repair

Runner v2 binds Falling-window source reconstruction to the already
frozen Phase-3 historical preprocessing route:

`raw_start = fall_start_frame + 15 * falling_local_index`

The annotation frame value is used directly as the Python source-array
index.

This is intentionally distinct from physical event timing, where the
curated zero-based event position / oriented FrameCounter correspondence
remains authoritative.

Consequently:

- UniVR: `fall_start_frame == fall_start_position`
- KFall: `fall_start_frame == fall_start_position + 1`

The historical `KFALL_106_T27_R05` exception remains a whole-trial
fallback with retained Activity labels and does not enter the
Falling-window branch.

## Qualification

Before any outer-test execution, v2 must:

1. exactly reconstruct all 2,269 training-calibration-represented
   Falling parents;
2. exercise all seven sequence fault families on a deterministic
   Falling parent in each fold;
3. feed the exact same corrupted window to paired FP32/PTQ models;
4. use no performance metric, threshold, probability delta, validation,
   outer-test, or OnField outcome as an acceptance criterion.

No robustness claim is permitted from this qualification.
