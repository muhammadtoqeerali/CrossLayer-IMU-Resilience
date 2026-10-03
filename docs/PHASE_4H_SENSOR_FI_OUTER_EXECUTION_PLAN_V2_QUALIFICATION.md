# Phase 4H — Outer Execution Plan v2 Qualification

**Status:** QUALIFIED_PRE_OUTER_EXECUTION_PLAN_V2_HISTORICAL_TRUTH

The v2 plan is statically qualified before any outer model or fault
execution.

The only scientific clarification relative to preserved v1 is the
trial-level historical truth denominator:

- stored-label inventory: 3391 Activity / 2918 Falling;
- historical evaluator truth: 3390 Activity / 2919 Falling.

The single difference is `KFALL_106_T27_R05` in outer fold 4.

Its event annotation/risk record is retained while its stored labels are
unchanged and contain zero Falling windows. Under the recovered
historical trigger rule, it is therefore a true Falling trial but cannot
produce a valid Falling-label trigger. It is prospectively frozen as a
false negative / missed event with sensor lead NA under every condition.

No relabeling is performed.

Everything else remains unchanged from v1: 61 subjects, 6309 trials,
273830 windows, 793 shards, all 12 fault families, 42,766,632 unique
fault instances, 479,750,160 model-window evaluations, all model seeds,
both model variants, and all three frozen operating points.

No model was loaded, no prediction was computed, and no fault operator
was executed during this qualification.
