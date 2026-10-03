# Phase 4H — Final Outer-Test Execution v1

**Status:** FROZEN_PRE_OUTER_EXECUTION

This protocol freezes the complete held-out C0/CS execution plan before
any outer-test model prediction or sensor-fault result is produced.

## Matrix

The final P0 sensor-fault evaluation includes:

- 61 frozen outer-test subjects, each appearing exactly once;
- all five outer folds;
- three frozen checkpoint seeds;
- two model variants: fresh 300 ms FP32 and qualified PTQ-v7;
- all 12 sensor-fault families;
- all L1/L2/L3 severities;
- every predeclared variant and stochastic replicate;
- all three validation-frozen operating points:
  `balanced`, `low_false_alarm`, and `timely_150ms`.

Operating points are reported independently. Outer-test results cannot
select or average away an operating point.

## Deterministic shards

The execution unit is one subject × one block, where the 13 blocks are
C0 plus the 12 fault families.

This gives exactly 793 immutable shards.

Every shard contains all three model seeds and both FP32/PTQ variants so
one exact fault instance can be applied to all six model realizations.

## Compact output

Raw per-window probabilities are not persisted by default.

Each shard instead persists deterministic subject-condition sufficient
statistics required by the frozen aggregation/reporting contract:
trial-level TP/FN/TN/FP, event detection counts, timing summaries,
coverage counts, frozen threshold-rule fields, and all required
seed/fold/family/severity/variant/replicate/model dimensions.

This preserves the subject as the primary inferential unit and avoids
treating overlapping windows as independent samples.

## Historical trigger semantics

Threshold and `required_consecutive` come unchanged from the matching
validation-frozen seed/fold/operating-point record.

Trigger runs reset at trial boundaries.

For Falling trials, the historical valid-trigger rule requires the
entire consecutive run to lie within retained Falling-labelled windows.

The source-row slice convention and the physical timing convention
remain separate:

- source reconstruction uses Phase-3 `fall_start_frame`;
- timing uses curated zero-based positions / FrameCounter alignment;
- sensor lead is computed from the confirmation window end.

## Resume policy

A shard is reusable only when its `_SUCCESS.json`, execution config,
shard-plan hash, runner hash, coverage counts, and output hashes all
match.

Partial or mismatched shards are never aggregated and must be recomputed
under the same frozen manifest.

## Held-out governance

Outer results cannot change severity, sampling, thresholds, aggregation,
reporting, fault-family inclusion, or operating-point inclusion.

If held-out behavior motivates a change, the original result is
preserved and a new protocol version must return to development/
calibration and be refrozen.
