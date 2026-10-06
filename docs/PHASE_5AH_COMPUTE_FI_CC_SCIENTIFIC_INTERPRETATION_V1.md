# Phase 5AH — Frozen Compute-FI CC Scientific Interpretation V1

## Status

**SCIENTIFICALLY_INTERPRETED_FROZEN_CC_AGGREGATE_WITH_UNAVAILABLE_TIMING_PRESERVED**

## Accepted basis

Scientific interpretation begins only after Phase 5AG technically accepted the
complete Phase-5AF compute-only aggregate.

The exact frozen Phase-5Z paired-effect contract is used.

Positive degradation means worse behavior under compute fault.

## Timing availability

The frozen subject-primary aggregate explicitly marks
`median_trigger_lead_ms` as `UNAVAILABLE_WITHOUT_IMPUTATION`.

This occurs for:

- all six clean variant × operating-point lead-time rows;
- all 144 CC lead-time strata.

These rows contain `null` estimate/lower/upper fields by design.

They are not interpreted as zero.

They are not converted into a synthetic finite value.

They are not dropped.

They are not imputed.

The protocol defines a no-detected-fall lead summary as NaN and requires
missing timing-summary reporting. The subject-primary aggregation layer
therefore leaves these summaries unavailable when the three-seed finite-value
contract cannot be satisfied.

## Available primary CC results

The remaining 576 CC strata are available and retain their existing frozen
subject-bootstrap estimates and confidence intervals.

For these rows:

- positive degradation is adverse;
- negative degradation is favorable relative to the paired clean
  counterfactual;
- a CI entirely above zero is entirely adverse;
- a CI entirely below zero is entirely favorable;
- a CI spanning zero includes zero.

No new inferential interval is computed.

## Clean absolute metrics

Twenty-four clean aggregate rows are available.

Six clean median-trigger-lead rows remain unavailable without imputation.

## Equal-target descriptive macros

Phase 5AH forms equal-target descriptive macros only when every contributing
target is available.

No unavailable target is removed to make a macro computable.

If any target is unavailable, the whole corresponding macro remains
`UNAVAILABLE_WITHOUT_IMPUTATION`.

No macro confidence interval is manufactured.

## FP32/PTQ comparison

Cross-variant comparison is restricted to the ten common targets.

The four PTQ-only targets remain separate descriptive PTQ results.

A common-target comparison is available only when all ten corresponding target
effects are available in both variants.

For available comparisons, negative PTQ-minus-FP32 degradation means PTQ is
descriptively less adverse; positive means descriptively more adverse.

Unavailable timing comparisons remain unavailable.

No cross-variant significance test or inferential CI is created.

## Timing sensitivity

Phase-5Z defines a separate event-cluster timing sensitivity procedure.

Phase 5AH does not run that bootstrap because it is not part of the technically
accepted aggregate being interpreted here.

The unavailable subject-primary timing summaries therefore remain unavailable.

## No score or selection

Directional counts are descriptive only.

They are not a scalar robustness score and are not used to select a model,
target, family, seed, fold, checkpoint, or operating point.

## Frozen boundary

Phase 5AH performs no threshold retuning, checkpoint selection, fault
resampling, protocol change, CSC generation, OnField evaluation, model forward,
or new fault execution.

Outer-result interpretation does not feed back into the frozen protocol.
