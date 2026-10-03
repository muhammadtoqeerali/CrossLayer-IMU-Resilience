# Phase 4H Outer Aggregation Adapter v1

Status: **FROZEN_POST_EXECUTION_PRE_OUTER_METRIC_ANALYSIS**

This adapter does not alter Phase 4H aggregation or reporting science. It
only binds the already-frozen analysis policy to the immutable executor row
schema before any outer performance values are inspected.

## Timing binding

The frozen aggregation protocol names the conceptual timing metric
`sensor_lead_ms` and predeclares three subject-level timing summaries:

- `median_sensor_lead_ms`
- `q25_sensor_lead_ms`
- `q75_sensor_lead_ms`

For inferential replicate → variant → seed → subject aggregation,
`sensor_lead_ms` is therefore bound to `median_sensor_lead_ms`.

The q25/q75 fields remain descriptive timing summaries and are not substituted
for the subject median.

## Dataset-specific uncertainty

The frozen reporting policy requires three strata:

- overall 61-subject population
- UniVR
- KFall

The already-qualified overall helper performs subject resampling within the
fixed UniVR/KFall strata. Dataset-specific intervals are its one-stratum
specialization: resample subjects with replacement within exactly the frozen
29-subject UniVR or 32-subject KFall population, using the same bootstrap
namespace, seed algorithm, 10,000 replicates, and percentile interval.

No window, seed, fold, fault-instance, variant, or replicate is promoted to
an independent inferential unit.

## Missingness

Undefined metrics stay undefined. Point summaries use finite eligible subjects
with explicit coverage counts. A confidence interval is not manufactured for
an incomplete requested inferential stratum; reporting must classify such a
condition as `UNRESOLVED`.

## Governance

No outer performance value was used to define this adapter. No model,
threshold, operating point, family, severity, variant, metric, or favorable
condition is selected by this adapter.
