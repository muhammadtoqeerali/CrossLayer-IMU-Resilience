# Phase 5Z — Prospective CC Outcome-Analysis Protocol Freeze V1

## Status

**FROZEN_PROSPECTIVE_CC_OUTCOME_ANALYSIS_PROTOCOL**

This phase freezes the compute-only (`CC`) outcome-analysis contract before
opening any Phase-5 prediction JSONL, loading any outer-test label array, or
computing any CC metric.

No Phase-5 prediction outcome was inspected to design this protocol.

## Scope

The only regimes analyzed under this protocol are:

- `C0`: clean compute reference;
- `CC`: compute-fault-only execution.

`CSC` is not generated in Phase 5Z and remains reserved for the later
cross-layer phase. OnField remains excluded.

## Frozen operating points

The three operating points remain exactly:

- `balanced`;
- `low_false_alarm`;
- `timely_150ms`.

The exact 45 validation-selected rows are embedded in the protocol:

3 checkpoint seeds × 5 folds × 3 operating points.

Both `threshold` and `required_consecutive` are immutable.

No outer-test threshold selection or retuning is allowed.

## Class probability

The binary class order remains Activity=0, Falling=1.

For both the accepted Phase-5M canary and all Phase-5R fleet artifacts,
the Falling probability is read directly as element 1 of:

- `clean_softmax_values`, or
- `faulted_softmax_values`.

No logits need to be reconstructed and no canary-specific probability
approximation is permitted.

## Mixed Phase-5M / Phase-5R provenance

The accepted estate contains:

- one Phase-5M fault shard;
- one Phase-5M clean cache;
- 731 Phase-5R fault shards;
- 365 Phase-5R clean caches.

The probability semantics are identical.

For Phase-5M transient faults, the execution window is
`parent.window_index`.

For Phase-5R records, the execution window is
`execution_window_index`.

Phase-5M contains no persistent shard.

## Ground truth

Stored labels are loaded only after this freeze.

Each stored label is normalized using the exact historical evaluator rule:
Falling=1 iff the stripped upper-case label contains `FALL`; otherwise
Activity=0.

A trial is true Falling when either:

1. it contains retained Falling windows, or
2. it has a frozen risk/timing record.

The historical trial `KFALL_106_T27_R05` therefore remains a true fall via
its frozen risk record while its stored labels remain unchanged.

No relabeling is performed.

## Trial reconstruction

Windowing remains:

- 100 Hz;
- 300 ms;
- 30 samples;
- 50% overlap;
- 15-sample / 150-ms stride.

Activity-window ends follow the frozen historical ordering.

Falling-window ends are anchored to the frozen fall-start position.

FrameCounter remains authoritative for annotated event timing.

Sensor-only trigger lead is the frozen quantity:

`(impact_FrameCounter - decision_window_end_FrameCounter) * 10 ms`.

Runtime-adjusted timing is not claimed in this compute-FI phase.

## Scenario construction

### Clean C0

A clean scenario contains all chronological clean probabilities for one
model variant, checkpoint seed, outer subject, and trial.

### Compute-fault CC

The primary fault identity is `outer_instance_id`.

A CC scenario is reconstructed from the clean parent-trial sequence and
exactly one fault identity.

For transient faults, exactly one window is replaced by its faulted
probability.

For persistent faults, all records carrying the same `outer_instance_id`
must cover the frozen onset through the final trial window contiguously;
those windows replace the corresponding clean probabilities.

Different `outer_instance_id` values are never combined into one
pseudo-trial.

### Paired clean counterfactual

Every CC identity receives an identical clean parent-trial counterfactual.

This paired clean distribution is the reference for CC degradation.
Therefore the CC comparison does not contrast an injection-weighted fault
distribution against a differently weighted unique-trial clean distribution.

## Trigger semantics

The exact Phase-4 implementation is retained.

A window satisfies the threshold when:

`P(Falling) >= threshold`.

A run triggers after `required_consecutive` satisfying windows.

After one trigger episode, the detector remains disarmed until a
below-threshold window resets and rearms it.

On true-fall trials, a valid trigger requires the full consecutive trigger
span to lie in retained Falling-labelled windows.

The confirmation window is:

`trigger_start + required_consecutive - 1`.

Timing uses the confirmation window.

## Metrics

The exact Phase-4 event metrics remain:

- balanced accuracy;
- macro F1;
- fall recall;
- fall precision;
- Activity specificity;
- recall by 150 ms;
- false triggers per Activity hour;
- median trigger lead;
- mean trigger lead;
- TN / FP / FN / TP.

Primary reporting metrics are:

- balanced accuracy;
- fall recall;
- false triggers per Activity hour;
- recall by 150 ms;
- median sensor-only trigger lead.

No new composite robustness score is created.

## False-trigger denominator

On Activity trials, all trigger episodes are false episodes and Activity
duration is the last reconstructed window end divided by sampling rate.

On fall trials, pre-fall/non-Falling trigger episodes are false episodes and
the Activity duration is fall-start position divided by sampling rate.

The rate remains:

`false episodes / max(Activity seconds / 3600, 1e-9)`.

## Timely detection

`recall_by_150ms` counts true-fall scenarios whose confirmation-window
sensor-only lead is at least 150 ms.

Its denominator is all true-fall scenarios, including missed events.

Lead-time summaries include detected true-fall scenarios with valid impact
positions only. Missing lead summaries are never imputed.

## Paired degradation direction

Positive degradation always means worse under compute fault.

For higher-is-better metrics:

`degradation = paired clean - CC`.

For false triggers/hour:

`degradation = CC - paired clean`.

For trigger lead:

`degradation = paired clean median lead - CC median lead`.

## Primary strata

Results remain separate by:

- model variant;
- persistence;
- operating point;
- fault family;
- representation class;
- target name;
- target role.

Transient and persistent results are never pooled for primary reporting.

FP32 and PTQ results are never pooled for primary reporting.

Bit position and element index are secondary descriptive strata.

## Aggregation

Each outer subject belongs to exactly one outer fold.

Within a subject and reporting stratum, the three checkpoint seeds
42, 123, and 2025 receive equal weight.

Subject-level seed-equal estimates then receive equal weight across outer
subjects.

Overlapping windows, fault identities from the same trial, and repeated
trial scenarios are not treated as independent inferential units.

Target-stratified paired effects are primary.

Any aggregate CC summary is metric-specific; no scalar score is allowed.

Targets are macro-averaged rather than weighted by record count.

FP32-versus-PTQ common-target comparison uses only the ten targets available
to both variants.

The PTQ fourteen-target summary is separate and descriptive.

## Uncertainty

Primary uncertainty is a 95% nonparametric subject-cluster percentile
bootstrap with 10,000 replicates and RNG seed `20261006`.

A resampled subject carries all of its trials, fault identities, overlapping
windows, targets, and checkpoint seeds.

Paired deltas remain paired inside the bootstrap.

Timing additionally receives an event-cluster bootstrap sensitivity analysis
with 10,000 replicates and RNG seed `20261007`.

No p-value is required.

## Integrity / abort behavior

Before any prediction row is parsed, the analyzer must verify accepted
artifact success markers and the recorded file SHA256 values.

The accepted estate remains exactly:

- 732 fault shards;
- 366 clean caches;
- 1,098 artifact directories;
- 20,170,008 outer fault identities;
- 31,441,800 total clean+fault model-window records.

Malformed JSON, missing identities, duplicate identities, missing clean
windows, missing persistent suffix rows, label-length mismatches, unknown
schemas, or absent frozen thresholds are fatal.

Bad records are never silently skipped.

Non-finite fault outputs are retained and counted. They are not imputed or
discarded.

## Freeze boundary

At completion of Phase 5Z:

- no Phase-5 prediction JSONL has been opened;
- no Phase-5 prediction JSON has been deserialized;
- no outer label array has been loaded;
- no threshold has been applied to an outer prediction;
- no CC metric has been computed;
- no aggregate CC result exists;
- no CSC result exists;
- no model forward has occurred;
- no fault execution has occurred.

Only after this protocol is frozen may an outcome-analysis implementation be
written and qualified.
