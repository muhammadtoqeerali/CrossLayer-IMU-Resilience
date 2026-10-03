# Phase 4I OnField External Evaluation Protocol v1

Status: **FROZEN_PRE_ONFIELD_MODEL_INFERENCE**

This protocol was frozen before any prospective FP32/PTQ model was evaluated
on retained OnField windows.

## External cohort

The evaluation uses the canonical Phase-3 root:

`/mnt/hdd16T/protechto/data/OnField/segments/300ms_50ov_npseg_filt_binary`

The retained cohort is fixed:

- subjects: 1001–1010;
- subjects: 10;
- trials: 16;
- 300-ms windows: 1,023,337;
- labels: Activity only;
- Falling windows: 0;
- rejected storage IDs: 999 and 1000.

## Model estate

No single deployment checkpoint exists.

For each model variant:

- seeds: 42, 123, 2025;
- folds: 1–5;
- all 15 seed×fold members are mandatory;
- no member may be selected or weighted by OnField performance.

The two frozen model variants are:

1. prospective FP32 300-ms CNN;
2. qualified static PTQ v7 mixed-precision CNN.

No probability-level ensemble is introduced.

## Operating points

Every checkpoint is evaluated at all three validation-selected operating
points:

- balanced;
- low_false_alarm;
- timely_150ms.

The same checkpoint probability stream is reused across all three operating
points. The frozen threshold table contains 45 rows: 15 checkpoints × 3
operating points.

OnField cannot alter thresholds or choose an operating point.

## Activity-only metrics

Primary subject-level metrics:

- Activity specificity;
- false triggers per Activity hour.

Supporting quantities:

- false-positive windows;
- false trigger episodes;
- Activity seconds;
- Activity window count.

A false trigger episode uses the unchanged historical trigger-episode rule
with the frozen threshold and required-consecutive count. Trigger state resets
at each trial boundary.

For a trial containing N windows, Activity duration is reconstructed as:

`((N - 1) * 15 + 30) / 100` seconds.

Overlapping windows are not treated as independent inferential samples.

## Checkpoint aggregation

Every checkpoint-specific result is retained.

For cohort summaries, each subject is aggregated without selection:

1. equal mean across five folds within each seed;
2. equal mean across the three seeds.

Because the estate is complete, every checkpoint has weight 1/15.

No checkpoint performance weighting is allowed.

## Uncertainty

The external subject is the sampling unit.

For each model variant × operating point × primary metric:

- first construct the subject-level 15-checkpoint estate summary;
- then bootstrap the 10 subjects with replacement;
- use 10,000 deterministic bootstrap replicates;
- report a percentile 95% interval.

Windows and trials are not bootstrapped as independent observations.

## Claim boundary

OnField may support Activity-side false-alarm statements only.

It cannot support:

- fall recall;
- event recall;
- missed-fall rate;
- fall timing or sensor lead;
- fall recovery;
- meaningful two-class balanced accuracy;
- external fall-detection effectiveness;
- FP32/PTQ equivalence.

No binary global robustness label is generated.

## Expected execution scale

- model-window evaluations: 30,700,110;
- threshold applications: 92,100,330;
- checkpoint-specific subject rows: 900;
- checkpoint-specific trial rows: 1,440;
- subject estate-summary rows: 60;
- cohort summary rows: 6.
