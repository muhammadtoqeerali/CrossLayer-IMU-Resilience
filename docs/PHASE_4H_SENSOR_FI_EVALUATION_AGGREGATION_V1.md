# Phase 4H — Sensor-FI Evaluation Aggregation v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0
**Model robustness results seen:** no

## Core principle

The primary independent inferential unit is the **subject**.

Overlapping 300 ms windows are not independent samples.

Stochastic fault replicates are nested within the same parent.

The three training seeds are repeated model realizations, not three
independent subjects.

The five folds are partition assignments, not five independent
experiments.

## Primary evaluation hierarchy

For each subject, family, severity, and checkpoint seed:

1. compute the requested metric for each predeclared fault variant and
   stochastic replicate;
2. average stochastic replicates with equal weight;
3. retain each variant result separately;
4. form a family-level equal-weight macro across variants;
5. average the three model seeds with equal weight;
6. produce one primary value per subject;
7. aggregate across subjects.

No family receives extra weight merely because it has more channel,
sign, gain, or rotation-axis variants.

## Classification metrics

Primary subject-level metrics:

- falling recall;
- activity specificity;
- balanced accuracy.

Secondary subject-level metrics:

- precision;
- F1.

Pooled window metrics may be shown descriptively but are not used as
independent observations for uncertainty.

Undefined metrics remain NA and are accompanied by numerator,
denominator, and coverage counts. They are never silently replaced by
zero.

## Fall-event and timing metrics

Annotated fall events are secondary independent units nested within
subjects.

An event is detected if at least one eligible mapped evaluation window
crosses the frozen decision threshold.

Offline sensor lead is:

`impact time - window end time`.

For detected events, timing uses the earliest threshold-positive
eligible window.

A missed event has no imputed lead time; it remains a miss and timing is
NA.

System lead including measured inference/supervisor latency is outside
this Phase-4H offline sensor-FI analysis.

## Paired degradation

C0 clean and CS sensor-fault outcomes must be paired on:

- subject;
- parent window/sequence;
- checkpoint seed/fold;
- threshold;
- preprocessing;
- fault instance where a same-backbone comparison is made.

For higher-is-better metrics:

`degradation = C0 - CS`.

Positive degradation therefore means worse performance under sensor
fault.

Both absolute C0 and absolute CS values are also reported.

## Uncertainty

The primary 95% interval is a paired subject bootstrap.

- 10,000 bootstrap replicates;
- sampling unit: subject;
- UniVR and KFall resampled separately;
- fixed stratum sizes: 29 UniVR + 32 KFall;
- same subject draw used for paired C0 and CS;
- deterministic SHA256-derived bootstrap seed.

Windows, fault replicates, model seeds, and folds are not independently
bootstrapped.

## Dataset reporting

The primary overall macro gives equal weight to each of the 61 subjects.

Dataset-specific UniVR and KFall summaries are also required.

## OnField

OnField is not part of Phase-4H fault generation or faulted evaluation.
It remains separate Activity-only external evaluation.

## Remaining pre-execution freeze

Only one pre-execution governance item remains after this protocol:

**robustness acceptance/reporting criteria**.

No sensor-fault/model experiment should execute before that final
criterion is frozen.
