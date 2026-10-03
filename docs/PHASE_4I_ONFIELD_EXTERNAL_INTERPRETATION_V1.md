# Phase 4I OnField External Interpretation v1

Status: **FROZEN_ACTIVITY_ONLY_EXTERNAL_INTERPRETATION**

This record interprets only the six already-frozen Activity-side cohort
summaries from the independent retained OnField evaluation.

## External cohort

- 10 retained subjects;
- 16 trials;
- 1,023,337 300-ms Activity windows;
- zero Falling windows;
- all 15 seed×fold checkpoints per model variant;
- all three frozen validation-selected operating points;
- 10,000-replicate subject bootstrap uncertainty.

## Frozen cohort summaries

### Prospective FP32 300 ms

| Operating point | Activity specificity | False triggers / Activity hour |
|---|---:|---:|
| balanced | 0.999010 [0.998652, 0.999336] | 18.328044 [12.971141, 23.659355] |
| low_false_alarm | 0.999005 [0.998651, 0.999329] | 4.044666 [2.387899, 6.072854] |
| timely_150ms | 0.995899 [0.994766, 0.996985] | 70.148137 [53.383819, 85.851338] |

### Qualified static PTQ v7

| Operating point | Activity specificity | False triggers / Activity hour |
|---|---:|---:|
| balanced | 0.998950 [0.998589, 0.999289] | 19.446624 [13.749373, 24.889839] |
| low_false_alarm | 0.998947 [0.998592, 0.999280] | 4.289012 [2.625922, 6.352724] |
| timely_150ms | 0.995657 [0.994513, 0.996824] | 74.611724 [57.455505, 91.100308] |

Intervals are the predeclared deterministic 10,000-replicate subject
bootstrap intervals.

## Interpretation

Activity specificity is high across all six frozen conditions.

False-trigger burden is strongly operating-point dependent in this external
Activity-only cohort. For both FP32 and PTQ, the observed ordering is:

1. `low_false_alarm`: lowest false triggers per Activity hour;
2. `balanced`: intermediate;
3. `timely_150ms`: highest.

This is a descriptive external result, **not an operating-point selection
rule**. All three operating points remain frozen and must remain reported.

FP32 and qualified PTQ are descriptively close at corresponding operating
points. No model-difference test or equivalence test was predeclared, so the
external results do not establish superiority or equivalence.

Subject-level false-trigger rates vary materially within conditions. This is
consistent with the predeclared protocol decision to use subjects, not
overlapping windows, as uncertainty units.

## Claim boundary

The retained OnField cohort contains zero Falling windows. Therefore this
evaluation cannot establish:

- fall recall;
- event recall;
- missed-fall rate;
- sensor lead time;
- recovery performance;
- meaningful external two-class balanced accuracy;
- external fall-detection effectiveness.

The results cannot be used to choose or modify checkpoints, thresholds,
operating points, model weighting, calibration, fault protocols, or any
scientific parameter.

No binary global robustness label is generated.
