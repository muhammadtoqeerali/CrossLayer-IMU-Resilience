# Phase 4I OnField External Execution Completion v1

Status: **COMPLETE_IMMUTABLE_ONFIELD_ACTIVITY_ONLY_EXTERNAL_EXECUTION**

Completion gate:
`QUALIFIED_COMPLETE_EXTERNAL_ACTIVITY_ONLY_EXECUTION`.

The independent retained OnField Activity-only evaluation completed using the
predeclared frozen Phase-4I protocol.

## Executed estate

- retained subjects: 10;
- retained trials: 16;
- retained 300-ms Activity windows: 1,023,337;
- FP32 checkpoints: all 15 seed×fold members;
- qualified PTQ checkpoints: all 15 seed×fold members;
- operating points: balanced, low_false_alarm, timely_150ms;
- model-window evaluations: 30,700,110;
- threshold applications: 92,100,330;
- trial/checkpoint/operating-point rows: 1,440;
- subject/checkpoint/operating-point rows: 900;
- subject-estate rows: 60;
- cohort summary rows: 6;
- subject-bootstrap replicates: 10,000.

## Integrity boundary

The run:

- stored no raw probability streams;
- generated no fall-side metrics;
- selected no checkpoint;
- selected no operating point;
- tuned no threshold;
- generated no binary global robustness label.

The result directory is immutable evidence for external Activity-side
false-alarm behavior.

## Interpretation boundary

After this completion record, the six cohort summaries may be read and
reported for:

- Activity specificity;
- false triggers per Activity hour;
- supporting Activity-side false-alarm quantities.

The retained OnField cohort contains no Falling windows and cannot support
claims about fall recall, event recall, missed-fall rate, lead time, fall
recovery, two-class balanced accuracy, or external fall-detection
effectiveness.

The external results remain prohibited from feeding back into tuning,
checkpoint selection, threshold selection, or operating-point selection.
