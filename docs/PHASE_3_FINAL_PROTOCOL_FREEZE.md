# Phase 3 Final Protocol Freeze

## Status

PHASE 3 PROTOCOL FROZEN

## Primary protocol

- 300 ms
- 100 Hz
- 30 samples per window
- 50 percent overlap
- 15-sample stride
- 150-ms stride
- 61 primary UniVRFall + KFall subjects
- 6,309 trials
- 273,830 windows

## Historical preprocessing source

The historical preprocessing code was recovered directly.

Annotated fall trials use two independent regions:

1. pre-onset Activity
2. onset-to-impact Falling

The two resulting window arrays are concatenated in that order.

Files not handled by the annotated-event route enter a separate whole-trial
fallback windowing branch.

## Critical index convention

Historical preprocessing used annotation frame values directly as Python
array slice indices.

UniVR:

`annotation frame == curated zero-based position`

KFall:

`annotation frame == curated zero-based position + 1`

Observed convention audit:

`[{'count': 2346, 'dataset': 'KFALL', 'impact_frame_minus_position': 1, 'onset_frame_minus_position': 1}, {'count': 573, 'dataset': 'UNIVR', 'impact_frame_minus_position': 0, 'onset_frame_minus_position': 0}]`

This explains the remaining one-window KFall discrepancies produced by the
rejected Phase-3N-R v2 zero-based-position reconstruction.

## Route reconstruction

Total trials:

6309

Total windows:

273830

Annotated event trials:

2919

Exact historical piecewise event routes:

2918

Piecewise-route mismatches:

1

Whole-trial fallback-qualified mismatches:

1

Unresolved historical routes:

0

Mapped stored windows:

273830

## Historical route exception

Event:

`KFALL_106_T27_R05`

Its frozen labels remain unchanged.

Its current event annotation remains available.

It does not reproduce the annotated piecewise preprocessing route.

It does reproduce the recovered source's whole-trial fallback geometry.

This is retained as historical lineage discordance rather than relabeled or
deleted.

## Classification ground truth

The inherited `labels.npy` files remain immutable classification ground truth.

Current onset/impact annotations do not overwrite historical labels.

## Event timing ground truth

The curated event index and oriented `FrameCounter` form the event-time layer.

FrameCounter events audited:

2919

Missing FrameCounter:

0

Annotation/FrameCounter mismatches:

0

FrameCounter discontinuity events:

0

Non-100-Hz events:

0

## Stored-window to raw-sample mapping

Ordinary and whole-trial fallback windows:

`raw_start = 15 * j`

`raw_end = raw_start + 29`

For annotated piecewise trials, Activity windows:

`raw_start = 15 * j`

For annotated piecewise Falling windows:

`raw_start = fall_start_frame + 15 * j`

For both:

`raw_end = raw_start + 29`

The historical preprocessing slice coordinate and the physical event-position
coordinate are intentionally kept separate.

## Safety lead-time metric

Physical event timing uses FrameCounter.

Sensor-only lead:

`(impact_FrameCounter - decision_window_end_FrameCounter) * 10 ms`

Runtime-adjusted lead:

`sensor-only lead - measured runtime latency`

Deadline margin:

`runtime-adjusted lead - 150 ms`

Deadline met:

`margin >= 0`

Runtime latency remains deferred until actual MCU measurement.

## INT8 calibration

Five fold-specific deterministic calibration sets are frozen.

Each contains 4,096 training-only window identities.

Validation, outer-test and OnField samples are prohibited calibration inputs.

No calibration has yet been executed.

## OnField

Retained external Activity-only IDs:

`1001-1010`

Permanently rejected:

`999`, `1000`

## Phase-3 boundary

No prospective model was trained.

No model predictions were opened.

No INT8 calibration was executed.

No fault was injected.

No outer-test model outcome was opened.

No OnField model outcome was opened.
