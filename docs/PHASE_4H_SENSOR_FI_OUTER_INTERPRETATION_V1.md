# Phase 4H Sensor-FI Outer Interpretation v1

Status: **FROZEN_INTERPRETATION_OF_HELD_OUT_OUTER_RESULTS**

This document summarizes the already-frozen held-out Phase 4H sensor-fault
report. It is descriptive/inferential reporting only and cannot feed back
into prospective tuning.

## Reporting inventory

- Family × severity inferential records: 4536
- Bootstrap replicates: 10000
- Required strata: overall 61-subject, UniVR, KFall
- Execution interpretability: EXECUTION_INTERPRETABLE

## Direction-status inventory

Across all three strata:

- DEGRADATION_SUPPORTED: 2677
- IMPROVEMENT_SUPPORTED: 268
- NO_DIRECTIONAL_CONCLUSION: 1555
- UNRESOLVED: 36

For the overall 61-subject stratum:

- DEGRADATION_SUPPORTED: 928
- IMPROVEMENT_SUPPORTED: 96
- NO_DIRECTIONAL_CONCLUSION: 476
- UNRESOLVED: 12

`NO_DIRECTIONAL_CONCLUSION` is not equivalence.

## Main held-out patterns

### Axis loss

L3 axis loss is a severe repeated failure mode across both frozen model
variants and all operating points. Recall and balanced accuracy degrade
strongly. Precision and sensor-lead inference are unresolved at this
condition because valid subject-level coverage is incomplete.

### Other sensor faults

Bias, noise, orientation, clipping/saturation, dropout, delay, drift, scale
factor, and stuck-channel faults show severity- and operating-point-dependent
degradation patterns. Frame-loss conditions are often non-directional at the
frozen P0 severities.

Some frozen conditions support metric-specific improvement. These results are
reported as such; they are not interpreted as a global robustness pass and
must not trigger post-hoc operating-point selection.

### FP32 versus qualified PTQ

The clean FP32 and qualified PTQ references are descriptively close across the
three frozen operating points. No equivalence test was predeclared, so this
must not be converted into an equivalence claim.

### Dataset strata

UniVR and KFall should not be collapsed conceptually. Their direction statuses
disagree for a nontrivial set of fault conditions, particularly timing.

## Missingness

Exactly 36 inferential records are unresolved. Their only structures are:

- axis_loss / L3 / precision
- axis_loss / L3 / sensor_lead_ms

No missing value is replaced by zero.

## Governance boundary

These held-out outer-test results may now be reported and interpreted.

They may **not** be used to alter:

- model weights or architecture;
- thresholds or operating points;
- fault families, variants, or severities;
- calibration;
- reporting rules;
- statistical procedures.

No binary global robustness label is generated.
