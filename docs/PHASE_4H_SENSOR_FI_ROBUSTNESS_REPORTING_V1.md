# Phase 4H — Robustness Acceptance and Reporting v1

**Status:** FROZEN_PRE_SANITY_RESULT
**Evidence tier:** P0

## Why there is no arbitrary global robustness threshold

The project currently has no validated application-level safety margin
and no P2/P3 evidence establishing an acceptable universal loss in
recall, specificity, balanced accuracy, event recall, or lead time.

Therefore this protocol does **not** invent a percentage-point margin
and does **not** create a binary global `robust / not robust` label.

That would overstate what P0 software fault injection can support.

## Execution acceptance

An experiment may be labelled `EXECUTION_INTERPRETABLE` only when all
frozen protocol hashes, checkpoints, fault instances, result dimensions,
and missingness accounting checks pass.

This is a protocol-compliance acceptance gate, not a model-robustness
claim.

## Condition-level direction status

For each frozen family, severity, model variant, metric, and required
stratum, define paired degradation as:

`C0 - CS`

for higher-is-better metrics.

Using the frozen paired subject-bootstrap 95% interval:

- `DEGRADATION_SUPPORTED`: lower CI bound > 0;
- `IMPROVEMENT_SUPPORTED`: upper CI bound < 0;
- `NO_DIRECTIONAL_CONCLUSION`: CI includes 0;
- `UNRESOLVED`: non-finite result or incomplete required coverage.

`NO_DIRECTIONAL_CONCLUSION` is not equivalence and must not be described
as `no degradation`, `safe`, or `robust`.

## Complete reporting

All 12 sensor-fault families and all three L1/L2/L3 severities must be
reported.

Both current Phase-4 model variants are required:

- prospective 300 ms FP32;
- qualified mixed-precision PTQ v7.

All three checkpoint seeds and all five outer folds are required for
final held-out evaluation.

Family/severity macros and every predeclared individual variant are
reported. Stochastic realizations cannot be selectively removed.

Clean C0, faulted CS, paired degradation, 95% CI, direction status,
coverage, and denominators must appear together.

## Claim boundary

Allowed claims are explicitly P0 and condition-specific.

This phase does not establish physical realism, hardware fault
tolerance, HIL robustness, MCU fault tolerance, field robustness,
safety certification, P2 robustness, or P3 robustness.

Outer-test results cannot change severity, sampling, aggregation,
reporting, fault-family selection, or metric selection.

Any held-out result suggesting a protocol change requires preserving
the original result, creating a new version, returning to
development/calibration, and refreezing before another held-out run.
