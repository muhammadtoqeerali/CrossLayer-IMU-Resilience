# CrossLayer-IMU-Resilience

Cross-layer runtime resilience for safety-critical wearable Edge AI under
sensor and embedded-computation faults.

## Core objective

This project studies the complete wearable TinyML system rather than only
the classifier.

The main scientific question is:

> Can a resource-constrained wearable AI system detect and mitigate
> degradation originating from both physical IMU sensing and embedded
> computation while preserving safety-critical timing and resource
> constraints?

Conceptual pipeline:

    IMU
      ->
    Sensor Integrity
      ->
    TinyML Inference
      ->
    Compute Integrity
      ->
    Runtime Supervisor
      ->
    Recovery / Fallback
      ->
    Safety Decision

## Mandatory operating conditions

    C0   clean sensor + clean computation
    CS   sensor fault + clean computation
    CC   clean sensor + compute fault
    CSC  sensor fault + compute fault

Sensor faults and compute faults are intentionally kept scientifically
distinct.

## Research sequence

The project is developed in this order:

    baseline
      ->
    sensor fault engine
      ->
    compute fault engine
      ->
    cross-layer vulnerability characterization
      ->
    sensor integrity
      ->
    compute integrity
      ->
    adaptive runtime supervisor
      ->
    recovery / fallback
      ->
    embedded validation

## Repository philosophy

Engineering infrastructure may be reused from earlier projects such as
RC-RGD-IMU.

Scientific assumptions are not inherited automatically.

Every reused component must be audited first.

## Structure

    configs/       experimental and frozen protocols
    docs/          scientific contracts and project history
    models/        task architectures
    src/           reusable implementation
    experiments/   phase-specific experiment entry points
    embedded/      portable and MCU-specific runtime
    manifests/     provenance and immutable run metadata
    scripts/       utilities
    tests/         scientific and engineering tests
    requirements/  dependency specifications

## Current state

Phase 1 is complete.

The repository now contains a clean CrossLayer-native migration of the
historical DATE-2025 400-ms pre-impact fall-detection workload.

Validated Phase-1 evidence includes:

- exact historical checkpoint located and SHA-256 verified
- protected task model migrated without prior reliability mechanisms
- 63,173 parameters verified
- exact logit parity on 68 deterministic vectors
- exact penultimate-feature parity
- historical decision-semantic parity
- task-state reload parity
- inherited OOD/reliability/integrity logic excluded from the clean baseline

Primary candidate:

    DATE2025_CNN_400MS_RECONSTRUCTED

The primary candidate is not yet frozen.

Next:

    Phase 2 — baseline reproduction, selection, quantization and freeze

Sensor-fault, compute-fault and runtime-protection protocols remain unfrozen.
No publication-facing CrossLayer fault result exists yet.
