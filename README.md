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

Current phase:

    Phase 0 — scientific and repository foundation

No primary task model is frozen yet.

No sensor-fault taxonomy is frozen yet.

No compute-fault taxonomy is frozen yet.

No final MCU target is frozen yet.

No publication-facing experimental result exists yet.
