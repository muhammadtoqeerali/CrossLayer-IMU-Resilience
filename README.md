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

Phase 2 is complete.

The primary unprotected FP32 reference workload is frozen as:

    DATE2025_CNN_400MS_RECONSTRUCTED

Authoritative contract:

    configs/baseline/frozen_fp32_reference_v1.json

Frozen evidence includes:

- exact historical checkpoint identity
- canonical tensor-state identity
- historical 0.9 strict deployment decision
- 63,173 parameters
- 147,712 Conv/Linear MACs per 400-ms window
- validated fixed-batch FP32 ONNX graph
- zero FP32 parity tolerance violations
- zero decision differences
- deterministic repeated ONNX export
- static PTQ protocol

Final calibrated INT8 deployment is intentionally deferred until the clean
calibration partition is frozen.

Current phase:

    Phase 3 — dataset/event/split/timing/calibration protocol freeze

No CrossLayer fault characterization result has been used to choose the
baseline or quantization protocol.
