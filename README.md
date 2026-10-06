# CrossLayer-IMU-Resilience

<!-- CURRENT_PROJECT_STATUS_START -->
## Current project status

**Phase 5 — prospective compute-fault C0/CC study: COMPLETE / FROZEN / COMMITTED / PUSHED**

The repository has completed the prospective P0 compute-fault study for the
frozen **C0 (clean)** and **CC (compute-only fault)** regimes.

### Final Phase-5 estate

- **61** outer-test subjects
- **6,309** trials
- **273,830** stored 300 ms / 50% overlap windows
- **366** clean caches
- **732** compute-fault shards
- **20,170,008** outer fault identities
- **29,798,820** fault records
- production lineage: **1** exact grandfathered R3/V2 transient shard +
  **731** R5/V3 shards

### Accepted scientific reporting

Positive paired degradation means **worse under compute fault**.

Of the **720** primary CC strata:

- **576** are available for subject-primary interpretation;
- **144** are `median_trigger_lead_ms` strata that remain
  **UNAVAILABLE_WITHOUT_IMPUTATION**.

Of the **30** clean aggregate rows:

- **24** are available;
- **6** median-trigger-lead rows remain unavailable without imputation.

No unavailable timing values were dropped or imputed.

Across the available descriptive equal-target comparisons:

- persistent faults are more adverse than transient faults in **24 / 24**
  matched comparisons;
- on the **10 targets common to FP32 and PTQ**, PTQ is descriptively less
  adverse in **21 / 24** available comparisons and more adverse in **3 / 24**.

These results **do not support a universal FP32/PTQ superiority claim** and
were not used to select a model variant, target, fault family, checkpoint,
threshold, operating point, or fault sample.

### Governance boundary

Phase-5 outer results are frozen and must not feed back into model selection,
threshold retuning, checkpoint selection, fault resampling, or Phase-5 protocol
changes.

Phase 5 makes **no CSC claim**, **no OnField fall-performance claim**, and
**no MCU / physical-realism claim**.

Any subsequent combined **C0 / CS / CC / CSC** study must be prospectively
specified and frozen independently before execution.

### Repository state

Phase-5 completion commit:

`af87658588f6fbb9f475beea8a446e548292a566`

Detailed Phase-5 evidence, manifests, qualification records, scientific
interpretation, and final governance freeze are available under:

- `configs/evaluation/`
- `configs/faults/`
- `experiments/phase_05/`
- `manifests/`
- `docs/`
- `tests/`
<!-- CURRENT_PROJECT_STATUS_END -->

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
