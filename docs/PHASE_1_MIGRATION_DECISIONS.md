# Phase 1 Migration Decisions

## Status

Phase-1 engineering decisions are being frozen before code migration.

These classifications refer to the role of inherited RC-RGD-IMU components
inside CrossLayer-IMU-Resilience.

They do not imply that the full previous scientific method is inherited.

## Classification vocabulary

### REUSE

The scientific/engineering semantics are suitable substantially unchanged.

The new repository may retain the implementation with provenance and namespace
changes only where required.

### ADAPT

The implementation contains useful validated logic but must be separated from
old project-specific assumptions or interfaces.

### REFERENCE_ONLY

Useful precedent or comparator.

It must not become part of the proposed CrossLayer method automatically.

### REJECT

The component is inappropriate for the intended role or would contaminate the
scientific experiment.

---

## Decisions

| Inherited component | Decision | CrossLayer role |
|---|---|---|
| `src/imu_reliability/baseline/date2025_cnn400.py` | ADAPT | Primary protected-baseline candidate |
| `src/imu_reliability/baseline/historical_decision.py` | ADAPT | Historical decision-semantic reference and optional faithful mode |
| Historical DATE checkpoint | REUSE IF SHA-256 VERIFIED | Candidate baseline weights |
| `docs/protected_baseline_v1.md` | REFERENCE_ONLY | Historical provenance evidence |
| `experiments/00_baseline/*` | REFERENCE_ONLY | Reconstruction and lineage evidence |
| `models/architectures/baseline_cnn.py` | REFERENCE_ONLY | Generic 1D-CNN comparator/generalization |
| `models/architectures/ds_cnn.py` | REFERENCE_ONLY | Efficient comparator/generalization |
| `models/architectures/tcn.py` | REFERENCE_ONLY | Temporal-model comparator/generalization |
| `experiments/03_baselines/train_1dcnn_baseline.py` | REFERENCE_ONLY | Generic HAR training precedent |
| Existing ONNX export machinery | ADAPT | New protected-model export/parity pipeline |
| Existing portable C infrastructure | ADAPT | Embedded engineering scaffold |
| Existing reliability runtime decision logic | REFERENCE_ONLY | Old-method comparator/engineering precedent |
| Existing OOD threshold logic | REJECT for clean baseline | Would contaminate unprotected baseline |
| Existing sensor reliability/fusion logic | REJECT for clean baseline | Proposed baseline must remain unprotected |
| Existing integrity state machine | REFERENCE_ONLY | CrossLayer supervisor must be redesigned after fault characterization |

---

## Primary-baseline decision

The working primary protected-task candidate is:

`DATE2025_CNN_400MS_RECONSTRUCTED`

Reason:

- directly represents the target pre-impact fall-detection workload
- preserves the historical 400-ms sensing window
- preserves task-specific channel preprocessing
- is small enough for embedded analysis
- exposes internal layers suitable for compute-fault injection
- does not itself contain the new CrossLayer protection mechanism

The candidate remains **unfrozen** until checkpoint and executable baseline
verification are complete.

---

## Historical decision semantics

The inherited deployment-style decision uses the historical confidence rule:

`Falling iff P(Falling) > 0.9`

This rule must not be silently mixed with ordinary argmax classification.

CrossLayer evaluation will keep these semantics explicit.

During Phase 2 we will determine which decision rule is the primary frozen
safety decision and which is retained only for historical comparability.

No threshold will be chosen using final held-out CrossLayer test results.

---

## Embedded-runtime decision

The previous C runtime is not copied as the new CrossLayer supervisor.

It already contains previous-project integrity and OOD semantics.

Only its engineering patterns and parity-testing discipline may be adapted.

The CrossLayer runtime supervisor will be built only after:

1. sensor-fault characterization
2. compute-fault characterization
3. combined-fault characterization

This prevents the solution from being predetermined by the inherited method.

---

## Quantization/export decision

The existing ONNX export/parity infrastructure is useful but tightly coupled
to the earlier reliability project.

It will therefore be adapted into a task-only baseline export pipeline.

The initial migrated baseline exporter must:

- export only the protected task model
- contain no integrity model
- contain no OOD threshold
- contain no cross-layer supervisor
- verify source/checkpoint hashes
- verify fixed tensor shapes
- verify numerical parity

INT8 quantization comes later under a separately frozen Phase-2 protocol.

---

## Phase-1 boundary

Phase 1 selects and reconstructs the engineering baseline.

Phase 1 does not:

- tune a protection mechanism
- inject final study faults
- optimize thresholds
- perform final safety evaluation
- claim embedded hardware measurements
