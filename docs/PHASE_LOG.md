# Phase Log

## 2026-09-30 — Phase 0 COMPLETE

### Goal

Establish the scientific and engineering foundation for
CrossLayer-IMU-Resilience before importing inherited scientific code.

### Scientific state frozen in this phase

The project studies cross-layer resilience under four explicitly separated
operating regimes:

- C0 — clean sensing and clean computation
- CS — sensor fault only
- CC — compute fault only
- CSC — combined sensor and compute faults

The protected task is initially pre-impact IMU fall detection.

The primary dataset candidate is UNIVRFall.

The protected baseline has deliberately not yet been selected.

The final sensor-fault taxonomy has deliberately not yet been frozen.

The final compute-fault taxonomy has deliberately not yet been frozen.

The final MCU target has deliberately not yet been frozen.

No publication-facing experimental performance claim exists at this stage.

### Repository foundation added

- `README.md`
- `MASTER_PROJECT_CONTEXT.md`
- `AGENTS.md`
- `docs/RESEARCH_PLAN.md`
- `docs/EXPERIMENT_CONTRACT.md`
- `docs/BASELINE_SELECTION_PROTOCOL.md`
- `docs/CLAIM_LEDGER.md`
- `docs/NOVELTY_BOUNDARY.md`
- `docs/FAULT_TAXONOMY_CANDIDATE.md`
- `configs/project.yaml`
- `configs/datasets/local_paths.example.yaml`
- `src/crosslayer_resilience/`
- `scripts/project_doctor.py`
- repository contract tests
- requirements placeholders
- package metadata
- embedded/experiment/model/manifest directory contracts

### Local validation evidence

Environment observed during validation:

- Python 3.13.12
- Git 2.34.1

Validation results:

- project doctor: PASS
- repository contract tests: 8/8 PASS
- Git whitespace check: PASS
- forbidden generated/data paths tracked: none

### Scientific safeguards established

The repository now explicitly requires:

- split before stochastic fault injection
- no final-test tuning
- independent treatment of sensor and compute faults
- same-backbone causal comparisons
- evidence-tier separation
- no hardware claims from software injection
- frozen protocols before final confirmation
- provenance for publication-facing experiments
- distinction between hypotheses and confirmed evidence

### Phase-0 status

COMPLETE

### Next phase

Phase 1 — inherited-code audit and baseline reconstruction.

The next work must inspect the available implementations in:

- RC-RGD-IMU
- IMU_Reliability
- earlier pre-impact fall-detection code where relevant
- embedded/runtime infrastructure
- selected simulator infrastructure where relevant

Each candidate inherited component must be classified as:

- REUSE
- ADAPT
- REFERENCE_ONLY
- REJECT

No scientific code should be copied merely because it existed in a previous
project.

### Phase-1 primary objective

Identify the exact reusable implementation needed to reconstruct a clean,
compact, MCU-deployable protected task baseline without inheriting the
previous reliability-aware protection mechanism.

Initial baseline candidates remain:

- compact 1D CNN
- DS-CNN
- compact TCN

---

## 2026-09-30 — Phase 1A initiated

### Objective

Audit existing workstation research projects before importing any inherited
scientific implementation.

### Audit principle

The audit is read-only with respect to source projects.

No old model, dataset pipeline, runtime mechanism or experiment logic is
accepted merely because it exists.

Candidate inherited components will later be classified as:

- REUSE
- ADAPT
- REFERENCE_ONLY
- REJECT

### Primary questions

1. Where is the exact compact fall-detection model used in previous work?
2. Which clean baseline 1D CNN implementation is best suited for reuse?
3. Which DS-CNN and TCN implementations are available as baseline candidates?
4. Which data/window/training infrastructure is reusable?
5. Which quantization/export path already exists?
6. Which embedded implementation already exists?
7. Which fault-injection utilities can be adapted?
8. Which reliability-aware code must remain outside the clean protected
   baseline?
9. Which files are duplicates across projects?
10. What exact Git/file provenance accompanies every candidate?

### Status

AUDIT_IN_PROGRESS

---

## 2026-09-30 — Phase 1B initiated

### Objective

Perform deep provenance and dependency auditing of the candidate protected
baseline before any scientific implementation is copied.

### New evidence motivating this audit

The clean RC-RGD-IMU project contains two distinct baseline concepts:

1. generic compact HAR baselines
2. a reconstructed DATE-2025 400-ms pre-impact fall-detection CNN

The latter is directly aligned with the primary CrossLayer task and therefore
requires explicit evaluation as the leading protected-baseline candidate.

### Phase-1B questions

- Is the DATE-2025 reconstruction identical between the dirty development tree
  and clean published tree?
- Does the recovered checkpoint still exist locally?
- Can its checkpoint SHA-256 be independently verified?
- Are preprocessing and decision semantics recoverable?
- What baseline scripts depend on historical artifacts?
- What quantization/export infrastructure is reusable?
- What embedded runtime infrastructure is reusable?
- Which generic CNN/DS-CNN/TCN models should remain comparison/generalization
  candidates?

### Status

PROVENANCE_AUDIT_IN_PROGRESS

---

## 2026-09-30 — Phase 1C checkpoint and migration audit

### Purpose

Resolve the historical DATE checkpoint location and freeze component-level
migration decisions before code copying.

### Working baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

Status:

NOT YET FROZEN

### Migration principle

Only the clean protected task baseline and required engineering
infrastructure may be migrated.

Previous reliability, OOD and integrity mechanisms cannot be silently
included in the unprotected baseline.

### Next step

If the checkpoint anchor is verified:

- migrate baseline architecture
- migrate explicit historical decision semantics
- create CrossLayer-native model manifest
- add architecture/shape/parameter tests
- add checkpoint checksum verification
- build task-only inference smoke test

If the checkpoint cannot be located:

- retain the architecture as a reconstructed candidate
- do not claim recovery of historical weights
- move to a reproducible retraining/reconstruction protocol in Phase 2

---

## 2026-09-30 — Phase 1D controlled baseline migration

### Objective

Migrate the DATE-2025 reconstructed protected task workload into the
CrossLayer namespace without inheriting the prior reliability mechanism.

### Migrated candidate

`DATE2025_CNN_400MS_RECONSTRUCTED`

### Provenance

Historical checkpoint:

`/mnt/hdd16T/protechto/checkpoints/CNN/400ms/2025-02-25_12_24_47/best-checkpoint.ckpt`

Historical checkpoint SHA-256:

`ee7c0079bfb8555bff45c3077cc24eaa4373c57729045d92a831a1d7a3ea9bb1`

Historical source repository:

`muhammadtoqeerali/RC-RGD-IMU`

Clean source commit:

`e1db7880e17a5632bc4ce238125923f986e85519`

### Scientific boundary

The migrated baseline contains only:

- historical IMU preprocessing
- task CNN
- optional penultimate task feature
- explicit historical decision semantics
- ordinary argmax decision semantics

It does not contain:

- sensor-integrity protection
- OOD rejection
- reliability fusion
- compute-integrity protection
- runtime supervisor
- recovery logic

### Freeze status

The architecture/weight candidate is migrated for validation.

The Phase-2 baseline is still NOT FROZEN.

Dataset protocol, primary decision semantics, quantization and final reference
metrics remain Phase-2 decisions.


### Phase-1D validation implementation note

The first local Phase-1D validation attempt reached successful checkpoint,
trusted-loader and strict state-dict loading checks, then stopped in the
new CrossLayer canonical-state hashing helper.

Cause:

A zero-dimensional integer BatchNorm state entry could not be directly viewed
as `torch.uint8`.

Resolution:

The hasher now flattens each contiguous tensor before the byte-wise view.
A regression test explicitly covers scalar integer state entries.

This was a provenance-hashing implementation defect only. It did not indicate
a model, checkpoint or source-provenance mismatch.

---

## 2026-09-30 — Phase 1 COMPLETE

### Phase

Inherited-code audit and protected-baseline reconstruction.

### Final primary candidate

`DATE2025_CNN_400MS_RECONSTRUCTED`

### Historical checkpoint

Verified.

SHA-256:

`ee7c0079bfb8555bff45c3077cc24eaa4373c57729045d92a831a1d7a3ea9bb1`

### Migration validation

- parameter count: 63,173
- deterministic parity vectors: 68
- maximum absolute logit difference: 0.0
- maximum absolute feature difference: 0.0
- historical decision parity: PASS
- normalized state reload parity: PASS
- baseline contamination check: PASS
- generated binary state artifact: Git-ignored

Canonical tensor-state SHA-256:

`c98987476536320191f8875316cf8caeeb0b3f2edc51d65be7ac7d2eaea03124`

### Scientific conclusion

A faithful task-only representation of the historical pre-impact
fall-detection workload now exists in the CrossLayer namespace.

No inherited OOD, sensor-integrity, reliability-fusion or runtime-supervisor
logic is part of the clean protected baseline.

### Baseline freeze state

NOT FROZEN.

### Next phase

Phase 2 — baseline reproduction, selection, export, quantization and freeze.
