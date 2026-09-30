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

---

## 2026-09-30 — Phase 2A initiated

### Objective

Establish the executable protected-baseline environment and determine export
and quantization feasibility before freezing Phase-2 baseline identity.

### Methodological boundary

Phase 2A does not use final CrossLayer fault-study data.

Dataset-level accuracy, F1 and safety metrics are not frozen here because the
formal dataset/event/split/timing protocol is still pending.

Phase 2A focuses on:

- task-state identity
- executable environment
- model structure
- task decision interfaces
- host-side deterministic reference behavior
- export capability
- quantization capability
- model storage and operation structure

### Baseline candidate

`DATE2025_CNN_400MS_RECONSTRUCTED`

Status:

NOT FROZEN

### Phase-2A validation implementation note

The first Phase-2A execution-environment audit stopped before model inference
because Python treated `torch` as a local variable inside `main()`.

Cause:

Function-local imports of `torch.ao.quantization` and
`torch.ao.quantization.quantize_fx` created a local `torch` binding, making an
earlier `torch.inference_mode()` reference invalid.

Resolution:

Quantization capability discovery now uses `importlib.import_module(...)`.

This was an audit-script scope defect only.

It did not alter or invalidate the protected model, checkpoint, migrated state
or Phase-1 parity evidence.

---

## 2026-09-30 — Phase 2B initiated

### Objective

Create and validate a task-only FP32 ONNX representation of the protected
baseline.

### Baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

### Scope

Phase 2B validates:

- protected task-state identity
- fixed input/output tensor contract
- FP32 ONNX graph validity
- PyTorch versus ONNX Runtime numerical parity
- decision parity
- deterministic artifact checksum
- exported graph operator inventory

### Scientific boundary

The exported graph contains the task model only.

It must contain no:

- OOD logic
- integrity logic
- reliability fusion
- recovery mechanism
- CrossLayer supervisor

### Quantization boundary

Final INT8 calibration is intentionally not performed in Phase 2B.

Representative calibration data must be selected only after the formal data
split/calibration protocol is frozen.

### Phase-2B validation implementation note

The first FP32 ONNX parity attempt successfully produced the fixed-shape ONNX
graph, then the parity harness incorrectly supplied all 324 numerical vectors
as one batch.

The deployment graph intentionally accepts `[1,40,9]`.

Resolution:

Parity vectors are now executed serially through the fixed batch-1 ONNX graph
and concatenated only after inference.

The export contract was not changed to dynamic batching.

This preserves the intended single-window embedded deployment semantics.

### Phase-2B FP32 parity hardening

The FP32 ONNX parity criterion was made explicit after the initial validation
showed a larger absolute error on seeded high-amplitude numerical stress
vectors.

Structured deterministic vectors use an absolute-error criterion.

Seeded stress vectors use the explicit combined absolute-plus-relative
criterion.

All vectors must preserve both task decision interfaces.

Repeat-export determinism is also recorded.

These synthetic stress inputs remain numerical checks only and are not
dataset-performance evidence.

---

## 2026-09-30 — Phase 2C initiated

### Objective

Audit the installed quantization toolchain and ONNX operator support before
using representative calibration data.

### Scientific boundary

Phase 2C does not perform final static INT8 calibration.

No project dataset sample is used.

No validation, held-out test, external-test or fault-injected sample may be
used to choose quantization parameters in this phase.

### Primary quantization direction

The deployment quantization candidate is static post-training quantization.

This preserves the validated historical task weights while allowing activation
ranges to be calibrated later from an explicitly frozen clean calibration
partition.

Dynamic quantization may be exercised only as an engineering toolchain smoke
test.

It is not the final deployment candidate and is not performance evidence.

### Phase-2C result

The installed quantization toolchain and graph-level operator support were
audited without project dataset samples.

Static post-training quantization is retained as the primary deployment
quantization candidate.

Final activation calibration remains deferred until the clean representative
calibration partition is frozen.

Dynamic quantization, when available, is engineering smoke evidence only and
is not the deployment baseline.

Quantizer selection may not use final-test or fault-robustness outcomes.

---

## 2026-09-30 — Phase 2 COMPLETE

### Frozen reference

`DATE2025_CNN_400MS_RECONSTRUCTED`

### Freeze scope

FP32 protected reference baseline.

### Primary decision

Historical strict `P(Falling) > 0.9`.

### FP32 deployment representation

Fixed batch-1 ONNX representation frozen by SHA-256.

### Numerical validation

- 68 structured vectors
- 256 seeded numerical stress vectors
- zero elementwise tolerance violations
- zero argmax decision changes
- zero historical decision changes
- repeated export byte-identical
- repeated runtime outputs bit-identical

### Quantization

Static PTQ protocol frozen.

Final calibrated INT8 artifact deferred until Phase 3 freezes the permitted
clean calibration partition.

Dynamic quantization remains nonblocking engineering smoke evidence only.

### Dependency amendment

Static INT8 realization will occur after Phase-3 calibration-partition freeze
and before any fault characterization.

### Next phase

Phase 3 — dataset, event, split, timing and calibration-partition freeze.

---

## 2026-09-30 — Phase 3A initiated

### Objective

Perform a read-only provenance and structure audit of historical and candidate
pre-impact fall datasets before defining any new subject/event split.

### Scientific boundary

Phase 3A does not:

- create new dataset partitions
- move or copy raw data
- relabel samples
- regenerate windows
- choose calibration subjects
- expose final-test data to model selection
- perform quantization calibration
- inject faults

Historical split logic is recorded as provenance only.

No historical split is automatically accepted as the new CrossLayer split.

---

## 2026-09-30 — Phase 3B initiated

### Objective

Recover and independently verify the subject partition that produced the
historical protected DATE-2025 workload.

### Rationale

The protected model is already trained.

Creating an arbitrary new subject split could place historical training
subjects into a nominal new test set and create subject leakage.

Therefore the historical subject partition must be recovered before any
CrossLayer evaluation partition can be frozen.

### Scientific boundary

Phase 3B:

- does not generate a new random split
- does not change subject membership
- does not inspect task predictions
- does not perform INT8 calibration
- does not inject faults
- does not use final-test outcomes

The historical split remains a candidate until its lineage and counts are
independently verified.

### Phase-3B result

The historical DATE-2025 subject split was recovered and independently
reconciled against the complete historical protected 400-ms combined tree.

The reconstruction exactly matches the preserved trial and window totals.

Candidate CrossLayer role mapping:

- historical train -> development
- historical validation -> calibration
- historical test -> held-out confirmation
- subjects 999 and 1000 -> augmentation-only

There is no subject overlap among the three protected partitions.

No trial in the historical protected tree is unassigned.

The split remains a verified candidate until Phase-3 label/event/timing
lineage is complete.

---

## 2026-09-30 — Phase 3C initiated

### Objective

Audit task labels, protected-trial structure, raw-file pairing and acquisition
timing provenance before freezing event and timing semantics.

### Scientific boundary

Phase 3C is read-only.

It does not:

- change the recovered subject split
- create new windows
- relabel trials
- evaluate model predictions
- inspect fault-study outcomes
- perform INT8 calibration
- inject faults
- define detector thresholds

Processed window labels are not automatically interpreted as physical
fall-onset or impact timestamps.

Pre-impact timing claims remain blocked until the required event-time
provenance is demonstrated.

### Phase-3C first-attempt implementation correction

The first Phase-3C audit stopped because a generic CSV-header detector reported
that the legacy local `UniVrFallOriginalDataset` tree did not expose a
recognized timestamp field.

This is not interpreted as evidence that the UniVRFall dataset lacks temporal
metadata.

The official UniVRFall release documents:

- 100-Hz laboratory IMU streams
- `TimeStamp(s)`
- `FrameCounter`
- subject-specific Excel annotation files
- fall-onset frames
- fall-impact frames
- video-synchronized frame-level ground truth
- subject-independent 5-fold cross-validation

Therefore the failed assertion was an audit-assumption defect.

Phase-3C is repaired by distinguishing:

1. the public/current dataset specification,
2. legacy local historical representations,
3. the actual raw/annotation files used by the protected historical pipeline.

No dataset content or split was modified by the failed audit.

### Phase-3C repaired result

External dataset documentation clarified the temporal-label boundary.

UniVRFall provides video-synchronized fall-onset and fall-impact frames and
documents a subject-independent five-fold protocol.

KFall likewise provides fall-onset and fall-impact frame annotations.

Therefore physical pre-impact timing is potentially recoverable and must not
be dismissed merely because the processed `segments.npy` arrays omit
acquisition metadata.

The protected historical checkpoint still retains its recovered historical
train/validation/test split as checkpoint provenance.

The five-fold protocol is treated separately as a fold-specific
training/evaluation protocol.

A single frozen checkpoint cannot be reused as five independently trained
cross-validation models.

The next step is to freeze the exact five-fold membership and map annotation
event frames onto protected historical trials.

---

## 2026-09-30 — Phase 3D initiated

### Objective

Identify the authoritative subject-independent five-fold implementation used
by the historical Protechto training workflow.

The audit must establish:

- exact KFold source code
- source-code checksum
- subject enumeration rule
- number of folds
- shuffle policy
- random seed
- train/test fold construction
- validation-subject construction inside each training fold
- exact 400-ms dataset root used by candidate five-fold runs
- surviving fold-specific checkpoints and their run grouping

### Important separation

The recovered 47/6/14 split remains provenance for the already-trained frozen
DATE-2025 checkpoint.

Five-fold evaluation is a separate repeated-training protocol.

A single checkpoint cannot be presented as five-fold cross-validation.

### Timing issue retained

The local historical UniVR annotation tree contains 574 rows with both onset
and impact, while the public dataset summary reports 573 fall events.

This discrepancy remains open and must be reconciled before event-time freeze.

### Scientific boundary

Phase 3D is read-only.

It does not:

- generate new folds
- retrain models
- modify subject membership
- choose model hyperparameters
- open held-out predictions
- perform quantization calibration
- inject faults

### Phase-3D result

The five-fold audit was narrowed from the broad repository search to the
historical Protechto implementation and the 400-ms CNN checkpoint lineage.

The authoritative KFold and validation-split calls are now recorded by source
path, line, checksum and AST-extracted arguments.

Candidate complete fold-specific 400-ms checkpoint runs are recorded
separately from the frozen single-checkpoint baseline.

No new folds were generated.

The one-event UniVR annotation discrepancy remains open:

- public dataset summary: 573 fall events
- local historical onset+impact rows: 574

This difference must be reconciled before event-time and physical lead-time
freeze.

---

## 2026-09-30 — Phase 3E initiated

### Objective

Resolve the workstation lineage of:

- historical 300-ms fold checkpoints
- historical 400-ms fold checkpoints
- their fold-specific grouping
- their dataset/window roots
- the recent Falling Simulator dataset implementation
- the exact subject-enumeration semantics used by KFoldDataloader

### Baseline policy

The 400-ms DATE historical workload remains the primary CrossLayer reference.

The 300-ms checkpoint family is retained as a temporal-resolution sensitivity
and generalization comparator.

The 300-ms family does not replace the frozen 400-ms reference.

### Falling Simulator policy

The recent Falling Simulator may become the preferred dataset-lineage and
physical-fault-calibration source only after its UniVR and KFall trial identity
is reconciled with the protected historical representation.

### Scientific boundary

Phase 3E is read-only.

It does not:

- create replacement folds
- retrain checkpoints
- alter subject membership
- open held-out predictions
- perform INT8 calibration
- inject faults

### Phase-3E result

The workstation checkpoint estate was audited beyond the narrow checkpoint
roots used in Phase 3D.

Both 300-ms and 400-ms fold-specific checkpoint families are treated as
historical evidence.

The 400-ms workload remains the primary frozen CrossLayer baseline.

The 300-ms family is reserved for temporal-resolution sensitivity and
generalization analysis.

The recent Falling Simulator is audited as a preferred dataset-lineage
candidate because it is reported to use the correct UniVR and KFall source
datasets.

No simulator dataset is promoted to authoritative CrossLayer status until its
trial identities are reconciled with the protected historical representation.

Exact historical KFold reconstruction also depends on the recovered subject
enumeration semantics.

---

## 2026-09-30 — Phase 3F initiated

### Objective

Reconstruct the exact candidate historical five-fold subject memberships and
bind existing fold checkpoint families to their training lineage.

Also locate the recent Falling Simulator / physics-based digital-twin source
that uses the correct UniVRFall and KFall datasets.

### Phase-3E correction

`KFoldDataloader.py` does not use uncontrolled filesystem subject order.

It uses:

`sorted(os.listdir(root_directory))`

after excluding augmentation-only subjects.

Therefore the historical KFold population is deterministic for a known
dataset root.

Sorting is lexicographic because subject identifiers are strings.

### Baseline policy

The frozen 400-ms DATE checkpoint remains the primary CrossLayer workload.

Existing 400-ms five-fold families currently identified are:

- CNNSplit
- LSTM
- ResNet

They are architecture comparators.

They are not five-fold replicas of the frozen DATE CNN.

The 300-ms checkpoint estate remains secondary temporal-resolution evidence.

### Scientific boundary

Phase 3F does not:

- retrain models
- create a novel fold policy
- choose checkpoints from performance
- inspect held-out predictions
- perform INT8 calibration
- inject faults

### Phase-3F result

Historical five-fold reconstruction now uses the exact loader semantics:

- lexicographically sorted subject directory names
- augmentation subjects excluded first
- five-fold KFold
- shuffle enabled
- random state 42
- validation subjects sampled from each training-fold index set
- validation fraction 0.2
- validation random state 42

The three complete 400-ms five-fold checkpoint families remain architecture
comparators rather than replicas of the frozen DATE CNN.

The recent simulator/digital-twin source was searched under broader naming
patterns so that dataset lineage does not depend on a repository literally
being named `Falling_simulator`.

---

## 2026-09-30 — Phase 3G initiated

### Objective

Bind the frozen DATE-2025 400-ms baseline to its authoritative real-data
subject population and distinguish its historical split from the prospective
five-fold replication protocol.

Also audit the curated subject-fold and fall-event assets recovered from the
recent HR_LR_Fallings / risk-data workflow.

### Current strongest lineage hypothesis

The candidate root:

`/mnt/hdd16T/protechto/data/back/UniVrFall_KFall/segments/400ms_50ov_npseg_filt_binary`

contains 69 subject directories.

After excluding augmentation-only subjects 999 and 1000, 67 subjects remain.

Those 67 subjects equal the historical protected population.

Its reconstructed five-fold fold-1 test set appears identical to the
historical 14-subject frozen-checkpoint test partition.

The inner train/validation partition differs and must remain explicitly
separate.

### Protocol separation

Historical frozen baseline:

- train = 47 subjects
- validation = 6 subjects
- test = 14 subjects
- augmentation-only = 999 and 1000

Prospective five-fold replication:

- same real subject population if lineage is confirmed
- fold-specific train/validation/test partitions
- each subject appears in outer test exactly once
- fold-specific training is required

### Scientific boundary

Phase 3G does not:

- retrain any model
- change the frozen DATE checkpoint
- create a new split from performance
- inspect held-out predictions
- perform INT8 calibration
- inject faults

### Phase-3G result

The frozen DATE-2025 400-ms baseline population is now compared directly
against the 67-subject combined real-data root.

The historical 14-subject held-out test partition is checked against every
reconstructed five-fold outer test partition.

The historical inner train-validation split remains distinct from the current
five-fold inner split and is not rewritten.

Curated subject-fold and fall-event assets from the recent risk/simulator
workflow are inventoried for the subsequent timing-lineage freeze.

---

## 2026-09-30 — Phase-3 methodological pivot

New information about the historical preprocessing/training workflow shows
that a mature 300-ms subject-independent k-fold pipeline exists.

The project therefore distinguishes two roles.

### Historical reference

`DATE2025_CNN_400MS_RECONSTRUCTED`

This remains frozen as a historical 400-ms reference and reproducibility
anchor.

Nothing from Phase 2 is discarded.

### Prospective primary experimental protocol

Candidate primary window:

300 ms

Candidate primary population:

UniVRFall + KFall real subjects only.

OnField will not be mixed into the primary held-out subject folds unless later
evidence provides a compelling reason.

It is instead treated as augmentation / external robustness evidence.

### Subject identity rule

Manual numeric renaming is not accepted as the new scientific identifier.

Canonical IDs will encode dataset identity explicitly, for example:

`UNIVR_<subject>`

`KFALL_<subject>`

`ONFIELD_<subject>`

### Remaining unknowns

Before the 300-ms protocol can be frozen we must verify:

- exact meaning of the preprocessing `-o` argument
- exact historical window stride
- exact 300-ms dataset roots
- exact subject identities
- exact five-fold memberships
- exact CNN checkpoint lineage
- whether existing 300-ms checkpoints were trained on UniVR, KFall,
  combined data, or mixed OnField data
- event-onset and impact mapping

---

## 2026-09-30 — Phase 3H initiated

### Protocol clarification

The intended historical and prospective segmentation uses 50 percent overlap.

The later 95-percent-overlap experiments are excluded from the primary
CrossLayer protocol.

At 100 Hz:

- 300 ms = 30 samples
- 50 percent overlap = 15-sample stride = 150 ms
- 400 ms = 40 samples
- 50 percent overlap = 20-sample stride = 200 ms

### Primary protocol candidate

Window:

300 ms

Overlap:

50 percent

Sampling:

100 Hz

Primary subject population:

UniVRFall + KFall only

Expected subjects:

- UniVRFall = 29
- KFall = 32
- total = 61

OnField is excluded from the primary held-out fold population.

OnField may later be used as training augmentation or external robustness
evidence under an explicitly frozen policy.

### 400-ms role

400 ms with the same 50-percent overlap remains:

- historical DATE baseline reference
- later controlled window-duration sensitivity condition

It is not the primary prospective CrossLayer dataset.

### 95-percent overlap

95-percent-overlap datasets and checkpoints are excluded from main protocol
selection.

They must not be mixed with the 50-percent-overlap evidence.

### Objective

Verify whether the existing 61-subject merged 300-ms root is an exact
file-level copy of the individually generated UniVRFall and KFall 300-ms
50-percent-overlap roots.

No regeneration is required if the roots match exactly.

### Phase-3H result

The primary prospective CrossLayer dataset is now evaluated under the intended
300-ms / 50-percent-overlap protocol.

The clean 61-subject UniVRFall + KFall root is checked against the separately
generated source-dataset segment roots at file level.

OnField is excluded from the primary fold population.

The later 95-percent-overlap experiments are explicitly excluded from primary
protocol lineage.

The 400-ms condition retains the same 50-percent-overlap rule and remains a
historical reference / controlled temporal-window comparison.

---

## 2026-09-30 — Phase 3I initiated

### Objective

Determine the correct scientific role of the OnField recordings.

The project does not discard OnField.

The current candidate design is:

- UniVRFall + KFall: 61-subject primary five-fold benchmark
- OnField: independent field-domain / generalization evidence

Historical augmentation recordings must be separated from independent
field-validation subjects before this role is frozen.

### Questions

Phase 3I determines:

- exact OnField processed subject IDs
- exact value of `DATA_AUGMENTATION_SUBJECTS`
- whether 999 and 1000 are augmentation-only storage identities
- whether exactly ten additional OnField subjects remain
- per-subject Activity/Falling label composition
- number of trials and windows
- whether the 300-ms and 400-ms OnField populations agree
- raw OnField source-tree structure
- whether the historical helper injects OnField data into training

### Scientific policy candidate

Independent OnField subjects must not be used for:

- model architecture selection
- hyperparameter selection
- fault-severity selection
- protection-threshold selection
- supervisor tuning
- INT8 calibration
- recovery tuning

They may be opened only after the corresponding protocol is frozen.

This preserves OnField as genuine external-domain evidence.

### Phase-3I result

OnField is retained as a first-class part of the CrossLayer study.

Its role is intentionally separated from the primary UniVRFall + KFall
five-fold benchmark.

The independent OnField subjects are reserved for field-domain /
generalization evidence.

Historical OnField augmentation storage identities are audited separately so
that augmentation data cannot contaminate the independent field evaluation.

### Phase-3I policy correction

The OnField cohort has been clarified from acquisition history.

Exactly ten OnField cases are retained:

- storage IDs 1001 through 1010

They contain Activity data only.

Storage IDs 999 and 1000 had known data-quality issues and are permanently
excluded from the prospective CrossLayer project.

Although historical code once used 999 and 1000 as augmentation storage IDs,
that historical behavior is not inherited by the new CrossLayer protocol.

The rejected cases are kept in provenance documentation only.

The ten retained OnField cases are reserved as an independent Activity-only
field-domain evaluation cohort.

They are not used for training, validation, INT8 calibration, model selection,
fault parameter selection, protection tuning or recovery tuning.

Primary fall-positive evaluation remains on the 61-subject UniVRFall + KFall
population.

---

## 2026-09-30 — Phase 3J initiated

### Objective

Freeze the exact subject-independent five-fold membership for the prospective
primary 300-ms CrossLayer experiment.

### Frozen population candidate

Primary data:

- UniVRFall: 29 subjects
- KFall: 32 subjects
- total: 61 subjects

Windowing:

- 300 ms
- 100 Hz
- 30 samples/window
- 50 percent overlap
- 15-sample stride
- 150-ms decision stride

### Fold-generation algorithm

The historical subject-level KFold policy is reused prospectively:

- subjects ordered by lexicographically sorted storage directory name
- KFold n_splits = 5
- shuffle = True
- random_state = 42
- outer test subjects come directly from KFold
- validation is selected only from the outer non-test indices
- validation test_size = 0.2
- validation random_state = 42

The folds are generated from the 61-subject UniVRFall + KFall root only.

OnField is not part of five-fold generation.

### Partition roles

Training partition:

- model fitting
- permitted source for a later deterministic INT8 calibration subset
- permitted source for development-only fault calibration

Validation partition:

- architecture/hyperparameter checks after architecture policy is frozen
- decision-threshold selection
- monitor/protection calibration
- recovery/supervisor calibration

Outer-test partition:

- locked evaluation only
- never used for model selection
- never used for fault severity selection
- never used for INT8 calibration
- never used for monitor/protection tuning

OnField retained subjects 1001 through 1010:

- external Activity-only evaluation
- unavailable to all training/tuning/calibration decisions

Rejected OnField IDs 999 and 1000:

- unavailable everywhere
- provenance only

### Scientific boundary

Phase 3J does not train models, evaluate predictions, calibrate INT8, inject
faults, select thresholds or inspect outer-test outcomes.

### Phase-3J result

The prospective primary 300-ms subject-independent five-fold membership is
now frozen.

The primary population contains exactly 61 dataset-qualified subjects.

Every subject appears in outer test exactly once.

No train/validation/test subject overlap exists within any fold.

OnField does not participate in primary fold generation.

INT8 calibration is restricted to future deterministic samples drawn from the
training partition of each fold.

Outer-test and retained OnField data remain unavailable to all model,
quantization, fault and protection tuning decisions.

---

## 2026-09-30 — Phase 3K initiated

### Objective

Recover and audit fall-event annotation lineage for the frozen 61-subject
300-ms primary population.

The audit maps:

processed subject / task / trial

to:

official UniVRFall or KFall fall-onset and fall-impact annotation.

### Frozen inputs

Primary processed data:

- UniVRFall: 29 subjects
- KFall: 32 subjects
- total: 61 subjects
- trials: 6,309
- window: 300 ms
- overlap: 50 percent
- stride: 150 ms

Five-fold membership is already frozen and is not modified in Phase 3K.

### Questions

Phase 3K determines:

- exact annotation workbook set used for UniVRFall
- exact annotation workbook set used for KFall
- annotation column semantics
- number of onset/impact rows
- processed trial mapping coverage
- duplicate annotation trial keys
- annotation rows without a processed trial
- processed Falling trials without an annotation
- onset >= impact anomalies
- UniVR 574-versus-573 event discrepancy
- availability of existing curated fall-event indexes
- whether annotation values are already sensor-sample indices or still require
  synchronization before physical lead-time computation

### Scientific boundary

Phase 3K does not:

- change five-fold membership
- train a model
- run model predictions
- inspect outer-test model outcomes
- inspect OnField model outcomes
- perform INT8 calibration
- inject faults
- select protection thresholds

Physical lead-time semantics remain unfrozen until this audit is complete.

### Phase-3K result

The official/local UniVRFall and KFall annotation workbooks were audited
against the exact 6,309 processed 300-ms primary trial identities.

Complete onset/impact rows, duplicate trial keys, unmatched annotation rows,
processed Falling trials without annotations and onset-impact ordering are
recorded explicitly.

Existing curated fall-event index assets are also inventoried.

Annotation frame values are not yet automatically converted to 100-Hz sensor
time.

Physical lead-time reporting remains blocked until Phase 3L establishes the
annotation-to-sensor synchronization semantics.

---

## 2026-09-30 — Phase 3L initiated

### Motivation

Phase 3K successfully audited the local annotation workbooks, but the direct
workbook-row-to-processed-trial join is not authoritative.

In particular:

- KFall has far more fall trial instances than workbook rows.
- workbook rows can encode multiple trial instances
- reducing every workbook row to one integer trial ID loses event instances
- therefore the direct Phase-3K workbook reconciliation must not be used as
  the final event mapping

Phase 3K remains useful as a provenance audit.

### Phase-3L source candidate

The existing curated event asset:

`FALL_EVENT_INDEX_COMBINED_LABELED.csv`

contains expanded event-level records with:

- dataset identity
- subject identity
- task
- trial
- source sensor file
- source annotation workbook
- fall-start frame
- impact frame
- zero-based fall-start position
- zero-based impact position
- sampling rate

Phase 3L independently verifies this event index against:

1. the frozen 6,309-trial primary 300-ms processed root
2. the processed Activity/Falling trial labels
3. local oriented sensor CSV files
4. local annotation workbooks
5. frame/position arithmetic
6. KFall FrameCounter evidence

### Scientific boundary

No model is executed.

No prediction is opened.

No fault is injected.

No INT8 calibration is performed.

No physical lead-time claim is frozen unless synchronization evidence is
sufficient.

### Phase-3L result

The expanded curated fall-event index was audited independently against the
frozen 300-ms primary processed population and the local oriented sensor
files.

The earlier Phase-3K scalar workbook-row join is explicitly non-authoritative.

The curated event index is the event-lineage candidate for the prospective
CrossLayer timing analysis.

KFall FrameCounter correspondence is checked directly against raw/oriented
sensor rows.

UniVR physical timing remains separately guarded until original timestamp
synchronization is qualified.

No model outcomes were inspected.

---

## 2026-09-30 — Phase 3M initiated

### Motivation

Phase 3L established that the curated event index contains 2,919 unique
annotated fall events.

It also exposed two dataset-specific conventions which must be normalized
before event-to-processed-trial reconciliation.

1. UniVR event-index subjects use an event namespace such as `1009` for
   source subject `SA09`, whereas the frozen primary processed root uses
   storage subject `9`.

2. KFall event-index subjects use the same protected-tree namespace as the
   processed root, such as `106` for source subject `SA06`.

Phase 3L also found that all 573 UniVR event records use a different
frame-to-position convention from KFall. This must be audited against the
original UniVR recordings rather than force-fitting the KFall convention.

### Protocol eligibility

A 300-ms model window at 100 Hz requires 30 samples.

The frozen safety deadline is 150 ms before impact.

An annotated fall event is capable of yielding a complete post-onset,
pre-deadline 300-ms Falling window only when:

`impact_position - onset_position - deadline_samples >= window_samples`

The Phase-3L discrepancy `KFALL_106_T27_R05` has a 410-ms onset-to-impact
duration. Therefore only 260 ms remains after enforcing the 150-ms
pre-impact deadline. This is shorter than a 300-ms window.

Phase 3M tests whether protocol eligibility exactly explains the
2,919 annotated events versus 2,918 processed fall-positive trials.

### Scientific boundary

No model is trained.

No model prediction is executed.

No outer-test outcome is opened.

No OnField outcome is opened.

No INT8 calibration is performed.

No fault is injected.

The five-fold membership remains unchanged.

### Phase-3M-R result

The Phase-3M strict equivalence between historical Falling labels and a
complete post-onset 300-ms window ending 150 ms before impact is rejected.

The canonical mapping of 2,919 annotated events to processed trials remains
valid.

All 2,919 annotated events correspond to processed trial identities.

Exactly one annotated fall event has no Falling-labelled processed window:

`KFALL_106_T27_R05`

This case remains explicit and is not silently removed.

Historical binary-label semantics and later safety-deadline timing semantics
are now represented as separate protocol layers.

The legacy UniVR raw logger parser was repaired to recognize `time[ms]`.

No model execution occurred.

---

## 2026-09-30 — Phase 3N initiated

Phase 3N closes the prospective data protocol.

It does not attempt to reconstruct historical labels from event annotations.

Two independent evidence layers are retained.

### Classification layer

The inherited 300-ms `labels.npy` files are immutable classification
ground truth.

The event-onset/impact annotations are not used to rewrite these labels.

### Safety-time layer

The curated fall-event index and oriented sensor FrameCounter are used for
event-time evaluation.

The legacy UniVR `time[ms]` stream is retained as provenance but is not the
authoritative event clock because Phase 3M-R found extensive duplicate
timestamp steps and representation-length differences.

### Causal decision timestamp

For 300-ms windows at 100 Hz with 50 percent overlap:

- window samples = 30
- stride samples = 15
- window `j` starts at sample `15*j`
- its final observed sample is `15*j + 29`

Offline sensor-only lead time is measured from that final observed sample to
the impact frame.

Measured runtime latency is subtracted later when MCU timing exists.

### Quantization calibration

Static PTQ calibration may use only the training subjects of the corresponding
frozen fold.

Validation, outer test and OnField are prohibited calibration sources.

Phase 3N freezes a deterministic calibration-window identity set but does not
perform quantization.

---

## 2026-09-30 — Phase 3N-R initiated

The first Phase-3N close attempt correctly validated the event FrameCounter
clock, the historical windowing source and deterministic training-only INT8
calibration identities.

Its whole-trial window-count check was rejected.

The recovered historical preprocessing source demonstrates that annotated
fall trials are segmented piecewise.

For a fall trial:

1. the pre-onset region is windowed independently as Activity
2. the onset-to-impact region is windowed independently as Falling
3. the fall-region window grid therefore restarts at fall onset

Consequently a processed fall-trial window index cannot universally be mapped
to raw sample position using `15 * window_index`.

Phase 3N-R reconstructs the exact piecewise raw-sample position of every
stored window and freezes that mapping.

`KFALL_106_T27_R05` remains an explicit historical annotation/processed-label
discordance. It is not deleted and its labels are not rewritten.

---

## 2026-09-30 — Phase 3N-R2 initiated

Phase 3N-R recovered the correct piecewise preprocessing structure but used
the curated zero-based event position as the historical Python slice index.

That assumption was rejected.

The historical source uses the annotation values `start_fall_frame` and
`end_fall_frame` directly in Python slicing.

This creates a dataset-specific consequence.

For UniVR, the curated event position and annotation frame use the same
index convention.

For KFall, the curated position is annotation frame minus one. Therefore
historical preprocessing used a split index one sample after the curated
zero-based event position.

Phase 3N-R2 reconstructs stored windows using the historical annotation frame
values themselves.

The fallback whole-trial preprocessing branch is also audited for the sole
current annotation/processed-label discordance `KFALL_106_T27_R05`.

### Phase-3N-R2 result

Phase 3 is complete.

The historical preprocessing coordinate system is now source-qualified.

For UniVR, annotation frame and curated zero-based event position coincide.

For KFall, annotation frame equals curated zero-based event position plus one.

Historical preprocessing used annotation frame values directly as Python
slice indices.

Using that source-faithful convention reconstructs 2,918 annotated event
trials exactly through the historical Activity-plus-Falling piecewise route.

The sole remaining event `KFALL_106_T27_R05` exactly matches the historical
whole-trial fallback geometry and retains its frozen Activity labels.

No trial was deleted or relabeled.

All 273,830 stored primary windows now have a qualified historical window
route and raw-sample index mapping.

The physical event clock remains the oriented FrameCounter.

Five deterministic training-only INT8 calibration identity sets remain
frozen.

No prospective model training, quantization execution, fault injection or
held-out model evaluation occurred in Phase 3.
