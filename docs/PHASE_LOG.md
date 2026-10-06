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

---

## 2026-09-30 — Phase 4A initiated

Phase 3 is frozen and committed.

Phase 4A determines whether an already-trained 300-ms CNN can serve as the
prospective baseline under the frozen 61-subject protocol.

Existing models are evaluated only for provenance compatibility.

Model accuracy is not used for selection.

Outer-test predictions are not executed.

OnField predictions are not executed.

A historical checkpoint is reusable only when its lineage is compatible with:

- CNN model family
- 300-ms window
- 50-percent overlap
- 61-subject UniVRFall + KFall primary population
- no OnField subjects in the five-fold population
- frozen five-fold subject membership
- compatible preprocessing and feature representation
- complete five-fold checkpoint family
- checkpoint selection independent of outer-test outcomes

If exact compatibility cannot be demonstrated, Phase 4 will train a new
prospective five-fold CNN family using the frozen protocol.

### Phase-4B prospective 300-ms CNN training protocol

The prospective FP32 CNN baseline uses the frozen Phase-3 300-ms,
50%-overlap primary dataset and frozen five-fold subject manifest.

The primary population remains the 29 UniVR + 32 KFall subjects.

Retained OnField subjects 1001-1010 remain external validation only.

OnField subjects are excluded from primary training, validation, outer-test,
checkpoint selection, threshold selection, and calibration.

Storage IDs 999 and 1000 remain permanently excluded.

Historical CNN checkpoints are not used to select the prospective baseline
because Phase 4A found no provenance-confirmed reusable historical run.

No model execution occurred during protocol freezing.

INT8 calibration remains deferred until the FP32 baseline is frozen.

Fault injection remains deferred until after the FP32 baseline is frozen.

## Phase 4H — Sensor-FI held-out outer interpretation v1

- Status: `FROZEN_INTERPRETATION_OF_HELD_OUT_OUTER_RESULTS`.
- Reporting source: immutable Phase-4H outer reporting v1; execution remained `EXECUTION_INTERPRETABLE`.
- Mandatory family×severity records: 4,536 with deterministic 10,000-replicate subject bootstrap for overall, UniVR, and KFall strata.
- Direction inventory across all strata: 2,677 `DEGRADATION_SUPPORTED`, 268 `IMPROVEMENT_SUPPORTED`, 1,555 `NO_DIRECTIONAL_CONCLUSION`, 36 `UNRESOLVED`.
- Overall 61-subject inventory: 928 degradation, 96 improvement, 476 no-direction, 12 unresolved.
- L3 `axis_loss` is a severe repeated held-out failure mode; precision and median sensor-lead inference are unresolved there because valid detection coverage disappears.
- Clean FP32 and qualified PTQ references remain descriptively close; no equivalence claim is made because no equivalence test was predeclared.
- UniVR and KFall remain separately reported because direction statuses differ for a nontrivial subset of fault conditions, especially timing.
- `NO_DIRECTIONAL_CONCLUSION` is not equivalence and no binary global robustness label is generated.
- Governance boundary: these held-out results may be reported/interpreted but cannot modify model weights, thresholds, operating points, fault severities, calibration, or reporting/statistical rules.
- Interpretation manifest SHA256: `a5daf46f342b669cc461d0f6d66d2d933349b2739f57e700c3e7d7073719025b`.

## Phase 4I — OnField Activity-only external interpretation v1

- Status: `FROZEN_ACTIVITY_ONLY_EXTERNAL_INTERPRETATION`.
- Independent retained external cohort: 10 subjects, 16 trials, 1,023,337 Activity windows, zero Falling windows.
- All 15 seed×fold checkpoints per model variant and all three frozen validation-selected operating points were retained.
- Prospective FP32:
  - balanced: Activity specificity 0.999010; 18.328044 false triggers/Activity hour.
  - low_false_alarm: Activity specificity 0.999005; 4.044666 false triggers/Activity hour.
  - timely_150ms: Activity specificity 0.995899; 70.148137 false triggers/Activity hour.
- Qualified static PTQ v7:
  - balanced: Activity specificity 0.998950; 19.446624 false triggers/Activity hour.
  - low_false_alarm: Activity specificity 0.998947; 4.289012 false triggers/Activity hour.
  - timely_150ms: Activity specificity 0.995657; 74.611724 false triggers/Activity hour.
- Uncertainty is the predeclared deterministic 10,000-replicate subject bootstrap over the 10 retained external subjects.
- The external false-alarm burden is descriptively operating-point dependent, but OnField cannot select an operating point; all three remain frozen and reportable.
- FP32 and PTQ are descriptively close at corresponding operating points, but no model-difference or equivalence test was predeclared; no superiority/equivalence claim is made.
- The Activity-only cohort cannot support fall recall, event recall, missed-fall rate, lead time, recovery, two-class balanced accuracy, or external fall-detection-effectiveness claims.
- No checkpoint, threshold, operating point, model weighting, calibration rule, or scientific parameter may be changed from these results.
- Interpretation manifest SHA256: `23f57b6ebc265e04f9ab607a9cb19d37ad80f19229748c5cf741d63f5c32f658`.

## 2026-10-03 — Phase 4 complete

Phase 4 — Sensor Fault Engine is complete.

Completed scope:

- prospective 300-ms FP32 baseline frozen under the Phase-3 protocol;
- qualified static PTQ v7 mixed-precision comparator across all 15
  seed×fold checkpoints;
- deterministic P0 sensor-FI protocol with 12 fault families;
- immutable 61-subject held-out sensor-FI outer execution;
- frozen subject-level outer reporting and interpretation without retuning;
- independent retained OnField Activity-only external execution and
  interpretation without model, threshold, checkpoint, or operating-point
  selection.

Held-out sensor-FI execution scale:

- 320,616 subject-condition rows;
- 42,766,632 unique fault instances;
- 479,750,160 model-window evaluations;
- 4,536 family×severity inferential reporting records.

External OnField execution scale:

- 10 retained subjects;
- 16 trials;
- 1,023,337 Activity windows;
- zero Falling windows;
- 30,700,110 model-window evaluations;
- 92,100,330 threshold applications.

Scientific boundary:

- held-out and external results did not feed back into tuning;
- `NO_DIRECTIONAL_CONCLUSION` is not equivalence;
- no FP32/PTQ equivalence claim is made;
- no global binary robustness label is generated;
- no MCU or physical-hardware claim is made.

Roadmap boundary:

- Phase 4 = Sensor Fault Engine;
- Phase 5 = Compute Fault Engine;
- Phase 6 = Cross-Layer Vulnerability Characterization over C0, CS, CC,
  and CSC.

Compute fault injection and the combined sensor+compute regime are therefore
not claimed as Phase-4 deliverables and remain future governed work.

Phase-4 completion manifest SHA256:
`ca5db70026d94ea4831ceb233012ff4f299f34608e90c06654a0f5303e7fee9c`.

## 2026-10-05 — Phase 5A compute-FI representation protocol frozen

Phase 5 Compute Fault Engine work has begun with a frozen P0
representation/identity contract before any compute-fault performance run.

Frozen initial families:

- INT8 persistent-weight single-bit flip;
- quantized-activation single-bit flip;
- quantized-buffer single-bit flip;
- FP32-activation single-bit flip;
- FP32-buffer single-bit flip.

Structural boundary:

- the qualified mixed-precision PTQ v7 model has one genuine INT8 persistent
  weight tensor: `conv_2.0.weight`;
- INT8 and FP32 corruption remain explicitly separate;
- quantization metadata corruption is excluded from v1;
- multiplicity is frozen at one for v1;
- transient and within-trial persistent semantics are defined;
- deterministic SHA-256 fault identity/replay is required.

Scientific boundary:

- no model inference was executed to create the freeze;
- no dataset was read;
- outer test and OnField remain prohibited;
- no CC/CSC performance result exists;
- evidence remains P0 software injection only;
- no physical MCU/SEU/register/cache equivalence is claimed.

Outer replicate cardinality and the final outer compute-FI sampling plan remain
unfrozen until exact operator implementation and development-safe
qualification pass.

Protocol SHA256: `279d5c9ebd865ae0d147ec62d0cd04d2ffb5d58d9c80544ee6c913f9a1964bd5`.

Freeze manifest SHA256: `c99b3d98dc965f4ff9b3cfa2d624f159f656fbe508a5d8cc4e67652d77a21a8e`.

## 2026-10-05 — Phase 5A bit-exact compute-FI operators qualified

The frozen Phase-5A P0 compute-FI representation contract now has a qualified
bit-exact operator implementation.

Qualification covered:

- 2048 exhaustive signed-INT8 scalar bit cases;
- 128 synthetic qint8 tensor element/bit cases;
- 512 synthetic FP32 tensor element/bit cases;
- frozen PTQ v7 `conv_2.0.weight` payload checks;
- representative frozen FP32 checkpoint tensor checks;
- exact transient and within-trial persistent schedule semantics;
- exact restoration after the same bit is flipped twice;
- preservation of quantized metadata;
- preservation of FP32 NaN/Inf outcomes without sanitization;
- no cross-trial state leakage.

Scientific boundary:

- no dataset was read;
- no model forward pass was executed;
- outer test and OnField were not read;
- no CC or CSC task-performance result was generated;
- evidence remains P0 software-level fault semantics only.

The outer compute-FI sampling plan remains unfrozen.

Qualification manifest SHA256: `c2ce24636f868c8f7d2a9c6250bc9a0364a7d423e49c80ea9fd5bdb61a414a9e`.

Qualification result SHA256: `1f33743a6fd882dce8a844617d026a027dd9c435db56eb016f0f251448abccbd`.

## 2026-10-05 — Phase 5A quantized activation/buffer dtype qualification corrected

A pre-commit Phase-5A audit found a representation-specific implementation
gap in the first compute-FI operator qualification:

- the persistent PTQ v7 quantized weight `conv_2.0.weight` is
  `torch.qint8` with `torch.int8` payload storage;
- the actual PTQ v7 quantized activation and intermediate-buffer tensors are
  `torch.quint8` with `torch.uint8` payload storage.

The frozen fault taxonomy and eight-bit XOR semantics were already generic
enough for the activation/buffer representation and therefore did not change.

The operator implementation and qualification were corrected to:

- retain exact qint8/int8 weight corruption semantics;
- add exact quint8/uint8 activation and buffer corruption semantics;
- preserve quantization metadata for both storage classes;
- retain exact single-element/single-bit mutation and double-flip restoration;
- use quint8 for transient/persistent quantized activation schedule
  qualification and clean-parent restoration checks.

No dataset, outer test or OnField evidence was used. No faulted task inference
was performed. The outer compute-FI sampling plan remains unfrozen.

Corrected qualification manifest SHA256: `078da4d254ca91f468c842641ecc80f96878046407119f2362909d9157383fae`.

Corrected qualification result SHA256: `31fae824bbfbb1142dc7ffb9a3c91d8d60e8b670cc006b06c32a757abff4b12b`.

## 2026-10-05 — Phase 5B paired synthetic compute-FI execution harness qualified

The Phase-5 compute-fault execution layer now has a qualified paired
clean/faulted synthetic harness.

Qualification demonstrated:

- exact frozen FP32 checkpoint reconstruction;
- eager PTQ v7 reconstruction with strict frozen-state loading;
- bitwise clean agreement between eager PTQ and frozen TorchScript;
- execution of all five frozen Phase-5A fault families;
- qint8/int8 persistent-weight injection;
- quint8/uint8 quantized activation and buffer injection;
- FP32 activation/buffer injection in FP32 and mixed-precision PTQ paths;
- exact single-element/single-bit mutation records;
- deterministic fault replay;
- transient one-inference restoration;
- persistent weight onset semantics;
- clean reset at a new trial/session;
- unchanged source checkpoint/PTQ/TorchScript artifact hashes.

Scientific boundary:

- qualification used synthetic 30x9 inputs only;
- no dataset partition was read;
- no outer test or OnField evidence was read;
- no CC task-performance or CSC result was generated;
- the outer compute-FI sampling plan remains unfrozen;
- evidence remains P0 software FI only.

Qualification manifest SHA256: `8293fc6b983ac40359c4dd7d358f79f9b00a8cece6c9a9649a07aa723cceadb1`.

Qualification result SHA256: `ee699b11f9e4d37fadab798b5818fadd2f0f347404cf13705e558b222c5859dd`.

## 2026-10-05 — Phase 5C real training-calibration paired compute-FI runner qualified

The Phase-5 compute-FI paired runner was qualified on real frozen
training-calibration inputs under a protocol frozen before execution.

Qualification used all 15 prospective model-estate members and exactly 105
predeclared paired cases: 30 FP32 cases and 75 PTQ-v7 cases.

Each fold used its first lexicographically ordered identity from the frozen
4096-window training-only calibration selection, with the same fold identity
reused across seeds 42, 123, and 2025.

Every case used the predeclared smoke coordinates:

- element index 0;
- bit position 0;
- inference index 0;
- transient-one-inference persistence;
- multiplicity 1;
- replicate index 0.

These coordinates do not constitute the outer compute-FI sampling plan.

All 15 eager PTQ reconstructions matched their frozen TorchScript references
bit-for-bit on the selected real training-calibration input.

All 105 fault IDs were unique, and active mutations changed exactly the
selected payload element/bit with qint8, quint8, or FP32 representation
semantics preserved as applicable.

The first execution attempt stopped only during result metadata serialization:
the frozen calibration identity label is textual (`Activity`/`Falling`) and
was incorrectly converted to integer. The runner was repaired to preserve that
label string. No model execution semantics, partition, fault coordinate,
105-case matrix, protocol choice, or acceptance gate changed. The exact same
frozen qualification was rerun.

Source FP32 checkpoint, PTQ-state, and TorchScript hashes remained unchanged.

No raw logits or probabilities were persisted. No accuracy, recall,
specificity, threshold, timing, or other task-performance value participated
in qualification or protocol selection.

Validation, outer-test, and OnField partitions were explicitly prohibited.

No CC outer-test result or CSC result was generated.

The outer compute-FI sampling/cardinality plan remains unfrozen.

Pre-execution protocol SHA256: `1df39a0c6b0c1e16e1ea3898a6dedecaafff93cbf7204e501f087da23e81333c`.

Qualification manifest SHA256: `e915a7ce99e0810e94864ae6ad0b198c0622399d04a4a247738863710284a932`.

Qualification result SHA256: `68a8d3b32bb86c0aa822e398a07c45c2d34821ba20bc23c9da8cf6b00375ab26`.

## 2026-10-05 — Phase 5D prospective outer compute-FI protocol frozen

The prospective Phase-5 outer compute-FI protocol was frozen before any
compute-fault outer-test model execution.

The frozen outer population is 61 subjects, 6,309 trials, and 273,830 windows.

All 15 checkpoint members remain required.

The five qualified compute-fault families remain required.

The target estate contains:

- 10 FP32 targets common to FP32 and PTQ;
- 4 PTQ-only targets;
- 14 model-independent sampling target strata;
- 24 variant-specific target strata.

Sampling uses one deterministic multiplicity-1, replicate-0 bit fault per
eligible target and parent.

Both frozen temporal modes are retained:

- transient one-inference faults, parented by outer windows;
- persistent-from-onset faults, parented by outer trials.

SHA-256 selects target element, bit, and persistent onset. Global RNG state is
not used.

Checkpoint seed and model variant are excluded from sampling-coordinate
identity. Therefore logical coordinates are reused across all three checkpoint
seeds and, for common FP32 targets, across FP32 and PTQ. Model-specific
FaultIdentity records remain distinct.

Frozen cardinality:

- transient model-independent sampling IDs: 3,833,620;
- persistent model-independent sampling IDs: 88,326;
- total model-independent sampling IDs: 3,921,946;
- transient model-specific fault IDs: 19,715,760;
- persistent model-specific fault IDs: 454,248;
- total model-specific fault IDs: 20,170,008;
- clean C0 window forwards: 1,642,980.

Persistent model-forward exposure will be derived exactly from frozen
trial-window counts and deterministic onset hashes in a separate execution-plan
qualification before any outer model forward. That calculation may not alter
sampling.

Transient instances are alternate-world single-inference pairs and may not be
concatenated into artificial faulted sequences.

Persistent instances are coherent trial sequences with clean reset at trial
boundaries.

Non-finite outputs are never sanitized or silently coerced to Activity/Falling
and must be explicitly reported.

Primary uncertainty remains subject-level; overlapping windows are not
independent uncertainty units. All three checkpoint seeds are equal-weighted
within subject.

Every predeclared target remains reportable. Outer outcomes cannot select
targets, bits, families, metrics, checkpoint seeds, or persistence modes.

Phase 5 will produce C0/CC only. CSC remains deferred to Phase 6.

No outer arrays, outer predictions, outer model forwards, OnField evidence, or
hardware/physical claims were used for this freeze.

Protocol SHA256: `395689f0eaa1a3c3c3aa8c6fbdeadf52f7d49f73f8b15a3bda12daeb88d3c0dd`.

Sampler SHA256: `6ad87aa55e9a553abc8f96cad8d7b8fbc72076acfa86e47a76e6226dd4f9c6e9`.

Freeze manifest SHA256: `eda67d2e063517e422151dfb1b5cef50b07c4c2f7ae7344ed8b07ac24e27c214`.

## 2026-10-05 — Phase 5D parent-bound outer execution identity technical repair

A read-only post-freeze identity audit found that the qualified Phase-5A
`fault_id` does not contain subject/task/trial/window parent identity.

The Phase-5A identifier therefore remains the immutable mutation-specification
ID, but it is not globally unique across outer parents.

A technical identity repair added a separate parent-bound `outer_instance_id`.

The identity hierarchy is now explicit:

- `sampling_instance_id`: parent-bound, model-independent sampling identity;
- `phase5a_fault_id`: unchanged model/checkpoint mutation-specification ID;
- `outer_instance_id`: parent-bound model/checkpoint execution-record ID.

`outer_instance_id` hashes the frozen sampling-instance ID, unchanged Phase-5A
fault ID, model variant, and checkpoint seed.

This technical repair changed no:

- sampling payload;
- element coordinate;
- bit coordinate;
- persistent onset;
- target;
- fault family;
- persistence mode;
- multiplicity;
- replicate count;
- reporting endpoint;
- frozen cardinality.

The planned parent-bound outer execution-record count remains **20,170,008**.

The existing `model_specific_fault_ids` cardinality field is retained for
provenance but is explicitly deprecated as a claim that Phase-5A `fault_id`
itself is unique across outer parents.

No outer data, predictions, model execution, fault injection, OnField evidence,
CC result, or CSC result was used for the repair.

Updated protocol SHA256: `0ba30c9d336433748378f8d3eed1ed45b673fb9690250b1bd3bbbc79f88c9c36`.

Updated sampler SHA256: `8886b74043ef9ba32f2cb8e999d318c340c239d818caad53f430594546d2417d`.

Updated freeze manifest SHA256: `2330244d8a20658b1804e006574d410b191faf451cefb59481a62f7419518695`.

## 2026-10-05 — Phase 5E deterministic outer compute-FI execution plan qualified

The already-frozen Phase-5D prospective outer compute-FI protocol was converted
into a deterministic, metadata-only, resumable execution plan.

The planner reads only frozen JSON metadata, filesystem parent names, and
`segments.npy` NPY headers.

It does not open label payloads, call `np.load`, create a memmap, materialize
signals, load a model, execute a model, inject a fault, or read outer
predictions.

The exact previously-qualified bindings were reproduced:

- 61 outer subjects;
- 6,309 outer trials;
- 273,830 outer windows;
- trial-window inventory SHA256
  `ffdd772db741c53ba452054e00ed8d9cba76412ed11ac4801edf4f5694f994d3`;
- subject inventory SHA256
  `139aa8efcf613770300f6cafbe04eb690d0ac20384bf914b72d5cc8dcd110208`;
- persistent-onset binding SHA256
  `4d17d427c2f0af5e30405c641b3a5f50d5eaf3c8aeae872a90be6a11876705d2`.

The immutable fault-shard unit is:

`subject × model_variant × checkpoint_seed × persistence`.

This produces exactly **732 fault shards**.

Clean C0 inference uses **366 subject × variant × seed caches** shared between
transient and persistent shards.

Frozen execution cardinality remains:

- transient outer instances: 19,715,760;
- persistent outer instances: 454,248;
- total outer instances: 20,170,008;
- clean model-window evaluations: 1,642,980;
- transient faulted model-window evaluations:
  19,715,760;
- persistent faulted model-window evaluations:
  10,083,060;
- total faulted model-window evaluations:
  29,798,820;
- total clean + faulted model-window evaluations:
  31,441,800.

Persistent exposure was derived from the already-frozen per-trial window counts
and deterministic onset hashes. No sampling decision changed.

Every fault shard references exactly one clean cache.

`outer_instance_id` remains the required primary execution-record key, while
Phase-5A `fault_id` remains unchanged mutation provenance.

No CC result or CSC result exists yet.

Planner SHA256: `a7da66a136e217f18731cc02346beff5482f8752bbfeb24256634485848d452a`.

Execution-plan SHA256: `95aecd14b4aa8dce70492f0c4d6d50a8bf5a8dafb2ec962aff9c033b8472bb54`.

Qualification manifest SHA256: `eb69004c52b389b0ad32e5de626a94de41683b761f523216a4290d5ef0772c95`.

## 2026-10-05 — Phase 5F compute-FI executor/resume core qualified

The atomic execution-record and resume core for the frozen Phase-5 compute-FI
outer executor was qualified without executing any outer shard.

The core reuses the already-qualified Phase-5 bit mutation and paired-execution
primitives and adds no new fault semantics.

Atomic artifact semantics are now qualified:

- write to `<artifact_id>.partial`;
- hash every declared output;
- atomically rename the temporary directory to the final directory;
- write `_SUCCESS.json` last;
- reuse only when artifact ID, artifact kind, frozen plan hash, executor hash,
  and every output SHA-256 match;
- reject corrupted successful output;
- reject partial final directories;
- reject partial temporary directories;
- require explicit recomputation for partial artifacts.

Synthetic resume tests passed for first write, exact reuse, corrupted-output
rejection, wrong-plan rejection, wrong-executor rejection, partial-final
rejection, partial-temp rejection, and recomputation cleanup.

Training-calibration integration was qualified on three representation classes:

- FP32 activation;
- PTQ quint8 activation;
- PTQ qint8 weight.

The reconstructed eager PTQ clean output remained bitwise equal to the frozen
TorchScript reference.

All fixture execution records had unique parent-bound `outer_instance_id`
values.

No outer payload, outer model forward, outer fault execution, outer prediction,
OnField evidence, CC result, or CSC result was used or generated.

The full outer `execute-shard` path is intentionally still disabled.

Executor SHA256: `77ee323f71a5dbf4c54cc454a5911849aa45c5d4bdcc6d63bb908d8cece363d0`.

Qualification manifest SHA256: `a77274b5f2f502342249977c175852e918dbc187612116e210b84b0f4b3de01b`.

## 2026-10-05 — Phase 5G outer compute-FI 732-shard dry run qualified

The frozen Phase-5E execution plan was wired to a Phase-5G outer shard
controller while retaining an explicit hard execution gate.

`outer_execution_enabled = false`.

The controller dry-run validated all **732 fault shards** and **366 clean
caches** without model execution.

The dry run SHA-256 verified all 45 frozen model artifacts:

- 15 FP32 checkpoints;
- 15 PTQ state dictionaries;
- 15 PTQ TorchScript artifacts.

The Phase-5E metadata/header-only plan regenerated exactly, including the
trial inventory, subject inventory, persistent onset binding, clean-cache
records, and every shard record.

Frozen execution totals remain:

- transient outer instances: 19,715,760;
- persistent outer instances: 454,248;
- total outer instances: 20,170,008;
- clean model-window evaluations: 1,642,980;
- transient faulted evaluations:
  19,715,760;
- persistent faulted evaluations:
  10,083,060;
- total faulted evaluations:
  29,798,820;
- total clean + faulted evaluations:
  31,441,800.

The `execute-shard` command was explicitly probed and correctly rejected before
dataset/model access because the execution gate is false.

The dry run used only filesystem metadata, `segments.npy` headers, JSON
metadata, and SHA-256 artifact reads.

It did not read segment payloads, open labels, call `np.load`, call
`torch.load`, load a model, execute a model, invoke a fault operator, execute an
outer shard, read an outer prediction, use OnField, or generate CC/CSC results.

Phase-5D sampling/identity and the Phase-5E plan remain unchanged.

Executor SHA256: `7112f1ae046bac730729a26c5a858dc04ca257d29d8dedf1f9930607bd061cb2`.

Dry-run result SHA256: `bb4ad8244b6d4c0fd821dc40b5036eef5c82d17b7f291ba313035ff328abdbd0`.

Qualification manifest SHA256: `7ea089c9465782ebe5baa45628b2720b02cec07a9c714ed2e83ba2db23b97fd5`.

## 2026-10-05 — Phase 5H complete compute-FI shard-style execution qualified

Complete transient and persistent shard-style compute-FI mechanics were
qualified on frozen training-calibration fixtures while the outer execution
gate remained false.

The first execution attempt stopped before the first fault-shard execution
because the qualifier's `shard_id` expression omitted one concatenation
operator. Before that failure, permitted training-calibration payloads had been
read, models and clean forwards had been exercised, and two clean-cache
artifacts had been written. No fault shard had executed.

The qualifier repair added only the missing concatenation operator. The frozen
Phase-5H qualification configuration did not change. The failed result estate
was discarded before the complete rerun.

The complete rerun qualified:

- 2 shared clean-cache artifacts;
- 14 fault-shard fixture artifacts;
- 7 transient shard fixtures;
- 7 persistent shard fixtures;
- 42 unique parent-bound execution records;
- 70 per-window effect records.

Transient active mask:

`True, True, True, True, True`.

Persistent active mask:

`False, False, True, True, True`.

All five representation families were covered.

Common FP32 target coordinates were identical across FP32 and PTQ variants.

The PTQ qint8 persistent-weight session was reset, and a distinct
training-calibration trial proved post-reset bitwise equality to the clean PTQ
model. Cross-trial leakage is rejected.

All 16 final atomic fixture artifacts passed success-marker and output-hash
reuse validation.

A later regression failure was documentation-only: the literal phrase
`omitted one concatenation operator` was split across two Markdown source
lines. Only that line wrap was repaired. No execution artifact changed and no
fixture rerun was performed.

The next regression attempt failed only because the shell omitted the project
`PYTHONPATH`, preventing import of `models.CNN`. Restoring the established
Phase-5 Python import path required no source, result, protocol, or fixture
change.

No outer-test payload, outer model forward, outer fault execution, frozen outer
shard execution, outer prediction, OnField evidence, CC result, or CSC result
was used or generated.

The outer execution gate remains false.

Runtime SHA256: `b3bf60c2637a03fa9ddb364637171d7f5d81e5b367586b82bd94972d3e1f91e7`.

Qualifier SHA256: `af6c072496be63257cce46ea5b2f6beea8807d2c395232b7b0e66b9cf3ab05cc`.

Qualification result SHA256: `3f680e2ed4d1bfbf0654ddc8395a1d233e40891efd8e9d21e4f3ca104189ebd0`.

Qualification manifest SHA256: `f028e6e12268e55f6863cfa70e98cf4b765b4120bcb928e29658b51cd1f64a94`.

## 2026-10-05 — Phase 5I final pre-outer no-selection audit and execution authorization frozen

The final Phase-5 pre-outer governance audit passed without reading or
executing outer data.

No prior Phase-5 manifest records completed outer execution, a CC result, or a
CSC result.

The frozen sampler and execution sources contain no executable logic for
best/worst seed selection, fold selection, shard ranking, outer-metric-driven
selection, threshold retuning, or adaptive fault resampling.

The deterministic sampler contains no global Python/NumPy/Torch random
sampling path and remains SHA-256-derived.

The prospective estate remains complete and unchanged:

- 61 subjects;
- all 5 folds;
- seeds 42, 123, and 2025;
- FP32 and PTQ-v7;
- transient and persistent modes;
- 732 fault shards;
- 366 clean caches;
- 20,170,008 outer execution identities;
- 31,441,800 total clean + faulted model-window evaluations.

The frozen operating points remain:

- balanced;
- low_false_alarm;
- timely_150ms.

No single best seed/fold/member may replace the all-15 estate.

Phase 5I freezes prospective authorization for exactly this estate:

`outer_execution_authorized = true`.

No outer shard is executed by this freeze.

The prior Phase-5G dry-run gate remains false and unchanged. A new execution
configuration must bind the Phase-5I gate SHA before outer execution can start.

Outer outcomes remain prohibited from changing sampling, targets, elements,
bits, onsets, persistence, multiplicity, thresholds, or operating points.

OnField remains unavailable for tuning.

Phase 5 remains C0/CC only; CSC remains Phase 6.

No physical or MCU equivalence claim is made.

No-selection audit SHA256: `ffe073a22a4d29a9ade54a21f437731ccf6f05428b9c03b4afb04af62f42cc4d`.

Execution gate SHA256: `ef7ab5c5dbf5959c37a7649a88ddfc7ca5a1839a7d98d0bd46b98bd6b7958fb3`.

Gate-freeze manifest SHA256: `9536dc626918e9f49dcd4b8af4f0e805747f80384d6f74a280d4c0d1e0d59c0a`.

## 2026-10-05 — Phase 5K fault-only compute-FI execution qualified

A source audit showed that the previously qualified paired execution harness
performs a clean-reference forward in addition to the requested faulted
forward.

That paired behavior is correct for qualification but is incompatible with the
frozen prospective Phase-5E forward accounting, where clean forwards are
already budgeted separately through the clean-cache estate.

Phase 5K therefore added a separate fault-only execution layer without
modifying the paired harness.

The new layer reuses the existing qualified target mappings, mutation
operators, mutation-record validation, PTQ interpreter logic, and qint8 state
mutation semantics.

It contains no embedded clean-reference forward.

Training-calibration equivalence qualification covered all five representation
families under transient and persistent semantics: 14 cases and 70
fault-only sequence executions.

For every qualification inference, the fault-only path matched the paired
harness in:

- faulted output bitwise;
- mutation record;
- fault ID;
- input SHA-256;
- active/inactive schedule.

FP32 and PTQ-weight paths were instrumented to prove one model forward per
fault-only invocation. PTQ activation/buffer execution uses exactly one FX
interpreter run per invocation.

No outer payload, outer forward, outer fault, outer prediction, CC result, or
CSC result was generated.

Phase-5D sampling/identity, the Phase-5E plan, and Phase-5I authorization
remain unchanged.

Fault-only module SHA256: `f8bb4095bbfedc69c24fa4cc6e79ccf4914e14fa502ce080a06de01e740112cb`.

Qualifier SHA256: `e8a8d0995789805446f5284a07511daae083b955d817560206f4310190eb18f2`.

Qualification result SHA256: `dda8d57589b4ffa4e0864f90e03a2a5511a1f4c2b9132096dcee1fbec38997d2`.

Qualification manifest SHA256: `120701fa970fe4b809353668d5d5eb14c8ae065a5e543992c3d5c2439c8815a7`.

## 2026-10-05 — Phase 5L hash-bound canary-only outer execution configuration frozen

A canary-only prospective outer execution configuration was frozen without
reading or executing outer payloads.

The first Phase-5L attempt stopped before writing the config because it assumed
a non-existent `model_applicability` target field.

A read-only schema probe established the frozen Phase-5D field is
`model_variants`.

The corrected binding uses:

`"fp32" in target["model_variants"]`

and yields exactly the 10 common FP32 targets already frozen in Phase 5D,
matching the frozen canary target count exactly.

No scientific protocol, sampling, identity, target, canary, or execution
semantics changed.

Exactly one fault shard is authorized:

`p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9`

Exactly one clean cache is authorized:

`p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b`.

Every other shard remains unauthorized.

Full-fleet execution remains unauthorized.

The frozen canary forward budget is:

- 2,481 clean forwards;
- 24,810 faulted forwards;
- 27,291 total forwards.

Clean forwards are separate-cache only.

Fault forwards are Phase-5K fault-only execution only.

No embedded clean-reference fault-loop forward is permitted.

No outer payload, outer model load, outer forward, outer fault execution,
prediction, CC result, or CSC result existed at freeze.

Phase-5L config SHA256: `b169e2997499dfb75927938d98d39018d01f60a93e04148d2b4fcb70f30692be`.

Phase-5L manifest SHA256: `51c2b91138766bef371fdd9c32e471b254f7f0b4203d62c90e24dc24868086fa`.

## 2026-10-05 — Phase 5M exact outer canary executor statically qualified

The exact executor for the single Phase-5L-authorized canary was implemented
without invoking its outer execution path.

The validation path hard-binds the frozen Phase-5L config SHA and validates all
frozen dependency hashes, the Phase-5E plan, checkpoint hash, 44-trial/2,481
window canary inventory, 10 FP32 target mappings, and exact forward budget.

Exactly the frozen shard-zero ID is accepted.

The second Phase-5E shard was explicitly rejected before any signal-array or
model access.

Static call-path auditing proves `validate-config` and `validate-shard` contain
no `np.load`, `torch.load`, model loader, fault runner, or execution-boundary
call.

Only the `execute-canary` path can reach the outer array loader.

The executor implements a separate 2,481-forward clean-cache pass and a
24,810-forward Phase-5K fault-only pass.

No paired runner is used in the fault loop.

The executor reuses Phase-5F atomic/resume semantics, with success markers bound
to both the frozen Phase-5E plan SHA and exact executor source SHA.

Fault records use the frozen Phase-5D sampler to rederive
`sampling_instance_id`, element, bit, transient inference index, Phase-5A
`fault_id`, and parent-bound `outer_instance_id`.

No labels, thresholds, or metrics are read or computed by the executor.

At qualification time no outer payload, model load, forward, fault execution,
prediction, CC result, or CSC result was produced.

Executor SHA256: `aebc8e6d9d89ee44eb31e8b4bb39afa1efe46f597b0dacdfd1d9d0efc9a1a645`.

Static qualification manifest SHA256: `55ef847d03a5cfac9ec55c9b7ab38358ea447a119fe1bc273ea9287f8d9914ee`.

## 2026-10-05 — Phase 5O single outer canary technically accepted

The single frozen Phase-5N outer canary was accepted on technical execution
integrity only.

Acceptance used no prediction-value interpretation and no scientific
performance outcome.

The accepted canary contains exactly:

- 1 clean-cache artifact;
- 1 fault-shard artifact;
- 44 trials;
- 2,481 windows;
- 10 FP32 targets;
- 2,481 clean forwards;
- 24,810 faulted forwards;
- 24,810 outer execution instances.

Both atomic success markers validate.

Every output-file hash bound by those success markers matches.

The JSONL files contain exactly 2,481 clean records and 24,810 fault records.

No partial or unauthorized artifact is present.

No clean-reference forward occurred inside the fault loop.

Prediction JSONL rows were not deserialized for Phase-5O acceptance.

No labels or OnField data were read.

No threshold, metric, or outcome-driven selection was computed.

No sampling, identity, target, bit, onset, persistence, threshold, or operating
point was changed.

This technical acceptance is not an aggregate CC result.

CSC remains absent.

The remaining 731 fault shards remain unexecuted and full-fleet authorization
remains false.

Technical acceptance result SHA256: `d67dba6513a62c8192137c3b19d1806db91e4b257cb17b6481ad11a86a897d72`.

Acceptance freeze manifest SHA256: `09fe4836d954f98876656ebb383c306a60c444b6225dd1f99e1c67577bf1a74f`.

## 2026-10-05 — Phase 5P prospective full-fleet completion authorization frozen

Prospective authorization to complete the exact frozen Phase-5E outer estate
was frozen after Phase-5O technical canary acceptance.

The authorization uses only canary execution integrity and does not use
prediction outcomes, labels, metrics, thresholds, OnField, or fault-effect
magnitude.

The already accepted canary is retained as satisfied estate:

- 1 fault shard;
- 1 clean cache;
- 24,810 outer execution instances;
- 27,291 model-window evaluations.

The exact remaining Phase-5E complement is authorized:

- 731 fault shards;
- 365 clean caches;
- 20,145,198 outer execution identities;
- 1,640,499 clean model-window evaluations;
- 29,774,010 faulted model-window evaluations;
- 31,414,509 remaining model-window evaluations.

No subset was selected from canary outcome.

The complete frozen estate remains exactly 732 fault shards, 366 clean caches,
20,170,008 outer execution identities, and 31,441,800 total model-window
evaluations.

Phase 5P executes no additional shard.

A new fleet executor is required before execution may continue. It must bind
the Phase-5P gate SHA, reuse the accepted canary artifacts, preserve the
Phase-5E estate exactly, and be qualified before any additional outer
execution.

No aggregate CC result or CSC result is generated by this freeze.

Phase-5P gate SHA256: `1f8d6dbd6491cbacf21c0be25053201e0f38a9ad8e987261468611d220af3719`.

Phase-5P manifest SHA256: `874bfa1fad5639c2d879439cdbd58cee75dbc32e2a40d7fc2f840f4c33a50c19`.

## 2026-10-05 — Phase 5R full-fleet outer executor qualified pre-outer

The Phase-5P-hash-bound full-fleet compute-FI executor was implemented and
qualified before any additional outer execution.

The executor supports all four frozen production execution classes: FP32
transient, FP32 persistent, PTQ-v7 transient, and PTQ-v7 persistent.

It preserves the frozen 10-target FP32 and 14-target PTQ-v7 inventories.

Persistent onset bindings and exposure counts are rederived from the frozen
Phase-5D/5E metadata before execution.

Fault execution uses only Phase-5K fault-only interfaces.

No paired fault runner or embedded clean-reference fault-loop forward is used.

PTQ qint8 weight state is reset after every execution sequence.

The accepted canary artifacts are hash-validated and reused.

Training-calibration qualification used five real windows from frozen fold-1
training-only calibration data and covered six production orchestration routes,
for exactly 30 fault-only sequence executions.

Transient active mask passed as `[True, True, True, True, True]`.

Persistent active mask passed as `[False, False, True, True, True]`.

PTQ eager clean inference matched the frozen TorchScript reference bitwise.

PTQ qint8 weight reset passed after transient and persistent routes.

A schema-only qualification repair changed one Phase-5K metadata assertion from
the non-existent key `onfield_payload_used` to the actual frozen key
`onfield_used`. The Phase-5R result field `onfield_payload_used=false` remained
unchanged.

No executor logic, protocol, qualification matrix, partition, sampling,
identity, target, persistence, gate, or outer estate changed because of that
repair.

No validation or OnField payload was used.

No additional outer-test payload, outer forward, or outer fault execution
occurred.

The outer estate remains exactly one executed canary fault shard and one clean
cache.

Full-fleet execution remains unstarted.

No aggregate CC result and no CSC result exists.

Phase-5R executor SHA256: `82424cf132a7272c9dc8a7ffd2271472b19990a209490f9dbc94bec7a7af9188`.

Phase-5R qualifier SHA256: `423686e7a79d86b14e1c97b08b9ae8b61e721ada11eee42e7b59d7e980e9d4d9`.

Phase-5R qualification result SHA256: `d02d858e4c4c292e76fc60369d5483bf2fc2f87e95138aa59848851f045120f8`.

Phase-5R qualification manifest SHA256: `76a6b922f13508a50cc616376596c083f8b0075c24faaa00284b5c8d17e4bf78`.

## 2026-10-05 — Phase 5S final full-fleet execution activation frozen

Final prospective full-fleet execution activation was frozen without executing
an additional outer shard.

The activation binds the unchanged Phase-5E plan, Phase-5P completion gate,
exact qualified Phase-5R executor, Phase-5R qualification result, and Phase-5R
qualification manifest.

The accepted canary remains reuse-only.

The exact activated complement is:

- 731 fault shards;
- 365 clean caches;
- 20,145,198 outer execution identities;
- 1,640,499 clean forwards;
- 29,774,010 faulted forwards;
- 31,414,509 remaining model-window evaluations.

After successful completion the estate must remain exactly the original
Phase-5E totals: 732 fault shards, 366 clean caches, 20,170,008 outer execution
identities, and 31,441,800 total model-window evaluations.

Activation is outcome-independent and uses no prediction values, metrics,
thresholds, labels, or OnField data.

Phase 5S performs no model load, model forward, fault execution, or additional
outer payload read.

The first complete Phase-5 regression invocation after activation freeze failed
during test collection only because the shell omitted the previously required
Protech/CrossLayer PYTHONPATH entries. No scientific artifact, activation
artifact, executor, outer payload, model state, or fault execution was changed.

The regression was resumed under the restored environment without regenerating
the Phase-5S activation artifacts.

At activation freeze the outer estate remains exactly one accepted canary fault
shard and one accepted clean cache.

No aggregate CC result or CSC result exists.

Phase-5S activation SHA256: `3c47fc42aa257897d1d4df4e96378789da1849e263f7277265d90ff35d2482c8`.

Phase-5S freeze manifest SHA256: `5153fbce714594d7440e98191fadb06ac5b51ca01995f4243f80dadb053c77d8`.

## 2026-10-06 — Phase 5U complete outer estate technically accepted

The complete frozen Phase-5E compute-fault outer estate was technically
accepted after full-fleet execution.

Accepted cardinalities are exactly:

- 732 fault shards;
- 366 clean caches;
- 1,098 atomic artifact directories;
- 20,170,008 outer execution identities;
- 1,642,980 clean records;
- 29,798,820 faulted records;
- 31,441,800 total model-window records.

All Phase-5R-produced artifacts were verified against their atomic success
markers, metadata, output hashes, and frozen Phase-5E cardinalities.

The accepted Phase-5N canary was independently reverified against its frozen
Phase-5O hashes.

Record JSONL files were read only as binary streams for SHA256 verification,
newline counting, and byte-size accounting.

Prediction JSON rows were not deserialized.

No prediction outcome, softmax value, class count, CC metric, threshold, label,
or OnField payload was inspected or used.

Post-execution regression explicitly deselects one frozen historical
filesystem-state assertion:

`tests/test_phase5r_compute_fi_outer_full_fleet_executor.py::test_outer_estate_remains_single_accepted_canary`

That Phase-5R test correctly described the pre-fleet state but is no longer a
valid assertion about the live result root after successful Phase 5T execution.

The frozen Phase-5R test and frozen Phase-5R manifest remain byte-identical and
hash-consistent; neither was rewritten to accommodate the later lifecycle
state.

No scientific sampling, identity, plan, gate, or executor changed.

No aggregate CC result and no CSC result was generated.

Phase-5U technical acceptance SHA256: `87932aebe474d515d5d43063724c8f3f0026084a6df40225e3666d83e3ae4f88`.

Phase-5U freeze manifest SHA256: `5c82fb20f83ed7479efa85d50945313904f5046dc2c11c3e950e0b30c63559da`.

## 2026-10-06 — Phase 5Z prospective CC outcome-analysis protocol frozen

The prospective compute-only (`CC`) outcome-analysis contract was frozen
before opening any Phase-5 prediction JSONL or loading any outer-test label
array.

The protocol binds:

- the exact 45 validation-selected threshold/consecutive rows;
- Activity=0 / Falling=1 class semantics;
- direct Phase-5M and Phase-5R clean/fault softmax fields;
- exact Phase-4 `>=` trigger and consecutive-run behavior;
- the frozen historical truth and FrameCounter timing semantics;
- one `outer_instance_id` per compute-fault scenario;
- clean-prefix/faulted-window reconstruction for transient and persistent
  faults;
- a paired clean parent-trial counterfactual for every compute-fault identity;
- target/family/model/persistence/operating-point stratification;
- equal checkpoint-seed weighting within subject;
- equal subject weighting;
- subject-cluster bootstrap uncertainty;
- no composite robustness scalar;
- no record-count-weighted target aggregation;
- strict integrity abort rules.

The accepted Phase-5M canary and Phase-5R fleet both expose direct
`clean_softmax_values` and `faulted_softmax_values`, so no canary-specific
probability reconstruction is needed.

The first full repository regression invocation after the Phase-5Z freeze
failed during collection only because the shell omitted the previously required
Protech/CrossLayer PYTHONPATH entries. The Phase-5Z protocol, documentation,
test, and manifest were not regenerated or changed.

Regression was resumed under the restored import environment.

At freeze time and throughout this repair:

- prediction JSONL opened: false;
- prediction JSON deserialized: false;
- outer label array loaded: false;
- threshold applied to outer prediction: false;
- CC metric computed: false;
- aggregate CC result generated: false;
- CSC result generated: false;
- model forward executed: false;
- fault execution executed: false.

Phase-5Z protocol SHA256: `3ad80af971aa80de0661b6eb622987cc08b0a72f596e097af56265bf22a3d9a4`.

Phase-5Z freeze manifest SHA256: `f92b08aaebed81f85d737e5da02cf2291b4fc46c20f8c73f4d8a79076047fd9c`.

## 2026-10-06 — Phase 5AA CC analyzer core qualified pre-outcome

The pure compute-fault CC analysis core was qualified before any prospective
Phase-5 prediction JSONL or outer label array was opened.

The qualification completed 16 synthetic/source-equivalence checks covering
the frozen threshold matrix, historical trigger and event-metric semantics,
mixed Phase-5M/Phase-5R direct softmax extraction, transient and persistent
scenario reconstruction, persistent integrity failures, paired clean
counterfactuals, degradation direction, equal seed/subject weighting,
subject-cluster bootstrap determinism, and non-finite accounting.

The first qualifier invocation stopped on an exact floating-point comparison
of `0.9 - 0.7` against decimal `0.2`. The analyzer itself was unchanged. Only
the synthetic assertion was repaired to use `math.isclose` with zero relative
tolerance and `1e-12` absolute tolerance. All 16 qualification checks then
passed.

No prospective outer payload or outer labels were used.

No threshold was applied to outer predictions.

No outer CC metric, aggregate CC result, or CSC result was generated.

The prospective CC I/O runner remains unimplemented and unqualified.

Analyzer SHA256: `74c052101b01a33b40e157727cc485d3fd8d716c4f4bdc6fd27f0d5e5398ea66`.

Qualifier SHA256: `a0b312e15fa6e7306ce0334f8c06e72c55ef6bd266e4fc6f3cb9e30a3f29ac93`.

Qualification result SHA256: `2e7edad2d3aede47ce8edce68bf58fa6b6deeac15b108b69e8ff8774b9c03c45`.

Phase-5AA manifest SHA256: `05c6c262751b766cddf8d2483ab270cca34cb446f2218d52732861ddf2f5714d`.

## 2026-10-06 — Phase 5AB prospective CC I/O runner qualified pre-outcome

The prospective compute-only CC artifact-I/O runner was implemented and
qualified using temporary synthetic filesystem fixtures only.

The qualification records 12 named checks covering clean sequence joins,
historical fallback truth, historical window-end reconstruction, label-length
validation, malformed-JSON rejection, Phase-5M exact-hash and transient
adaptation, Phase-5R hash-before-parse and persistent adaptation, risk-index
joining, stored-label normalization, and tamper rejection before parsing.

The first qualification invocation reached its final bookkeeping assertion
but expected 13 checks while exactly 12 named checks were recorded. Only that
expected count was corrected from 13 to 12. The runner remained byte-identical
and no scientific semantics changed.

The accepted prospective outer result root was not accessed.

No accepted clean or fault prediction JSONL was opened.

No accepted outer label array was loaded.

No outer prediction was deserialized or thresholded.

No outer CC metric, aggregate CC result, or CSC result was generated.

Runner SHA256: `0135d48f7e34622af49584b25b02371b42e0cc81da6adbe7b6d60ca915d96b5f`.

Qualifier SHA256: `428fe86d56cc21933c2299fd22a9040b1fea840a83c6be8c41abcce86245dc1c`.

Qualification result SHA256: `ac8371f2655a6954f4463712ba2a4163f0274b92b213dad2cd5dfd2431f3cf9f`.

Phase-5AB manifest SHA256: `5e0dbc7b66a6e85b04cabcecb8a08bc215f8e00506116356ea6b755ce1979116`.

## 2026-10-06 — Phase 5AC pre-outcome CC execution gate frozen

The pre-outcome compute-only execution gate was frozen after qualification of
the Phase-5AA analysis core and Phase-5AB artifact-I/O runner.

The gate binds the frozen Phase-5Z scientific protocol, Phase-5AA analyzer,
Phase-5AB I/O runner, Phase-5E execution plan, Phase-3 timing manifest,
accepted Phase-5O canary acceptance, and complete Phase-5U estate acceptance.

The future accepted result root and outer dataset root are recorded as
immutable path strings only.

Phase 5AC did not stat, list, or open the accepted outer result root.

No accepted prediction JSONL or outer label array was opened.

No outer prediction was deserialized or thresholded.

No CC metric, aggregate CC result, or CSC result was generated.

Phase 5AC authorizes only implementation and synthetic/static qualification of
the final end-to-end outcome executor.

A separate final activation binding that qualified executor hash is required
before the first accepted prediction or label payload may be opened.

Phase-5AC gate SHA256: `2850a03b1f07893ee77c7a52d35bd77cb14f50ca4168289f608123485b633f17`.

Phase-5AC manifest SHA256: `4282d818f2e7b6ff38435ddcce212424dcee149fec1ebeba58f32109a96e61c9`.

## 2026-10-06 — Phase 5AD final CC outcome executor qualified pre-outcome

The final end-to-end compute-only outcome executor was implemented and
qualified exclusively against temporary synthetic accepted-style estates.

Production mode binds the exact Phase-5AC gate, Phase-5Z protocol,
Phase-5AA analyzer, Phase-5AB I/O runner, Phase-5E plan, accepted outer-root
string, and accepted dataset-root string.

Qualification exercises both Phase-5R-style integrity metadata and an
accepted-canary-style externally frozen hash path, transient and persistent
fault reconstruction, all three operating points, paired-clean/faulted
metrics, paired degradation, non-finite accounting, equal checkpoint-seed
weighting inside subject, equal subject weighting, the frozen subject-cluster
bootstrap, clean-baseline deduplication across persistence, and explicit
no-imputation behavior for missing timing summaries.

No accepted outer result root was accessed, statted, or listed.

No accepted prediction JSONL or outer label array was opened.

No accepted outer prediction was deserialized or thresholded.

No accepted CC metric, aggregate accepted CC result, or CSC result was
generated.

A final one-way activation binding the exact qualified executor hash is still
required before first accepted outcome access.

Executor SHA256: `e564d4db2e0dfe3ff0aa35cf9a30061455dc995eac0b94644bdbfa29e3e7d7a5`.

Qualification result SHA256: `a726592c10310eba7abedd11ea8ad0644fc2b692e8f89af4d21b023992bff620`.

Phase-5AD manifest SHA256: `fe0207f90a28654a937b07fd22b3f8a42e36d39d68b77e78c9c227a24abc797e`.

## 2026-10-06 — Phase 5AE final one-way CC outcome execution activation frozen

The final one-way prospective compute-only outcome activation was frozen.

The activation binds the exact Phase-5AD end-to-end executor and all frozen
Phase-5Z / 5AA / 5AB / 5AC lineage, the Phase-5E estate, accepted canary and
fleet technical acceptance, exact dataset root, and exact risk-index hash.

The activation authorizes the exact qualified executor to perform the first
accepted outer prediction and label reads, apply only the frozen validation
operating points, compute C0/CC outcomes, paired degradation, and the frozen
subject-primary aggregate.

It does not authorize threshold retuning, model selection, fault resampling,
new model forwards, new fault execution, OnField use, or CSC generation.

Production must process all 732 frozen Phase-5E shards before aggregate
interpretation.

After the first accepted payload read, observed outcomes cannot change any
scientific rule.

The activation itself did not stat, list, or open the accepted outer result
root and did not load outer labels.

Activation SHA256: `f1a8d55601c5dfa766262f623e96b478c19c02d090e4e1d0cecb6b3e25a8eeb5`.

Activation manifest SHA256: `035b6c2dcfa7ccfcf4a83d17add8cef4fbdd982b64e9247191f05a4b0cf1df9a`.

## 2026-10-06 — Phase 5AF-R1 post-boundary nonfinite softmax adapter repair qualified

Phase-5AF crossed the one-way accepted-outcome boundary and then stopped before
completing any shard because the V1 analysis consumer could not decode the
frozen producer representation for non-finite softmax values.

The representation audit established that the accepted estate contains 61,752
dict-valued softmax elements, all in faulted softmax fields, all with the exact
shape `{"nonfinite":"nan"}`, and no unexpected dictionary shape.

The scientific protocol was not changed.

The historical V1 analyzer remains byte-identical.

A versioned V2 analyzer restores the Phase-5M/5R serialized IEEE token to its
original floating value before applying the already-frozen analysis semantics.

Synthetic qualification passed for finite V1/V2 equivalence, NaN/+inf/-inf
decoding, strict malformed-token rejection, frozen NaN comparator behavior,
NaN scenario reconstruction, and unchanged event-metric semantics.

Phase-5AF was not resumed by this repair step.

V2 analyzer SHA256: `0cfb60c9b32951482ffcc5be0ca14ec49c91b7d669e43eddf39fe28f26d01da5`.

Repair qualification result SHA256: `25e9a484ca8c9bd83cb936ab2d2349c7aa9d978737425fe1f83b12d1d650c2a4`.

Repair manifest SHA256: `08e4e9fd5f70d7e49db59128dd7fb95a1ef56c9ac099f96420541389ca01925a`.

## 2026-10-06 — Phase 5AF-R2 repaired CC execution chain qualified

The post-boundary repaired compute-only execution chain was qualified without
resuming the accepted Phase-5AF outcome run.

The qualified chain is the versioned V2 analyzer, V2 I/O runner, and V2 final
executor.

The V2 I/O runner binds existing I/O semantics to the qualified V2 analyzer.

The V2 final executor binds the V2 analyzer and V2 I/O runner and requires an
explicit repair-binding artifact.

Synthetic accepted-style qualification passed seven named checks covering
exact repaired-module hashes, transient and persistent dict-valued NaN
handling, non-finite accounting, all three frozen operating points,
paired-clean/faulted metric construction, and finite reconstruction semantics.

The initial R2 qualifier expected eight checks while seven named checks were
recorded. Only that bookkeeping expectation was corrected. The V2
implementation files remained byte-identical.

No scientific rule changed.

Phase-5AF remained at zero completed shards and was not resumed.

V2 analyzer SHA256: `0cfb60c9b32951482ffcc5be0ca14ec49c91b7d669e43eddf39fe28f26d01da5`.

V2 I/O runner SHA256: `b8f14826f8d31c067f5753c633cec761b4b9e2ad2004e8367e38809e2fdc7bf2`.

V2 executor SHA256: `cfb5d618af37b5164c1e861fb6888b3c4dbe04caee2644a1218cd09f35b4aa50`.

R2 qualification result SHA256: `edd64e473537aef3ba1d71de290bbcd311589f065662b23313cf6cef2de8641c`.

R2 manifest SHA256: `8f2af6ee06656f11eaad39164f85fd1a85b37b0b6636396901492b9d1b08df4f`.

## 2026-10-06 — Phase 5AF-R3 post-boundary continuation authorization frozen

The post-boundary continuation authorization was frozen after qualification of
the versioned V2 analysis, I/O, and final-executor chain.

The original Phase-5AF execution start remains immutable and is hash-bound by
R3. It records the already-crossed one-way outcome boundary under the original
Phase-5AE activation.

There were zero completed shards before repair.

R3 authorizes continuation only with the exact qualified V2 chain. The
historical V1 executor is not authorized to resume production.

The repair scope remains implementation compatibility only: restoring the
already-frozen Phase-5M/5R serialized IEEE non-finite representation.

No threshold, metric, aggregation, uncertainty, stratum, checkpoint, fault
membership, CSC, OnField, model-forward, or fault-execution rule changed.

R3 itself did not resume Phase-5AF, read a new accepted prediction payload,
load an outer label array, apply a threshold, or compute a CC metric.

R3 authorization SHA256: `ebc59be015964a53dee08a903d08a5ba8687bcc4f22fbb8cd936238a439613e3`.

R3 manifest SHA256: `45f5f49803b994a905db02483744cb8ddb8f22a3cf78522b487aabd81730fb1f`.

## 2026-10-06 — Phase 5AF-R4 canonical persistence execution chain qualified

The post-boundary persistence vocabulary audit established that the canonical
frozen persistent value is `persistent_from_onset_until_trial_end` across the
compute-FI contract, Phase-5E plan, Phase-5Z protocol, Phase-5R producer, and
accepted records.

The V2 analyzer alone used the accidental noncanonical spelling
`persistent_from_onset_to_trial_end`.

A versioned V3 chain was qualified. The V3 analyzer differs from V2 only by
using the canonical persistence literal. No alias was introduced.

The prior non-finite decoding repair is preserved.

One transient R3/V2 production shard had already completed before this second
compatibility failure. It was preserved byte-for-byte and was neither deleted
nor rerun during R4.

Synthetic qualification passed ten checks, including canonical persistent
suffix behavior, rejection of the noncanonical alias, transient V2/V3
equivalence, dict-valued NaN preservation, and end-to-end transient/persistent
execution under all three frozen operating points.

No scientific rule changed and production was not resumed.

V3 analyzer SHA256: `22e7ac0d587527abd659585596a2c7b43df7af2f249c5aa594be566677e32544`.

V3 I/O runner SHA256: `ad625947791a1d7763bacac17be2941382f2777349d02bc653053cfa123151d7`.

V3 executor SHA256: `6b99d63805b2dfd64d1a44ec419cb9e9e5ca34573d1df4bdbe2083fdb12df61f`.

R4 result SHA256: `7548f14b4bf95c615feeffd1a757312513cda70adabde4e6454b613cb968f036`.

R4 manifest SHA256: `60b5820013b5e7f74a71a783d3f9bc2bd7a9c40b769886df43866f802abac850`.

## 2026-10-06 — Phase 5AF-R5 V3 continuation authorization frozen

The post-boundary V3 continuation authorization was frozen.

R5 binds the exact qualified V3 analyzer, V3 I/O runner, and V3 executor plus
the complete R4 repair evidence.

Exactly one transient R3/V2 shard had completed before the canonical
persistence-vocabulary failure.

That shard is grandfathered by exact shard ID, outcome hash, and success-marker
hash. It must be preserved byte-for-byte and may not be deleted or rerun.

R4 qualified transient V2/V3 behavioral equivalence, so only this exact
already-completed transient shard is authorized for V2 reuse.

The remaining 731 Phase-5E shards must execute with the exact V3 chain.

No scientific rule changed.

R5 itself did not resume production, read accepted prediction payloads, load
outer labels, apply thresholds, or compute CC metrics.

R5 authorization SHA256: `81dee95ad55bd7dbc63d7abf3cf1caecf6497348f5a74574fdafc8d59d5a329d`.

R5 manifest SHA256: `045940fa23020abb35ae96b69525d18680b898ab104e98bbcc36e01c71e92497`.

## 2026-10-06 — Phase 5AG complete CC outcome execution technically accepted

The completed Phase-5AF prospective compute-only outcome execution was
technically accepted before scientific interpretation.

Technical acceptance verified all 732 frozen Phase-5E shard outputs: one exact
grandfathered transient R3/V2 shard and 731 exact R5/V3 shards.

The grandfathered shard remained byte-identical and was not rerun.

Accepted cardinalities are 20,170,008 outer identities, 29,798,820 fault
records, 30 clean aggregate rows, and 720 CC stratum aggregate rows.

The ordered shard-outcome digest was independently reconstructed in frozen
Phase-5E order and matched the completed execution.

The aggregate, execution summary, and final success artifacts were accepted by
exact SHA256.

The aggregate structure retains FP32/PTQ separation, transient/persistent
separation using the canonical persistent vocabulary, and all three frozen
operating points.

Phase 5AG did not print, rank, optimize, or scientifically interpret metric
values and performed no threshold retuning, checkpoint selection, resampling,
CSC, OnField access, model forward, or new fault execution.

Technical acceptance result SHA256: `9bd147a666db08fa369a1f3f7a144a71415fe0fb71b6f513603a326343c63b19`.

Technical acceptance manifest SHA256: `388970ca6836439799228e0e43e92d62cdbfac4f411ec84e031c7cd4260451ac`.

## 2026-10-06 — Phase 5AH frozen CC aggregate scientifically interpreted

Scientific interpretation was performed only after Phase 5AG technically
accepted the complete frozen Phase-5AF aggregate.

The exact Phase-5Z paired-effect contract was retained: positive degradation
means worse behavior under compute fault.

Of the 720 primary CC strata, 576 have available subject-primary estimates and
existing bootstrap intervals.

All 144 unavailable CC strata are `median_trigger_lead_ms` and remain
`UNAVAILABLE_WITHOUT_IMPUTATION`.

Likewise, six clean median-trigger-lead summaries remain unavailable.

No unavailable timing value was dropped, converted to zero, or imputed.

Equal-target descriptive macros remain unavailable whenever a contributing
target is unavailable.

Common-target FP32/PTQ timing comparisons likewise remain unavailable.

No new subject bootstrap, event bootstrap, inferential CI, scalar robustness
score, rank-based selection, threshold retuning, checkpoint selection, fault
resampling, protocol change, CSC, OnField use, model forward, or new fault
execution occurred.

Interpretation result SHA256: `3841ceb4a1af3ca2f1c32a7d1126ab3f94e6debbb46222f2635e694cec6012fb`.

Interpretation manifest SHA256: `7abeb3a117b0759e0e36698a0437061058fa890cd39d0b70e4d4dbd6d22f6177`.

## 2026-10-06 — Phase 5AI frozen compute-FI CC scientific findings reported

The Phase-5AH scientific interpretation was frozen into a bounded reporting
layer without any new outer-outcome inference or selection.

Persistent faults had larger descriptive equal-target degradation than
transient faults in all 24 available matched variant × operating-point ×
metric comparisons. Six median-trigger-lead comparisons remained unavailable
without imputation.

On the ten common FP32/PTQ targets, PTQ had lower descriptive degradation in
21 of 24 available comparisons and higher degradation in three. Because the
pattern is not uniform and no cross-variant inferential CI was constructed, no
universal variant-superiority claim or model selection was made.

All six clean and all 144 CC subject-primary median-trigger-lead summaries
remain unavailable without imputation. No subject-primary median-lead
robustness conclusion was reported.

No new bootstrap, inferential CI, significance test, scalar robustness score,
threshold retuning, checkpoint selection, fault resampling, protocol change,
CSC, OnField use, model forward, or new fault execution occurred.

Scientific report SHA256: `8a72edb086d255b45c9ad0fd6f9b0d207854611bfa052e6959e11f50a2f6e63f`.

Phase-5AI manifest SHA256: `4cf491e940769091773ebe9587abafebb3b4969be50e61f790e0e2d3cc29bc93`.

## 2026-10-06 — Phase 5 compute-fault study complete and frozen

Phase 5 completed the prospective C0/CC compute-fault study and froze the full
production, technical-acceptance, scientific-interpretation, and reporting
lineage.

The accepted estate contains 732 fault shards, 20,170,008 outer fault
identities, and 29,798,820 fault records.

Production outcome analysis consists of one exact grandfathered transient
R3/V2 shard plus 731 exact R5/V3 shards.

Phase 5AG technically accepted the completed execution before interpretation.

Of 720 primary CC strata, 576 are available and 144
`median_trigger_lead_ms` strata remain unavailable without imputation.
Likewise, six clean timing rows remain unavailable.

Persistent faults have larger descriptive degradation than transient faults in
all 24 available matched equal-target comparisons.

On the ten common FP32/PTQ targets, PTQ has lower descriptive degradation in
21 of 24 available comparisons and higher degradation in three. No universal
variant-superiority claim or model selection is made.

No unavailable timing values were dropped or imputed.

No outer-result feedback into model, target, family, operating-point,
checkpoint, threshold, fault sampling, or protocol selection occurred.

Phase 5 makes no CSC, OnField fall-performance, MCU, or physical-realism claim.

Phase-5AJ final synthesis SHA256: `80adeec431226833529410a0fe1b7550da53aca068c9ca7f0d25fedc4c066760`.

Phase-5AJ final freeze manifest SHA256: `81beb2a7f6abdc8aac4e48dff8075416af4780aadd91ac447030b448bd5c0abb`.

The Phase-5 scope is now complete and eligible for a single major completion
commit and push after this freeze passes final regression.
