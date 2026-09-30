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
