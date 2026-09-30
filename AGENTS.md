# Agent Operating Contract

Any AI agent or developer modifying this repository must read this file.

## Mandatory context

Before scientific work read:

1. `MASTER_PROJECT_CONTEXT.md`
2. `docs/RESEARCH_PLAN.md`
3. `docs/EXPERIMENT_CONTRACT.md`
4. `docs/CLAIM_LEDGER.md`
5. `docs/PHASE_LOG.md`
6. relevant phase configuration

## Main project objective

Develop cross-layer resilience for safety-critical wearable TinyML under:

- sensor faults
- compute faults
- combined sensor and compute faults

Primary initial application:

IMU-based pre-impact fall detection.

## Scientific rules

1. Do not use final test data for model selection or tuning.

2. Split parent data before stochastic fault generation.

3. Keep sensor and compute faults scientifically separate.

4. Do not describe arbitrary corruption ranges as physically realistic.

5. Do not claim physical hardware evidence from software injection.

6. Do not claim recovery unless a frozen recovery definition is satisfied.

7. Do not equate classification accuracy with safety.

8. Do not silently modify frozen protocols after held-out testing.

9. Do not copy RC-RGD-IMU scientific assumptions without audit.

10. Do not change the protected backbone merely because another model makes
    the proposed method look better.

11. Record provenance for publication-facing experiments.

12. Preserve separate evidence roles for:
    - development
    - calibration
    - final test
    - external validation
    - physical validation

## Engineering rules

1. Prefer small validated phase transitions.

2. New reusable source modules require tests.

3. Generated datasets, checkpoints and large experiment outputs are normally
   excluded from Git.

4. Scientific protocols belong under `configs/`.

5. Reusable implementation belongs under `src/crosslayer_resilience/`.

6. Experiment entry points belong under `experiments/`.

7. Embedded implementation belongs under `embedded/`.

8. Reproducibility manifests belong under `manifests/`.

9. Update `docs/PHASE_LOG.md` at every completed phase.

10. Keep this file synchronized with project-wide changes.

## Current phase

Phase 2 — protected-baseline reproduction, selection, quantization and freeze.

## Phase-1 validated primary candidate

`DATE2025_CNN_400MS_RECONSTRUCTED`

Phase-1 evidence established:

- exact historical external checkpoint located
- historical checkpoint SHA-256 verified
- task architecture migrated into the CrossLayer namespace
- 63,173 trainable parameters verified
- 68 deterministic parity vectors executed
- exact task-logit parity obtained
- exact penultimate-feature parity obtained
- historical streaming-decision parity obtained
- normalized state-dict reload parity obtained
- no inherited OOD/integrity/reliability protection exists in the clean baseline

The candidate is still **not Phase-2 frozen**.

## Current task

Phase 2 must determine and freeze the executable protected baseline without
using final held-out CrossLayer fault results.

Phase 2 must address:

- final protected-baseline identity
- reference execution environment
- checkpoint/state identity
- task decision semantics
- clean task reference behavior
- model export
- quantization
- FP32 versus quantized parity/acceptability
- model size and compute/resource evidence
- deployment feasibility

The generic compact 1D CNN, DS-CNN and TCN remain comparison or later
generalization candidates. They must not replace the primary candidate merely
because they yield more favorable fault-robustness results.

The formal dataset/event/split/timing protocol remains a subsequent protocol
freeze and must be completed before final fault characterization.

## Publication sequence

First scientific milestone:

Cross-layer fault characterization.

Main scientific milestone:

Adaptive sensor + compute integrity, selective protection, recovery/fallback
and embedded validation.
