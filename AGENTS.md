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

Phase 0.

## Next phase

Audit RC-RGD-IMU before copying scientific implementation.

Classify inherited components as:

- REUSE
- ADAPT
- REFERENCE_ONLY
- REJECT

## Initial protected-baseline candidates

- compact 1D CNN
- DS-CNN
- compact TCN

Reliability-aware models are not initially accepted as the protected baseline.

## Publication sequence

First scientific milestone:

Cross-layer fault characterization.

Main scientific milestone:

Adaptive sensor + compute integrity, selective protection, recovery/fallback
and embedded validation.
