# Baseline Selection Protocol

Status:

CANDIDATE

## Purpose

Select the fixed protected task workload used for cross-layer experiments.

This is not an architecture competition.

## Candidates

1. compact 1D CNN
2. depthwise-separable CNN
3. compact TCN

The earlier lightweight DATE-style CNN should be recovered and audited if
its exact implementation is available.

## Primary baseline exclusions

The protected baseline must not initially contain:

- proposed sensor reliability gating
- proposed compute integrity checks
- proposed cross-layer runtime control
- proposed recovery logic

RC-RGD-IMU reliability-aware models may remain comparison systems.

## Audit evidence

For each candidate record:

- source provenance
- reproducibility
- parameter count
- model size
- operations if reliably measurable
- RAM estimate
- quantization support
- deployment compatibility
- clean performance
- inference latency where measurable

## Selection principle

Choose the simplest architecture that provides:

- credible task performance
- stable reproducibility
- practical INT8 deployment
- meaningful internal structure for compute fault injection

Do not select based on which model makes the proposed method appear strongest.

## Frozen baseline record

When selected record:

- source path
- source Git SHA
- architecture configuration
- training recipe
- random-seed policy
- preprocessing
- checkpoint checksum
- FP32 metrics
- INT8 metrics
- export checksum

---

## Phase-1 evidence update

The inherited source audit identified an additional and more task-specific
candidate:

### DATE2025_CNN_400MS_RECONSTRUCTED

This candidate reconstructs the earlier pre-impact fall-detection workload
rather than a generic HAR benchmark.

It is therefore the current **primary candidate pending provenance
verification**.

It is not yet frozen.

The generic:

- compact 1D CNN
- DS-CNN
- compact TCN

remain important comparison and later generalization candidates.

The DATE reconstruction may become the primary protected workload only if
Phase 1 confirms:

- source provenance
- architecture consistency
- preprocessing semantics
- checkpoint availability/integrity or defensible reconstruction procedure
- task decision semantics
- dataset/split lineage
- suitability for later quantization and compute-fault injection

If those requirements fail, the project returns to clean-baseline selection
among the generic compact candidates.

---

## Phase-1 outcome

Phase 1 established a validated task-specific primary candidate:

`DATE2025_CNN_400MS_RECONSTRUCTED`

Verified Phase-1 facts:

- historical checkpoint exists at its pinned external location
- checkpoint SHA-256 matches the historical frozen value
- migrated model accepts the trusted state with strict loading
- parameter count is 63,173
- logits are exactly identical to the trusted implementation on 68
  deterministic parity vectors
- 256-dimensional penultimate features are exactly identical
- historical streaming decisions are identical
- the clean migrated baseline contains no inherited OOD or reliability logic

This is a migration/provenance result.

It does not itself constitute Phase-2 baseline freezing.

Phase 2 must freeze the executable baseline identity, decision semantics,
reference environment, export/quantization path and clean reference evidence.
