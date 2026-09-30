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
