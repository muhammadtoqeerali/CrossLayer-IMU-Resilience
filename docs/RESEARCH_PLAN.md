# Research and Implementation Plan

## Phase 0 — Foundation

Freeze:

- research problem
- repository structure
- scientific rules
- reproducibility rules
- publication logic

Exit criteria:

- repository doctor passes
- tests pass
- Git status manually reviewed

---

## Phase 1 — Inherited-Code Audit

Audit workstation and RC-RGD-IMU implementation.

Inspect:

- model architectures
- dataset loaders
- preprocessing
- event/window construction
- training
- evaluation
- quantization
- export
- embedded runtime
- sensor corruption engine
- integrity logic
- runtime logic
- tests
- experiment registries
- manifests

Each reusable item receives one classification:

    REUSE
    ADAPT
    REFERENCE_ONLY
    REJECT

Output:

Machine-readable migration manifest.

---

## Phase 2 — Baseline Freeze

Initial candidates:

- compact 1D CNN
- DS-CNN
- compact TCN

Evaluate:

- reproducibility
- clean performance
- parameter count
- model size
- computational cost
- quantization compatibility
- MCU suitability

Select the simplest credible protected workload.

Freeze:

- source
- architecture
- training
- weights
- preprocessing
- quantization
- export
- clean metrics

---

## Phase 3 — Dataset and Timing Freeze

Freeze:

- dataset version
- subject split
- event definition
- windowing
- stride
- label semantics
- preprocessing
- pre-impact timing
- lead-time computation
- calibration partition
- held-out test partition
- fault seed partitions

---

## Phase 4 — Sensor Fault Engine

Candidate categories:

- bias
- drift
- scale
- noise
- clipping
- freeze
- axis loss
- dropout
- frame loss
- timing jitter
- delay
- orientation/alignment error

Each fault requires:

- scientific definition
- injection layer
- temporal definition
- severity definition
- provenance
- deterministic replay

---

## Phase 5 — Compute Fault Engine

Initial injection targets:

- INT8 weights
- activations
- intermediate buffers

Fault dimensions may include:

- layer
- tensor
- element
- bit
- time
- transient/persistent
- multiplicity

Require bit-exact tests and deterministic replay.

---

## Phase 6 — Cross-Layer Vulnerability Characterization

Evaluate:

    C0
    CS
    CC
    CSC

Measure:

- task errors
- silent errors
- unsafe errors
- severity response
- layer sensitivity
- bit sensitivity
- sensor/compute interaction
- timing effects

Primary output:

Cross-layer vulnerability map.

Potential DATE LBR milestone.

---

## Phase 7 — Sensor Integrity

Build the smallest justified mechanism capable of identifying dangerous
sensor degradation observed in Phase 6.

---

## Phase 8 — Compute Integrity

Candidate mechanisms include:

- checksums
- activation invariants
- signatures
- selective recomputation
- duplicated critical operations
- temporal consistency

Mechanism selection must follow Phase-6 evidence.

---

## Phase 9 — Runtime Supervisor

Possible evidence:

- sensor integrity
- compute integrity
- task evidence
- deadline slack

Possible actions:

- continue
- mitigate sensor
- recompute
- run protected path
- fallback

Initial implementation should be deterministic and explainable.

---

## Phase 10 — Recovery and Fallback

Compare:

    baseline
    sensor-only protection
    compute-only protection
    always-on protection
    adaptive cross-layer protection

Measure both protection benefit and cost.

---

## Phase 11 — Embedded Validation

After hardware freeze measure:

- mean latency
- tail latency
- deadline misses
- Flash
- RAM
- power
- energy
- normal-path overhead
- protection-path overhead

Real MCU evidence must be separated from host-side estimates.

---

## Phase 12 — Causal Ablations

Use:

- same backbone
- same weights
- same fault instances
- same splits

Remove one mechanism at a time.

---

## Phase 13 — Generalization

Evaluate at least one additional:

- model architecture

and preferably one additional scientifically compatible:

- dataset

---

## Phase 14 — Frozen Confirmation

Before final execution freeze:

- baseline
- model weights
- split
- fault taxonomy
- severities
- fault seeds/policy
- monitors
- thresholds
- runtime policy
- recovery rules
- metrics

Then execute held-out confirmation.

---

## Phase 15 — Publication

Generate all numerical paper evidence from traceable experiment artifacts.

Main paper emphasis:

- cross-layer problem
- vulnerability characterization
- adaptive protection
- recovery/graceful degradation
- embedded cost
- safety/deadline evidence

---

## Dependency amendment after Phase 2

The original phase sequence listed baseline quantization before the formal
data-protocol freeze.

Phase-2 implementation showed that final static INT8 calibration depends on a
representative clean calibration partition.

Therefore the executable dependency is now:

1. freeze the FP32 protected reference baseline
2. freeze the static-PTQ protocol
3. freeze the dataset, event, subject, split and calibration partitions
4. execute static INT8 calibration using only the permitted calibration split
5. freeze the quantized deployment variant
6. begin sensor, compute and combined fault characterization

This is a methodological dependency correction.

It does not change the research questions.

No fault-study outcome may influence quantizer calibration or quantizer
selection.
