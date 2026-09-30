# CrossLayer-IMU-Resilience Master Project Context

## 1. Project identity

Project:

CrossLayer-IMU-Resilience

Research direction:

Cross-layer runtime resilience for safety-critical wearable TinyML under
physical IMU faults and embedded-computation faults.

This is a new scientific project.

Selected engineering infrastructure may later be inherited from RC-RGD-IMU.

Scientific assumptions from earlier projects are not automatically valid.

---

## 2. Core problem

Wearable AI reliability is commonly evaluated at one layer at a time.

Sensor-robust models may still fail when their weights, activations,
intermediate memory or computation are corrupted.

Hardware-protected inference may still fail when the physical sensor itself
becomes unreliable.

This project studies both domains within the same end-to-end system.

The central question is:

Can a resource-constrained wearable AI system identify and mitigate
degradation from both physical sensing and embedded computation while
preserving safety-critical timing and low system overhead?

---

## 3. Mandatory fault domains

The project distinguishes:

    CLEAN
    SENSOR
    COMPUTE
    COMBINED

These must never be silently collapsed into one generic corruption category.

---

## 4. Primary application

The primary application is wearable IMU-based pre-impact fall detection.

The application provides a safety-critical setting in which:

- a missed event matters
- a false trigger matters
- decision timing matters
- embedded resource limits matter

Application-specific findings must remain distinguishable from general
cross-layer dependability claims.

---

## 5. Primary dataset candidate

UNIVRFall is the intended primary dataset candidate.

The final data protocol is not yet frozen.

At minimum the final protocol must enforce:

- subject-disjoint evaluation
- split before stochastic fault generation
- no corruption-realization leakage across partitions
- no final-test tuning
- explicit event semantics
- explicit timing semantics
- explicit preprocessing provenance

An independent dataset should later be considered when scientifically
compatible.

---

## 6. Protected task model

The protected task model must initially be simple, compact and suitable for
MCU deployment.

Initial candidate families:

- compact 1D CNN
- depthwise-separable CNN
- compact TCN

The primary baseline must not initially contain the proposed cross-layer
protection mechanisms.

Reliability-aware models may later be used as comparators.

---

## 7. Sensor fault domain

Candidate sensor faults include:

- constant bias
- gradual bias drift
- scale-factor error
- additive measurement noise
- clipping or saturation
- frozen or stuck channels
- axis loss
- intermittent dropout
- packet or frame loss
- acquisition timing jitter
- delay
- orientation or alignment error

This is a candidate list only.

Fault families and severities must be validated before freezing.

Arbitrary perturbation magnitudes must not be described as physically
realistic.

---

## 8. Compute fault domain

Candidate compute-side targets include:

- quantized weights
- activations
- intermediate feature buffers
- SRAM-resident inference state
- selected intermediate arithmetic values

The project should distinguish where scientifically relevant:

- transient faults
- persistent faults
- single-bit faults
- multi-bit faults
- burst or word-level corruption

Software fault injection is the initial controlled evidence tier.

Physical fault injection is optional later evidence.

Software injection must never be represented as physical hardware evidence.

---

## 9. Cross-layer operating regimes

The first major characterization uses:

    C0  = clean
    CS  = sensor fault only
    CC  = compute fault only
    CSC = sensor + compute fault

The goal is to determine how faults propagate to task and safety outcomes.

Important outcomes include:

- correct decision
- incorrect decision
- detected degradation
- silent incorrect decision
- missed dangerous event
- false protective trigger
- timing failure
- recovery success

Exact metric definitions must be frozen before final testing.

---

## 10. Project 2 role

The fault-injection infrastructure is implemented before the adaptive
protection framework.

It must determine:

- which sensor faults matter
- which compute faults matter
- which layers/locations are vulnerable
- which faults create silent failures
- how vulnerability changes with severity
- how sensor and compute faults interact

This produces the empirical basis for the protection system.

---

## 11. Project 1 role

After characterization the project develops:

- sensor-integrity evidence
- compute-integrity evidence
- runtime supervision
- selective recovery
- conservative fallback

Protection should be activated according to evidence rather than simply
duplicating all computation continuously.

Always-on protection remains an important comparator.

---

## 12. Intended runtime structure

Conceptually:

    IMU
      ->
    sensor-integrity assessment
      ->
    task inference
      ->
    compute-integrity assessment
      ->
    runtime supervisor
      ->
    action

Candidate runtime actions:

- normal continuation
- sensor mitigation
- selective recomputation
- protected inference
- safe fallback

The exact state machine is not yet frozen.

The first implementation should prefer deterministic and explainable runtime
logic.

---

## 13. Same-backbone comparison principle

The final causal comparison should conceptually include:

    B0
    baseline

    B0 + S
    sensor-side protection

    B0 + C
    compute-side protection

    B0 + S + C always active
    always-on combined protection

    B0 + adaptive cross-layer supervisor
    proposed system

The backbone and fault instances should remain identical whenever possible.

---

## 14. Primary evaluation philosophy

Classification accuracy alone is insufficient.

Candidate dependability and safety outcomes include:

- silent unsafe decision rate
- unsafe decision rate
- fault detection coverage
- recovery success rate
- protection false-trigger rate
- deadline miss rate
- pre-impact lead-time margin

Resource evidence includes:

- mean latency
- tail latency
- Flash
- RAM
- power
- energy
- incremental normal-path overhead
- fault-path overhead

Exact metric definitions must be frozen before final held-out evaluation.

---

## 15. Hardware status

STM32F722 was used as an inherited target candidate in earlier work.

For this project it is not yet frozen.

The final MCU must be selected after confirming:

- actual available board
- toolchain
- memory
- compute capability
- profiling support
- deployment compatibility

No real-hardware claim may be made until actual hardware execution exists.

---

## 16. Evidence provenance tiers

P0

Offline synthetic or software-injected evidence.

P1

Firmware-injected evidence on the deployed software stack.

P2

Hardware-in-the-loop or physical-interface fault evidence.

P3

Physical sensing/configuration/trusted-status evidence.

These evidence levels must remain distinct.

---

## 17. Data-role separation

Maintain separate roles for:

- development
- validation/calibration
- final held-out testing
- external evaluation
- physical hardware confirmation

Final held-out data must never determine:

- architecture
- model selection
- thresholds
- fault severities
- fault budgets
- recovery thresholds
- runtime policies
- quantization choices

---

## 18. Corruption leakage rule

Dataset splitting occurs before stochastic fault injection.

Clean and corrupted derivatives of one parent sequence remain in the same
partition.

Development, calibration and final-test corruption seeds must remain
independent.

---

## 19. Baseline freezing

Before cross-layer fault characterization, freeze:

- dataset split
- preprocessing
- windowing
- labels
- baseline architecture
- training recipe
- checkpoint
- quantization method
- model export procedure
- clean reference metrics

The backbone must not later be changed merely because a different model
produces more favourable resilience results.

---

## 20. Fault-protocol freezing

Before final held-out testing freeze:

- fault families
- injection locations
- duration
- severity
- temporal behaviour
- transient/persistent semantics
- multiplicity
- randomization policy
- fault seeds or deterministic seed-generation policy

A later change creates a new protocol version.

---

## 21. Runtime-protocol freezing

Before final held-out testing freeze:

- monitor definitions
- calibration data
- thresholds
- runtime states
- recovery actions
- fallback rules
- deadline constraints
- hysteresis if used

Final held-out results must not tune these values.

---

## 22. Reproducibility

Publication-facing experiments must record:

- Git commit
- configuration
- dataset manifest
- model identity
- model checksum
- random seed
- fault specification
- software environment
- hardware environment where applicable
- generated artifact identity

---

## 23. Publication milestones

Milestone A

Cross-layer fault characterization.

This may support a DATE late-breaking result if evidence is sufficiently
novel and mature.

Milestone B

Complete adaptive cross-layer resilience framework with embedded validation.

This is the main full-paper target.

Scientific claims follow evidence.

---

## 24. Research questions

RQ1

How do physically meaningful IMU faults and embedded-computation faults
differ in their propagation to safety-critical TinyML decisions?

RQ2

What additional failure regimes appear when sensor and compute faults occur
together?

RQ3

Can lightweight sensor- and compute-integrity evidence detect sufficiently
dangerous degradation for runtime intervention?

RQ4

Can selective cross-layer protection reduce silent unsafe decisions while
using less normal-path overhead than always-on protection?

RQ5

Can the resulting system preserve the application's real-time safety
requirements on a constrained embedded target?

---

## 25. Non-goals

This project is not primarily:

- a neural architecture competition
- a generic augmentation project
- an accuracy-only paper
- an ordinary noisy-input robustness paper
- an isolated bit-flip study
- a permanently redundant inference system

Any new model component must be justified by the cross-layer dependability
problem.

---

## 26. Engineering inheritance

Primary engineering source candidate:

    muhammadtoqeerali/RC-RGD-IMU

TRUST-ROBOT provides a useful staged-development and protocol-freezing
example.

The falling simulator may later support physically grounded fault modelling.

Nothing is copied wholesale without audit.

---

## 27. Development phases

Phase 0
Scientific and repository foundation

Phase 1
Inherited-code audit and baseline reconstruction

Phase 2
Baseline reproduction, selection, quantization and freeze

Phase 3
Dataset, event, split and timing protocol freeze

Phase 4
Sensor fault engine

Phase 5
Compute fault engine

Phase 6
Cross-layer vulnerability characterization

Phase 7
Sensor-integrity mechanism

Phase 8
Compute-integrity mechanism

Phase 9
Adaptive runtime supervisor

Phase 10
Recovery and fallback

Phase 11
Embedded deployment and profiling

Phase 12
Causal ablations

Phase 13
Architecture and/or dataset generalization

Phase 14
Frozen final confirmation

Phase 15
Publication evidence generation and manuscript preparation

---

## 28. Current status

Date:

2026-09-30

Current state:

- workstation repository initialized
- GitHub remote configured
- Phase 0 scientific foundation being created
- no inherited model copied
- baseline not frozen
- dataset protocol not frozen
- sensor fault protocol not frozen
- compute fault protocol not frozen
- MCU target not frozen
- no publication-facing result exists

Next task after Phase 0:

Audit RC-RGD-IMU implementation and identify the exact reusable baseline,
data, training, quantization, embedded and testing infrastructure.

---

## 29. Mandatory future-agent rule

Before scientific modification read:

1. MASTER_PROJECT_CONTEXT.md
2. AGENTS.md
3. docs/RESEARCH_PLAN.md
4. docs/EXPERIMENT_CONTRACT.md
5. docs/CLAIM_LEDGER.md
6. docs/PHASE_LOG.md
7. the relevant phase configuration

An inherited file is not scientifically approved merely because it exists.
