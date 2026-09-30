# Experimental Contract

## Mandatory regimes

| ID | Sensor | Compute |
|---|---|---|
| C0 | clean | clean |
| CS | faulty | clean |
| CC | clean | faulty |
| CSC | faulty | faulty |

These conditions must remain individually reportable.

## Data roles

Maintain distinct:

- development
- calibration
- final held-out test
- external evaluation
- physical confirmation

## Leakage prevention

Split before stochastic fault injection.

All corrupted derivatives remain in their parent partition.

Development/calibration/test fault seeds must be independent.

## Same-backbone fairness

Causal comparisons should preserve:

- task architecture
- task weights
- preprocessing
- data partition
- fault instances
- task decision semantics

unless the changed quantity is explicitly under study.

## Fault provenance

Every fault realization should ultimately record:

- domain
- family
- target
- severity
- start
- duration
- persistence
- multiplicity
- seed
- parent sample/event
- provenance tier

## Candidate metric groups

Task:

- sensitivity
- specificity
- precision
- F1

Dependability:

- silent incorrect decision rate
- detected fault coverage
- residual error after protection
- recovery success

Safety:

- missed dangerous events
- false protection trigger rate
- pre-impact lead-time margin
- deadline miss rate

Resources:

- latency
- tail latency
- Flash
- RAM
- energy
- incremental protection overhead

Exact publication metrics must be frozen before final testing.

## Statistical principle

Overlapping windows must not automatically be treated as independent
statistical observations.

Final uncertainty analysis must respect subject/event/sequence structure.

## Protocol-change rule

If held-out results motivate a change:

1. preserve the original result
2. create a new protocol version
3. return to development/calibration
4. freeze the successor
5. rerun under a new experiment identity
