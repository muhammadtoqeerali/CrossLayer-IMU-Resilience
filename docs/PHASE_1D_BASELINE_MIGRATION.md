# Phase 1D Protected-Baseline Migration

## Candidate

`DATE2025_CNN_400MS_RECONSTRUCTED`

## Why this model

The project requires a protected task workload that:

- represents the target pre-impact fall-detection problem
- predates the proposed CrossLayer protection mechanisms
- has credible historical provenance
- is compact enough for MCU-oriented work
- exposes meaningful internal tensors for later compute-fault injection

The recovered DATE-2025 CNN satisfies these requirements better than a generic
HAR baseline.

## Historical checkpoint

The original external checkpoint was located and independently verified by
SHA-256.

It is intentionally not copied into Git.

A task-only state-dict artifact is generated locally after the trusted
historical loader succeeds.

The artifact remains under `artifacts/`, which is excluded from Git.

## Scientific distinction

Migration is not baseline freezing.

Phase 1 establishes that the candidate can be reproduced faithfully.

Phase 2 still has to freeze:

- dataset/split protocol
- baseline evaluation protocol
- primary decision semantics
- task reference metrics
- quantization path
- exported model identity
- embedded execution contract

## Contamination prevention

No inherited:

- OOD threshold
- integrity cause
- reliability gate
- trust state
- recovery action

is imported into the protected baseline.

Those mechanisms would invalidate the clean baseline needed for the later
cross-layer causal study.
