# Phase 5AF-R4 — Canonical Persistence Execution-Chain Repair V1

## Status

**QUALIFIED_POST_BOUNDARY_CANONICAL_PERSISTENCE_V3_CHAIN**

## Diagnostic result

The frozen persistence vocabulary is unambiguous.

The canonical values are:

- `transient_one_inference`
- `persistent_from_onset_until_trial_end`

The persistent value is present in:

- the Phase-5A compute-FI contract;
- the frozen Phase-5E execution plan;
- the Phase-5Z outcome protocol;
- the Phase-5R producer;
- accepted persistent prediction records.

The V2 analyzer alone used the noncanonical spelling
`persistent_from_onset_to_trial_end`.

## Repair

The V2 chain remains unchanged.

The versioned V3 chain is:

`cc_outcome_analyzer_v3.py`
→ `cc_outcome_io_runner_v3.py`
→ `cc_outcome_executor_v3.py`.

The V3 analyzer differs from V2 only by replacing the accidental persistence
literal `persistent_from_onset_to_trial_end` with the frozen canonical literal
`persistent_from_onset_until_trial_end`.

No alias is introduced.

The V3 I/O runner differs from V2 only by binding to the V3 analyzer.

The V3 executor differs from V2 only by binding to the V3 analyzer / V3 I/O
runner and requiring the V3 repair-binding hashes.

## Scientific invariance

The semantics are unchanged: a persistent fault applies from its frozen onset
through the final window of the same trial.

No threshold, comparator, metric, aggregation, uncertainty, target stratum,
checkpoint, fault identity, or membership rule changes.

The prior non-finite decoding repair is preserved.

## Production state

One transient R3/V2 shard completed before the persistence-vocabulary failure.

That shard is preserved byte-for-byte.

R4 neither deletes nor reruns it.

Because the V2→V3 analyzer change affects only the persistent branch, transient
V2/V3 reconstruction equivalence is explicitly qualified.

## Qualification

Synthetic qualification covers:

- exact V3 module hash binding;
- reproduction of the V2 canonical-persistence failure;
- canonical persistent suffix reconstruction;
- rejection of the noncanonical alias;
- transient V2/V3 equivalence;
- preservation of dict-valued NaN handling;
- transient V3 end-to-end execution;
- persistent V3 end-to-end execution;
- all three frozen operating points;
- paired-clean/faulted metric construction.

## Boundary

R4 does not resume production and does not authorize resume by itself.

No accepted threshold or CC metric is computed during R4.

A new continuation authorization must bind the exact V3 chain and explicitly
authorize preservation/reuse of the already-completed transient R3/V2 shard
before production continues.
