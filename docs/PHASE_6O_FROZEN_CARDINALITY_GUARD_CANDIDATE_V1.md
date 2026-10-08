# Phase6O frozen-cardinality guard candidate v1

## Objective

Close the coverage-accounting gap between a cryptographically valid
Phase6H success marker and the exact independently frozen CSC
per-shard workload.

The frozen Phase6J success-marker validator verifies the artifact
identity, gate/runtime bindings, required filenames, and SHA-256
of all four declared output files.

It does not independently compare actual JSONL row counts or
the coverage fields against Phase6K's per-shard workload.

This candidate qualifies an additional fail-closed guard without
modifying the frozen Phase6J runtime.

## Frozen expected-count census

The guard independently reconciles 732 frozen Phase6K shard rows
against the full Phase6N stream inventory and shard metadata.

The reconstructed global expected records are:

- 21,793,038 pair-member records
- 35,167,107 sensor-reference records
- 411,540,372 CSC fault-window records
- 19,926,021 simultaneous-overlap member-window records

The first three are physical JSONL record-count obligations.
The overlap figure is a separate scientific accounting obligation.

All 732 exact-count claims must pass. All 732 deliberately
decremented pair-member claims must fail.

## Disposable output attack qualification

An explicitly synthetic sentinel shard is created in temporary
storage using the frozen Phase6H preparation/commit helpers.

The qualification demonstrates that the existing frozen success
marker may continue to pass its hash/binding checks when:

1. Coverage counts in the success marker have been forged.
2. An output file has fewer records but its new SHA is correctly
   written back into the marker.

The new fixture guard rejects both conditions.

It also rejects malformed CRLF JSONL framing despite an updated
matching file SHA.

Temporary fixture directories and their success markers are
removed automatically.

## Critical scope boundaries

No real CSC shard outputs are read in this qualification.

The guard verifies full-fleet expected metadata, declared counts,
and physical record counts on synthetic fixtures only.

It does not prove that 411 million real fault records satisfy the
frozen scientific field/value constraints.

In particular, it does not independently count actual physical
overlap-true records in a real shard. That would require reading
and validating fault-record semantics in a separately authorized
production output stage.

The new guard is not yet wired into a real production writer or
reader. Frozen Phase6J remains unchanged.

The original historical Phase6E serialized digests remain
unreproduced and unchanged.

No model forward, fault application, checkpoint load, actual
CSC execution or authorized canary occurs.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
