# Phase6L producer-aware cache evidence freeze v1

## Qualification

This is an evidence-only freeze of the prospective Phase6L metadata
cache-validation interfaces, v1 and v2.

Both actual frozen-resolver audits verified 366 Phase5 clean caches,
including 732 output-file SHA-256 validations per audit and
1,642,980 clean-cache window evaluations.

The provenance split is exactly one accepted Phase5M canary and
365 Phase5R full-fleet clean caches.

All 61 subjects retain their six frozen model-variant/seed cache
identities. No new clean inference was performed.

## Validator provenance

The original v1 adapter accepted a caller-supplied resolver and
Phase5 validator. Its qualification audit independently pinned
their sources.

The prospective v2 adapter removes those caller-supplied validator
parameters and checks module paths, SHA-256 source identities,
module ownership, and function code origins.

Ordinary validator substitution is rejected by tests. These checks
are not a complete adversarial guarantee against sophisticated
in-memory Python code-object replacement.

## Scientific boundary

The original Phase6E digest serialization remains unrecovered.
The historical Phase6E binding SHA-256 values have not been
reproduced or replaced.

Phase6K frozen workload evidence is preserved unchanged.
Neither Phase6L version changes any frozen Phase6E or Phase6J file.

## CSC execution boundary

The frozen Phase6J execution body is deliberately unreleased.
Phase6L validates existing cache artifacts; it does not apply
sensor faults, inject compute faults, perform model forwards,
calculate CSC outcomes, or execute CSC shards.

This freeze does not authorize a CSC canary or fleet execution.
A separate prospective runtime, qualification process, and explicit
execution gate are required.

## Version-control governance

The nine pre-existing candidate files are retained byte-for-byte.
The candidate's original UNCOMMITTED_CANDIDATE status is preserved
as historical evidence.

This separate formal freeze binds all nine candidate file hashes,
frozen dependencies, its documentation, and regression tests.

The wider regression suite has one explicitly documented historical
Phase5R canary-only test deselection.

Publication of this freeze does not authorize experimentation.
