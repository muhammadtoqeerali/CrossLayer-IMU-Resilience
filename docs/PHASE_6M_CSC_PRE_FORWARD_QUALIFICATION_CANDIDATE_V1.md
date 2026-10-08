# Phase6M CSC pre-forward qualification candidate v1

## Scope

This is an evidence-only prospective qualification package. It
preserves the previously verified Phase6M metadata-only preflight,
representative frozen CSC request witness, and synthetic model-member
stream lifecycle.

## Full frozen fleet preflight

The independent Phase6M preflight reconciles 732 subject-family shards,
61 outer subjects, 12 sensor families, and all 366 frozen clean caches.
All 732 existing output-file SHA-256 validations succeeded through
the provenance-pinned Phase6L v2 resolver. The frozen numerical
workload includes 4,237,835 model-independent CSC pairs and
21,793,038 variant/seed-expanded pair-members.

These counts are planned metadata workload, not actual CSC inference.

## Canonical metadata witness

Frozen subject 9, fold 5, was re-enumerated through the independent
Phase6K metadata auditor, matching all 12 frozen shard descriptors
and covering all 28 compute strata.

Thirty-seven representative selected pairs generated 189 validated
Phase6G pre-forward requests. Frozen Phase6J synthetic hooks exercised
5,310 synthetic sensor-reference rows and 13,989 synthetic compute-fault
rows. No real model forward or fault injection occurred.

## Model-member stream lifecycle

The 189 request identities were grouped into 69 synthetic model-member
streams by subject-family, variant, and checkpoint seed. The frozen
Phase6J stream helper performed exactly one synthetic bundle load
and one synthetic release per stream, with 189 synthetic pair-hook calls.

These synthetic callbacks do not demonstrate real checkpoint loading,
CUDA cleanup, PTQ persistent-weight restoration, or model inference.

## Scientific limitations

No full-fleet canonical pair-member stream has been materialized or
qualified by Phase6M. The 189-request witness is a sample of real frozen
metadata, not a substitute for the 21,793,038 planned pair-members.

Original Phase6E historical digest serialization remains unresolved.
Independent Phase6K digests do not replace the historical digests.

Frozen Phase6E, Phase6J, Phase6K and Phase6L files remain unchanged.
There is no execution authorization.

## Governance

The three JSON evidence reports are preserved byte-for-byte. This
candidate binds the archived reports, all Phase6M source and test files,
and upstream frozen dependency hashes.

The single historical Phase5R canary-only regression assertion is
explicitly deselected from the full suite.

A separate freeze manifest documents this evidence without changing
the historical candidate status.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
