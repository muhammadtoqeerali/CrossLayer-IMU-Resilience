# Phase6K CSC metadata-evidence freeze v1

## Frozen qualification result

This freeze preserves a prospective, metadata-only Phase6K qualification
evidence package. The full independent numerical audit reproduced the
frozen Phase6E workload cardinalities across 61 outer-test subjects,
12 sensor families, and 732 nonempty subject-family CSC shards.

The retained surface contains 4,237,835 model-independent CSC pairs,
21,793,038 variant/seed-expanded pair-members, and 2,104 documented
source-trial structural omissions. It includes 1,018,215 zero-temporal-
overlap pairs; these were not silently discarded.

Temporal counts are 35,167,107 sensor-reference member-windows,
411,540,372 compute-faulted member-windows, 19,926,021 simultaneous-
overlap member-windows, and 426,781,458 union member-windows.

## Clean-cache lineage and ownership

Exactly 366 frozen Phase5 clean caches were verified using their success
markers, producer identities, coverage, and SHA-256 output-file hashes.
Their 1,642,980 clean window evaluations are reused, never regenerated.

The accepted Phase5M canary produced one clean cache; the qualified
Phase5R full-fleet executor produced the other 365. Both executable
source-file identities match the frozen qualification records.

Every subject owns six variant/seed clean-cache identities and 12 CSC
sensor-family shards. These are candidate subject-level cache references,
not a claim that all six model members are eligible for every pair.
The frozen per-pair compute-stratum eligibility remains controlling.

## Explicit unresolved scientific and integration boundaries

1. The historical Phase6E derivation serializer has not been recovered.
   Phase6E's original stream/binding SHA-256 digests have not been
   reproduced. Exact numerical equivalence and new Phase6K digests
   are independent evidence, not replacements for the original hashes.

2. The frozen Phase6J integration has not been upgraded or qualified
   for producer-aware runtime cache selection. A future prospective
   adapter must use the Phase5M canary producer for exactly one cache
   and the Phase5R producer for the remaining 365. Wrong-producer
   rejection has been tested, but real integration remains pending.

3. This freeze covers metadata numerical consistency, Phase5 clean-cache
   integrity, provenance, and subject-family ownership only. It does
   not authorize model loading, CSC fault application, model forwards,
   label/score reads, performance analysis, or threshold selection.
   The CSC outer execution authorization gate remains closed.

## Provenance and immutability

The existing nine-file Phase6K evidence candidate is deliberately left
unchanged. Its historical UNCOMMITTED_CANDIDATE status is a record of
how that evidence was assembled; this separate freeze binds its exact
file hashes without rewriting that original status.

The machine-readable freeze manifest pins the candidate, four archived
evidence reports, the auditor and regression tests, and the relevant
frozen Phase5/Phase6 runtime and acceptance sources. It also binds this
document and the dedicated freeze regression tests.

The freeze becomes a version-controlled evidence record only after an
explicit, separately checked Git commit. It is not an execution gate.
