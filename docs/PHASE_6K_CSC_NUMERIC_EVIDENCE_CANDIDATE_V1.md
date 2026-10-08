# Phase6K CSC numerical evidence — uncommitted qualification candidate

## Verified scope

The metadata-only audit independently reproduced the frozen Phase6E numerical
workload for 61 outer-test subjects, 12 sensor families, 732 nonempty CSC
shards, and 4,237,835 model-independent sensor/compute pairs.

The frozen workload expands to 21,793,038 model-variant/checkpoint-seed
pair-members. The verified temporal counts include 35,167,107 sensor reference
member-windows, 411,540,372 compute-faulted member-windows, 19,926,021
simultaneous-overlap member-windows, and 426,781,458 union member-windows.

The audit retains 1,018,215 zero-overlap model-independent pairs, and
reproduces the 2,104 source-trial structural omissions.

## Phase5 clean cache provenance

All 366 frozen clean-cache artifacts have been validated against their
success markers, expected metadata, producer identities, and output-file
SHA-256 hashes. The producer split is one accepted Phase5M canary and
365 Phase5R full-fleet caches.

The Phase6K ownership crosswalk identifies six subject-owned clean-cache
identities for each subject and 12 CSC subject-family shards. These six
identities are an ownership/candidate set, not a claim that every clean-cache
member is eligible for every pair. Frozen compute-stratum model eligibility
continues to control pair-member expansion.

## Scientific reproducibility limitation

The original Phase6E derivation serializer was not recovered.
The historical Phase6E row/stream SHA-256 digests have therefore not been
reproduced, despite exact numerical agreement.

The new Phase6K inventory digests are independent, differently structured
evidence. They do NOT replace or prove the historical Phase6E digests.
No frozen Phase6E or Phase6J file is changed by this candidate.

## Execution authorization limitation

The existing Phase6J cache resolver correctly rejects an unexpected
producer hash. Future producer-aware integration must select the Phase5M
hash for the accepted canary and the Phase5R hash for the other 365 caches.

CSC execution is NOT authorized. No CSC model forward or real fault
execution has been performed in Phase6K. No outer performance feedback was
used, and no thresholds or frozen experimental selections were changed.

## Evidence package

The accompanying manifest binds four independently verified JSON artifacts:
the complete numeric workload report, the Phase5 cache integrity audit,
the producer-aware compatibility audit, and the 732-shard ownership crosswalk.

This package is an uncommitted qualification candidate. Separate review,
prospective integration qualification, and an explicit future authorization
decision are required before any real CSC execution.
