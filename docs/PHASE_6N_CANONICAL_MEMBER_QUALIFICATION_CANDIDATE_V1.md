# Phase6N canonical member qualification candidate v1

## Full-fleet qualification

This evidence-only candidate preserves the full-subject and full-fleet
Phase6N metadata audit results.

Every one of the 61 frozen outer-test subjects and 732 subject-family
shards was independently re-enumerated through the frozen Phase6K
metadata auditor.

The full planned workload contains 4,237,835 model-independent
sensor/compute fault pairs, expanding into 21,793,038 eligible
model-variant/seed pair-members.

The 4,392 canonical member streams match the expected six candidate
model members per subject-family shard. All per-shard workload counts
match the independently frozen Phase6K numerical evidence.

## Complete subject-9 reconciliation

The previous full subject-9 audit is preserved byte-for-byte.
Its 38,107 pairs, 195,768 members and 72 member streams match
the new fleet-wide audit at the stream ownership and count level.

All 189 earlier Phase6M representative request IDs were already
reconciled in the independently preserved subject-9 report.

## Phase6G validation scope

Exactly 8,784 representative Phase6G pre-forward request payloads
were built and validated across the 61 subjects.

This is sampled contract validation. It does not establish that all
21,793,038 complete Phase6G request payloads were individually built
and validated.

## Digest provenance

Phase6N independently defines new metadata fingerprint and stream
serializations. The new digests do not reproduce the historical
Phase6E stream digests and must not replace them.

The two Phase6N reports also use different new stream digest schemas.
Their counts and identities are independently reconciled; their
different digest values are not claimed to be equivalent.

## Execution boundary

No IMU sample values, label arrays, clean prediction rows or probability
payloads were parsed.

No real sensor fault, compute mutation, model load, model forward,
CSC shard execution or output-artifact commit was performed.

The prospective CSC execution body remains unreleased. The frozen
Phase6J runtime and all earlier evidence freezes remain unchanged.

This evidence candidate is not an execution authorization.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
