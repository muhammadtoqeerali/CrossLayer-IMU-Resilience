# Phase6M prospective CSC pre-forward orchestration candidate v1

## Scope

This additive development stage constructs deterministic pre-forward
shard ownership descriptors for all 732 frozen CSC subject-family
shards, involving 61 subjects and 12 sensor families.

It binds the frozen Phase6K numerical inventory, Phase6K ownership
crosswalk, Phase6K evidence freeze, Phase6L evidence freeze, and
the provenance-pinned Phase6L v2 cache resolver.

The frozen global workload totals are checked, including 4,237,835
model-independent pairs and 21,793,038 seed/variant pair-members.

All 366 real Phase5 clean caches are revalidated by the actual
frozen resolver using the correct mixed Phase5M/Phase5R producer hashes.

The six model-member cache identities in each subject-level shard
descriptor represent candidate cache ownership. They do not establish
per-pair eligibility. Only the frozen compute-stratum eligibility can
determine which model members belong to a particular CSC pair.

## What this implements

- An independently verified frozen 732-shard routing plan
- Deterministic outer-fold, subject, and family identity checks
- Producer-aware six-cache subject ownership
- Real read-only Phase5 clean-cache hash validation
- Per-shard and global numerical census assertions
- A mandatory hard block on real CSC execution
- New regression tests for frozen coverage and negative cases

## What remains to implement

A prospective production body must enumerate each canonical CSC
pair-member from frozen metadata, construct and validate its Phase6G
pre-forward request, condition sensor inputs under the frozen Phase4H
protocol, route the frozen Phase5 compute-fault sequence, and write
the frozen Phase6H output/atomic-success contract.

The existing Phase6J functions provide qualified building blocks,
including condition_sensor_parent,
execute_bound_compute_fault_sequence,
run_model_member_stream, and output record constructors.

This Phase6M module does not yet connect those functions into a real
execution body. It does not load checkpoints, parse prediction rows,
read signal/label arrays, perform fault injection, or run model
forwards. No CSC scientific outcome is produced.

The original historical Phase6E digest serialization remains
unreproduced, and must not be substituted by independent Phase6K
digests.

## Governance

This is an uncommitted development candidate. All earlier frozen
Phase5, Phase6E, Phase6J, Phase6K, and Phase6L files remain unchanged.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE


## Phase6M schema reconciliation

The frozen Phase6K ownership crosswalk stores the authoritative
fold and subject on each enclosing subject-family shard. Its six
candidate cache references are compact records and may omit those
two repeated fields.

Phase6M reconstructs each model-member key from that verified
parent shard plus the cache reference's model variant and seed.
If a compact record contains optional fold or subject fields,
those values must agree with the enclosing shard.

The frozen crosswalk and its SHA-256 are unchanged.
