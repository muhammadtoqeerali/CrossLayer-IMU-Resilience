# Phase 5AF-R5 — V3 Continuation Authorization V1

## Status

**FROZEN_POST_BOUNDARY_V3_CC_CONTINUATION_AUTHORIZATION**

## Purpose

R5 authorizes continuation of the already-started Phase-5AF prospective
compute-only outcome analysis using the exact qualified V3 implementation
chain.

Scientific rules remain immutable because the one-way accepted-outcome
boundary has already been crossed.

## Qualified V3 chain

The authorized continuation chain is:

`cc_outcome_analyzer_v3.py`
→ `cc_outcome_io_runner_v3.py`
→ `cc_outcome_executor_v3.py`.

V3 preserves the earlier non-finite decoding repair and uses the canonical
frozen persistent vocabulary:

`persistent_from_onset_until_trial_end`.

The accidental consumer spelling
`persistent_from_onset_to_trial_end` is not authorized as an alias.

## Grandfathered completed shard

Exactly one transient shard completed under R3/V2 before the canonical
persistence-vocabulary defect was encountered.

R4 proved transient V2/V3 behavioral equivalence.

Therefore R5 authorizes reuse of exactly that already-completed shard by exact
shard ID, outcome hash, and success-marker hash.

The shard may not be deleted or rerun.

No other V2 shard is authorized for reuse.

## Remaining production work

The Phase-5E estate contains 732 fault shards.

R5 freezes:

- 1 exact grandfathered transient shard;
- 731 remaining shards;
- V3 execution for all remaining shards;
- exact Phase-5E order;
- aggregation only after 732 total valid shard results.

## Scientific invariance

R5 changes no threshold, comparator, event metric, paired-clean rule,
aggregation rule, uncertainty rule, target/family stratum, checkpoint
membership, fault identity, or fault membership.

Only C0 and CC remain authorized.

CSC and OnField remain excluded.

No model forward or new fault execution is authorized.

## Resume provenance

Production continuation must create a separate R5/V3 continuation marker
binding:

- R5 authorization hash;
- exact V3 executor hash;
- R4 manifest hash;
- original execution-start hash;
- exact grandfathered shard hashes.

Before processing the remaining shards, the grandfathered shard must be
revalidated byte-for-byte.

## R5 boundary

R5 itself does not resume production.

It does not read accepted prediction payloads, load outer labels, apply
thresholds, compute CC metrics, or generate aggregate outcomes.

The next step is governed production continuation for the remaining 731
shards.
