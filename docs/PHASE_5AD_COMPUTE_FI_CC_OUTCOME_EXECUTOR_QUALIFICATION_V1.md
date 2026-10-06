# Phase 5AD — Final CC Outcome Executor Qualification V1

## Status

**QUALIFIED_FINAL_CC_OUTCOME_EXECUTOR_PRE_OUTCOME**

## Scope

Phase 5AD implements the final end-to-end compute-only outcome executor and
qualifies it entirely on synthetic accepted-style estates.

No accepted Phase-5 outer prediction or outer label payload is used.

## Production binding

The executor has two explicit modes:

- synthetic qualification mode;
- production mode.

Production mode requires exact agreement with the Phase-5AC gate for:

- frozen Phase-5Z protocol;
- qualified Phase-5AA analyzer;
- qualified Phase-5AB I/O runner;
- exact Phase-5E plan hash;
- accepted outer result-root string;
- accepted outer dataset-root string.

Path mismatches are rejected before artifact access.

Qualification mode explicitly rejects the Phase-5AC accepted outer and dataset
roots.

## End-to-end shard analysis

For one frozen fault shard, the executor:

1. resolves its exact Phase-5E shard and clean-cache records;
2. verifies clean and fault artifacts before parsing JSONL;
3. checks planned clean/fault record cardinality;
4. loads the subject's frozen trial inventory;
5. reconstructs every clean trial in exact window order;
6. loads trial-local labels with exact window-count checks;
7. joins frozen risk/timing records;
8. groups fault rows by `outer_instance_id`;
9. validates parent-trial and shard identity consistency;
10. reconstructs one CC scenario per fault identity;
11. preserves fault-family / representation / target strata;
12. applies all three frozen validation operating points;
13. computes paired-clean and faulted historical event metrics;
14. computes predeclared paired degradation;
15. records non-finite fault counts.

Transient and persistent scenarios remain separate.

FP32 and PTQ remain separate.

Different `outer_instance_id` values are never combined into one scenario.

## Aggregation

The executor also provides final subject-primary aggregation.

For each metric-specific target stratum:

- the three frozen checkpoint seeds receive equal weight inside each subject;
- subjects then receive equal weight;
- primary uncertainty is the frozen 10,000-replicate subject-cluster
  percentile bootstrap.

Clean unique-trial baselines are deduplicated across transient/persistent shard
reuse before aggregation.

Timing values that cannot be computed are not imputed. A stratum containing a
missing/non-finite subject-seed timing value is explicitly marked unavailable
without imputation.

## Qualification estate

Qualification constructs temporary synthetic accepted-style estates spanning:

- two subjects;
- three checkpoint seeds;
- transient and persistent faults;
- all three operating points;
- clean caches shared by persistence;
- Phase-5R-style integrity verification;
- Phase-5M-style externally frozen exact hashes;
- trial-local `labels.npy`;
- risk/timing CSV.

The synthetic jobs exercise paired-clean metrics, faulted metrics, degradation,
non-finite accounting, equal-seed aggregation, subject-cluster bootstrap, and
clean-baseline deduplication.

## Boundary

At Phase-5AD completion:

- final outcome executor implemented: true;
- final outcome executor qualified: true;
- production execution performed: false;
- accepted outer root accessed: false;
- accepted outer root stat/list performed: false;
- accepted prediction JSONL opened: false;
- accepted outer label array loaded: false;
- accepted prediction deserialized: false;
- accepted threshold applied: false;
- accepted CC metric computed: false;
- aggregate accepted CC result generated: false;
- CSC result generated: false;
- model loaded/forwarded: false;
- fault execution: false;
- OnField used: false.

A final one-way activation artifact binding the exact Phase-5AD executor hash
is required before the first accepted outcome payload may be opened.
