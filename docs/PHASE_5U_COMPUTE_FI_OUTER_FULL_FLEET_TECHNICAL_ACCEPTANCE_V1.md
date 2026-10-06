# Phase 5U — Complete Outer Estate Technical Acceptance v1

**Status:** TECHNICALLY_ACCEPTED_COMPLETE_PHASE5E_ESTATE  
**Evidence tier:** P0  
**Date:** 2026-10-06

## Purpose

Phase 5U freezes technical acceptance of the complete prospective Phase-5
compute-fault outer estate.

This is a structural and cryptographic acceptance only.

It does not interpret prediction outcomes and does not compute CC metrics.

## Accepted estate

The accepted estate is exactly the frozen Phase-5E estate:

- 732 fault shards;
- 366 clean caches;
- 1,098 artifact directories;
- 20,170,008 outer execution identities;
- 1,642,980 clean model-window records;
- 29,798,820 faulted model-window records;
- 31,441,800 total model-window records.

## Integrity verification

Every Phase-5R-produced artifact is checked for:

- atomic `_SUCCESS.json`;
- `metadata.json`;
- expected JSONL output;
- frozen Phase-5E plan SHA;
- exact Phase-5R executor SHA;
- output hashes matching `_SUCCESS.json`;
- expected record cardinality;
- expected outer-identity cardinality;
- no labels;
- no OnField;
- no threshold selection;
- no metric generation;
- no embedded clean-reference forward in the fault loop.

The accepted Phase-5N canary artifacts are reverified against the frozen
Phase-5O technical-acceptance hashes rather than requiring the Phase-5R
executor SHA.

## Prediction-row inspection boundary

Record JSONL files are read only as binary byte streams for:

- SHA256 verification;
- newline-based record counting;
- byte-size accounting.

Prediction rows are not JSON-deserialized.

No predicted class is counted.

No softmax value is analyzed.

No threshold is applied.

No label is read.

No CC outcome metric is computed.

## Scientific boundary

Phase 5U changes no:

- fault sampling;
- identity;
- target;
- bit;
- onset;
- persistence;
- Phase-5E plan;
- Phase-5P gate;
- Phase-5R executor.

No outcome-based adaptation occurs.

No aggregate CC result is generated.

CSC remains ungenerated.

## Next boundary

Before prediction rows are deserialized for scientific outcome analysis, a
separate prospective CC analysis protocol must be frozen.

That protocol must specify aggregation, subject/seed handling, operating-point
usage, uncertainty, failure metrics, timing metrics where valid, and reporting
rules before any CC outcome is inspected.

## Post-execution regression compatibility

One frozen Phase-5R regression is intentionally lifecycle-bound:

`tests/test_phase5r_compute_fi_outer_full_fleet_executor.py::test_outer_estate_remains_single_accepted_canary`

That test asserts that the live outer-result root contains only the accepted
single canary. This was correct when Phase 5R was frozen, before full-fleet
execution.

After successful Phase 5T execution, the live outer-result root is required to
contain the complete frozen Phase-5E estate: 732 fault shards and 366 clean
caches.

The historical Phase-5R test and its frozen manifest are not modified, because
Phase 5S cryptographically binds the Phase-5R qualification manifest.

For post-Phase5T/5U repository regression only, this single historical
filesystem-state assertion is explicitly deselected.

Its scientific and implementation checks remain covered by the rest of the
frozen Phase-5R suite, while Phase-5U regression verifies the complete current
estate.

The Phase-5U suite also verifies that the frozen Phase-5R regression-test file
still matches the SHA256 stored in its frozen Phase-5R manifest.

