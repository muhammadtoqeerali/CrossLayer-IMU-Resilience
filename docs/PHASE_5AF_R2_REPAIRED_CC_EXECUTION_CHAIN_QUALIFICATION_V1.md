# Phase 5AF-R2 — Repaired CC Execution Chain Qualification V1

## Status

**QUALIFIED_POST_BOUNDARY_REPAIRED_CC_EXECUTION_CHAIN**

## Context

The authorized Phase-5AF outcome run crossed the one-way outer-outcome
boundary and stopped before completing any shard because the original
analysis consumer could not decode the already-frozen producer serialization
for non-finite softmax values.

Phase 5AF-R1 created and qualified a versioned V2 analyzer that restores the
exact Phase-5M/5R serialized IEEE values.

Phase 5AF-R2 closes the remaining implementation dependency by versioning the
I/O runner and final executor so the entire repaired execution chain is bound
to that qualified V2 analyzer.

## Repaired chain

The qualified continuation chain is:

`cc_outcome_analyzer_v2.py`
→ `cc_outcome_io_runner_v2.py`
→ `cc_outcome_executor_v2.py`

The historical V1 analyzer, V1 I/O runner, and V1 final executor remain
byte-identical.

## V2 I/O runner

The V2 I/O runner preserves the V1 filesystem, integrity, label-join, timing,
and scenario-reconstruction behavior.

Its only implementation delta is binding the existing analyzer imports to
`cc_outcome_analyzer_v2`.

## V2 final executor

The V2 final executor preserves the existing scientific execution surface and
binds:

- the V2 analyzer;
- the V2 I/O runner;
- an explicit post-boundary repair-binding artifact.

That repair binding independently hash-binds the versioned implementation
lineage without changing the original frozen pre-outcome gate.

## Scientific invariance

No scientific rule changes.

The repair does not alter:

- validation-selected thresholds;
- comparison semantics;
- outcome metrics;
- paired-clean counterfactual definition;
- aggregation rules;
- uncertainty rules;
- target/family strata;
- checkpoint membership;
- fault membership;
- transient/persistent separation;
- FP32/PTQ separation;
- CSC exclusion;
- OnField exclusion.

## Qualification

The repaired chain passed seven named synthetic qualification checks:

1. exact repaired-module hash binding;
2. transient dict-valued NaN end-to-end handling;
3. persistent dict-valued NaN end-to-end handling;
4. non-finite fault accounting;
5. all three frozen operating points;
6. paired clean/faulted metric construction;
7. finite reconstruction semantics preservation.

The initial R2 qualifier expected eight checks while exactly seven named
checks were recorded. Only that bookkeeping expectation was corrected from
eight to seven.

The V2 analyzer, V2 I/O runner, and V2 executor remained byte-identical during
that focused repair.

## Boundary

Phase 5AF has not resumed.

No accepted threshold was applied during R2 qualification.

No accepted CC metric was computed.

R2 itself does not authorize production continuation.

A separate post-boundary continuation authorization must bind the exact V2
analyzer, V2 I/O runner, V2 executor, R1 evidence, R2 evidence, and unchanged
scientific protocol before the accepted run resumes.
