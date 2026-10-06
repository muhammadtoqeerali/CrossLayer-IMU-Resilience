# Phase 5AJ — Phase-5 Compute-FI Final Synthesis V1

## Status

**PHASE5_COMPLETE_FROZEN**

Phase 5 is complete for the prospective compute-only C0/CC scope.

## Final estate

The accepted Phase-5 outcome estate contains:

- 61 outer subjects;
- 6,309 trials;
- 273,830 stored windows;
- 366 clean caches;
- 732 compute-fault shards;
- 20,170,008 outer fault identities;
- 29,798,820 fault records.

Production outcome analysis contains one exact grandfathered transient R3/V2
shard and 731 exact R5/V3 shards.

## Technical acceptance

Phase 5AG technically accepted the completed execution before scientific
interpretation.

## Scientific interpretation

The frozen paired-effect convention is:

**positive degradation means worse behavior under compute fault.**

Of 720 primary CC strata, 576 are available and 144
`median_trigger_lead_ms` strata are unavailable without imputation.

Of 30 clean aggregate rows, 24 are available and six median-trigger-lead rows
are unavailable without imputation.

No unavailable timing value was dropped or imputed.

## Scientific findings

Persistent faults have larger descriptive degradation than transient faults in
all 24 available matched equal-target variant × operating-point × metric
comparisons.

On the ten targets common to FP32 and PTQ:

- PTQ has lower descriptive degradation in 21 of 24 available comparisons;
- PTQ has higher descriptive degradation in three of 24;
- six median-lead comparisons are unavailable.

Therefore Phase 5 makes no universal FP32/PTQ superiority claim and selects no
model variant from the outer results.

## Governance

Outer results cannot be used to select or retune model variant, target, fault
family, operating point, checkpoint, threshold, fault membership, or Phase-5
protocol rules.

Phase 5 makes no CSC claim, no OnField fall-performance claim, and no MCU or
physical-realism claim.

## Completion boundary

Phase-5AJ performs no new scientific outcome analysis, bootstrap, inferential
CI, significance test, model forward, or fault execution.

After successful Phase-5AJ regression, the accumulated uncommitted Phase-5
estate is eligible for a single major Phase-5 completion commit and push.
