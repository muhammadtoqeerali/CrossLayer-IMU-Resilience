# Phase 5L — Hash-Bound Canary-Only Outer Execution Freeze v1

**Status:** FROZEN_HASH_BOUND_CANARY_ONLY_OUTER_EXECUTION_CONFIG  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5L freezes the exact authorization contract for the first prospective
outer compute-fault canary.

This phase does not read outer payloads and does not execute the canary.

## Target-schema repair

The initial Phase-5L freeze attempt assumed a non-existent
`model_applicability` field in the frozen Phase-5D target inventory.

The read-only schema probe showed that Phase-5D encodes applicability in the
`model_variants` array.

The corrected binding therefore authorizes exactly rows for which:

`"fp32" in target["model_variants"]`

This yields exactly the 10 common FP32 target strata already frozen in Phase 5D
and exactly matches the canary's frozen `target_count = 10`.

No sampling, identity, target, shard, or execution semantics changed.

## Frozen authorization chain

The canary configuration is cryptographically bound to:

- the frozen Phase-5D outer protocol;
- the frozen Phase-5E 732-shard plan;
- the qualified Phase-5F atomic/resume core;
- the frozen Phase-5I prospective execution gate;
- the Phase-5J deterministic canary binding;
- the Phase-5K fault-only execution module and qualification;
- the frozen FP32 checkpoint manifest.

## Authorized canary

Exactly one fault shard is authorized:

`p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9`

It is frozen Phase-5E shard index 0:

- subject 9;
- fold 5;
- FP32;
- checkpoint seed 42;
- transient one-inference persistence;
- 10 frozen FP32 target strata.

Exactly one clean cache is authorized:

`p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b`

Every other Phase-5E shard remains unauthorized by this configuration.

Full-fleet execution remains unauthorized.

## Forward budget

The canary is frozen to exactly:

- 2,481 clean model-window evaluations;
- 24,810 faulted model-window evaluations;
- 27,291 total model-window evaluations.

Clean forwards must be produced only by the separate clean-cache pass.

Fault forwards must use only the Phase-5K fault-only execution module.

No clean-reference forward may occur inside the fault loop.

## Selection governance

The canary is not selected by an outer result.

It is exactly the first shard in the previously frozen deterministic Phase-5E
ordering.

Canary outcomes may not alter seed, fold, shard, target, element, bit, onset,
persistence, thresholds, operating points, multiplicity, or replicate.

A successful canary does not itself authorize the remaining 731 shards.

A technical implementation failure must return to non-outer fixture validation
before retry.

## Freeze boundary

At this freeze point no outer payload, model load, model forward, fault
execution, prediction, CC result, or CSC result exists.
