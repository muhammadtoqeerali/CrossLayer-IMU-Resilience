# Phase 4H — Outer Execution v2: Historical Trial Truth

**Status:** FROZEN_PRE_OUTER_EXECUTION

Outer execution v1 is preserved unchanged.

## Single clarification

The v1 counts `3391 Activity / 2918 Falling` describe the retained
stored-label inventory only.

The recovered historical evaluator defines trial truth as:

`true_fall = (falling_count > 0) OR (risk record exists)`

Therefore the prospective outer classification denominator is:

- Activity: 3390 trials
- Falling: 2919 trials
- Total: 6309 trials

Exactly one trial differs between stored labels and historical truth:
`KFALL_106_T27_R05` in outer fold 4.

That trial has:

- zero retained Falling-labelled windows;
- one retained risk/event annotation;
- historical true event `FALLING`;
- frozen Phase-3 full-trial fallback lineage;
- no relabeling.

Because the historical trigger evaluator requires an entire trigger run
to lie in retained Falling labels for a true-fall trial, this exception
cannot produce a valid Falling trigger under the frozen retained labels.
It therefore contributes a false negative / missed annotated event under
every model, fault condition, and operating point, with sensor lead NA.

This behavior is frozen prospectively before any outer model prediction
or fault execution.

## Unchanged from v1

The clarification changes none of:

- outer subject/fold assignment;
- 793 subject × block shards;
- 12 fault families;
- L1/L2/L3 severities;
- variants or stochastic replicates;
- three model seeds;
- FP32/PTQ pairing;
- three frozen operating points;
- thresholds or required-consecutive rules;
- 42,766,632 unique fault instances;
- 479,750,160 model-window evaluations;
- 320,616 expected subject-condition rows;
- aggregation or reporting policies.

The next step is to implement and freeze the shard executor against this
v2 contract, followed by metadata/interface dry-run qualification only.
