# Phase 6D — Prospective CSC Pairing Protocol V1

## Status

**FROZEN_PROSPECTIVE_CSC_PAIRING_PROTOCOL_PRE_EXECUTION_PLAN**

This protocol defines the prospective P0 combined-fault regime:

- **C0**: clean sensor + clean compute
- **CS**: sensor fault + clean compute
- **CC**: clean sensor + compute fault
- **CSC**: sensor fault + compute fault

No CSC model execution is authorized by this step.

## Scientific rule

CSC is constructed causally:

1. reconstruct the exact selected frozen Phase-4H sensor corruption;
2. feed that corrupted parent through the same frozen model/preprocessing path;
3. inject the exact paired Phase-5 compute fault into the inference generated
   from that sensor-corrupted parent.

Cross-subject, cross-fold, and cross-trial pairing is forbidden.

## Sensor-side sparse prospective selection

The Phase-4H sensor universe remains unchanged:

- 12 sensor fault families;
- L1/L2/L3 severity;
- exact frozen target, timing, variant, replicate, and replay semantics.

For CSC, exactly one already-frozen sensor fault instance is selected per:

`outer parent × sensor family × severity`

using an immutable SHA256 hash-min rule.

This is a prospective computational subsample, not performance-driven
selection.

The exact pre-compute selected-sensor cardinality is:

- 4,107,450 stored-window-parent instances;
- 132,489 source-trial-parent instances;
- **4,239,939 total selected sensor instances**.

## Compute-side assignment

The full frozen Phase-5 compute surface is preserved:

- 14 targets;
- 5 compute fault families;
- 10 FP32/PTQ-common targets;
- 4 PTQ-only targets;
- transient one-inference persistence;
- persistent-from-onset-until-trial-end persistence.

Each selected sensor instance is assigned exactly one of the 28
`target × persistence` compute strata using a prospective SHA256 modulo-28
mapping.

Counts are not rebalanced after they are observed.

All 28 strata must have non-zero coverage in the derived execution plan or
plan derivation aborts.

## Exact CC coordinate reuse

The Phase-5 compute sampling coordinate is reused rather than resampled.

For transient compute faults:

- a stored-window sensor fault uses the same evaluation window;
- a source-trial sensor fault deterministically chooses one sensor-exposed
  evaluation window using the frozen Phase-4H parent-to-window mapping.

For persistent compute faults:

- the exact frozen Phase-5 persistent identity and onset for the same trial and
  target are reused;
- temporal overlap with the sensor fault is recorded but is not used to select
  or reject the pair.

This preserves a valid exact CC comparator.

## Pair identity

A model-independent `csc_sampling_pair_id` binds:

- sensor fault ID;
- sensor replay ID;
- Phase-5 sampling-instance ID.

A model-specific `csc_outer_instance_id` additionally binds:

- model variant;
- checkpoint seed.

## Comparators

Each CSC pair retains exact references for:

- C0: neither fault;
- CS: same sensor fault only;
- CC: same compute fault only;
- CSC: both faults.

The same subject, fold, trial, checkpoint, preprocessing, and validation-selected
operating-point threshold must be used.

## Governance

The outer subject remains the primary uncertainty unit.

Fault pairs are not independent subjects.

No outer-result-based model, sensor family, severity, compute target, bit,
persistence, checkpoint, threshold, or pair selection is permitted.

This phase makes no OnField, MCU-equivalence, physical-realism, hardware-rate,
P2, or P3 claim.

## Next step

Phase 6E must derive and freeze the exact pair inventory, stratum counts,
temporal-overlap counts, sharding, and model-forward workload before any CSC
outer model execution.
