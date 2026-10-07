# Phase 6D-R2 — CSC Structural-Omission Amendment V1

## Status

**FROZEN_PRE_PLAN_PROSPECTIVE_CSC_STRUCTURAL_OMISSION_AMENDMENT**

This is a prospective scientific amendment to the frozen Phase-6D CSC pairing
protocol and its R1 machine-exact clarification.

Unlike R1, `scientific_change = true`.

The amendment was qualified using training/calibration geometry only. No outer
performance outcome, validation result, OnField result, CSC model output, or
physical/MCU evidence was used.

## Why R2 is required

Phase 6D originally selected one frozen Phase-4H sensor instance for every
parent × sensor family × severity group before assigning a compute stratum.

The subsequent geometry-only feasibility work established that some frozen
source-trial sensor instances do not overlap any retained 300 ms evaluation
window. A transient compute fault cannot be assigned to a sensor-exposed
evaluation window in such a case.

A first prospective development/calibration repair attempted to restrict the
hash-min candidate pool to observable frozen instances while still requiring
one selected instance in every group. That repair failed: 2,250 of 235,641
training/calibration fold-parent × source-family × severity groups contained
no observable frozen candidate at all.

R2 therefore freezes explicit structural omission rather than fallback.

## Exact source-trial eligibility rule

For each source-trial parent × sensor family × severity group:

1. Generate only the exact frozen Phase-4H candidate instances.
2. Compute each candidate's active support using the frozen R1 rule.
3. Compute retained-window exposure using the frozen R1 30-sample historical
   window geometry and half-open overlap predicate.
4. A candidate is eligible iff it overlaps at least one retained evaluation
   window.
5. If the eligible set is nonempty, apply the unchanged Phase-6D SHA256
   hash-min ranking to that eligible set.
6. If the eligible set is empty, record
   `STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR`.
7. Only a retained sensor selection proceeds to the unchanged Phase-6D
   compute-stratum assignment.

There is no fallback, resampling, temporal relocation, replacement instance,
nearest-window substitution, or outcome-dependent reselection.

## Stored-window sensor faults

The five stored-window families are unchanged.

Their frozen Phase-6D selection and pairing rules remain authoritative.

## Interaction with R1

R1's historical-window geometry, sensor active-support rule, exposure
predicate, source-trial transient compute-window hash rule, and persistent
overlap rule remain unchanged.

For any **retained** source-trial CSC pair, a zero-length sensor-exposed window
set remains an internal consistency failure and must abort plan derivation with
no fallback.

R2 changes what happens earlier when the complete frozen candidate set for a
parent × family × severity group has no observable candidate: that group is
structurally ineligible, receives no compute-stratum assignment, and produces
no CSC pair.

## Training/calibration qualification

The authoritative development/calibration parent universe contains:

- 5,346 unique parents;
- 3,077 Activity-only parents;
- 2,269 parents containing Falling;
- 11,221 fold-parent memberships.

The exhaustive source-trial qualification contained 235,641
fold-parent × family × severity groups.

R2D retained 233,391 groups and identified 2,250 structurally ineligible
groups, an omission rate of 0.009548423237.

All 154,938 Activity-only groups remained eligible. The 2,250 omissions were
within the 80,703 Falling-containing groups.

The qualification retained:

- every fold × family × severity stratum;
- every source-family × severity surface;
- at least one source-trial family for every fold-parent × severity;
- Falling-event coverage for every source-family × severity cell.

The frozen R2D candidate-rule digest is:

`740b444fa8b9a08fbc5aa154b9f0ea2d4f77d38f33bd4921b363d49bf008b412`

## Scientific surface

R2 does not remove a sensor family or severity from the analysis surface.

All 12 sensor-fault families and all three severities remain part of the
prospective study. Structural missingness is reported explicitly and is never
imputed.

The primary uncertainty unit remains the subject. Fault pairs are not treated
as independent statistical samples.

## Pair-count consequence

The original Phase-6D nominal count of **4,239,939** selected sensor-compute
pairs is no longer authoritative after R2 because source-trial groups may be
structurally omitted.

No replacement pair is created for an omission.

The exact post-R2 pair inventory must be derived by a geometry-only outer
structural census after this freeze and before any CSC model execution.

## Governance

This amendment was frozen before any R2-based outer structural recheck, CSC
pair materialization, or CSC model forward.

No CS or CC performance outcome was used.

No threshold, checkpoint, sensor family, severity, compute target,
compute persistence mode, bit, or fault value was chosen from outer outcomes.

No OnField, physical-realism, or MCU-equivalence claim is made.

## Next

Run one read-only, geometry-only outer structural census under R2. That census
must derive the exact structural omissions and revised CSC pair count without
model execution, performance outcomes, fallback, resampling, or rebalancing.

Only after that census may Phase 6E freeze its exact execution plan.
