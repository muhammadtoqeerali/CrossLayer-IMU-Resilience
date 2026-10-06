# Phase 5P — Prospective Full-Fleet Completion Authorization v1

**Status:** FROZEN_PROSPECTIVE_FULL_FLEET_COMPLETION_AUTHORIZATION  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

Phase 5P freezes authorization to complete the exact prospective Phase-5E
outer compute-fault estate after the single Phase-5N canary passed Phase-5O
technical acceptance.

This phase does not execute an additional outer shard.

## Authorization basis

Authorization is based only on Phase-5O technical integrity evidence.

Phase-5O did not deserialize prediction rows for acceptance and did not use:

- prediction outcomes;
- labels;
- OnField;
- thresholds;
- metrics;
- fault effects;
- scientific performance.

Therefore Phase-5P authorization cannot constitute outcome-based shard or model
selection.

## Already satisfied estate

The accepted canary already satisfies:

- 1 fault shard;
- 1 clean cache;
- 24,810 outer execution instances;
- 2,481 clean forwards;
- 24,810 faulted forwards;
- 27,291 total model-window evaluations.

Those artifacts must be reused after validating their success-marker and
output-file hashes.

They must not be rerun merely because full-fleet completion is later enabled.

## Remaining authorized estate

The remaining authorization is exactly the Phase-5E plan minus the accepted
canary:

- 731 fault shards;
- 365 clean caches;
- 20,145,198 outer execution instances;
- 1,640,499 clean model-window evaluations;
- 29,774,010 faulted model-window evaluations;
- 31,414,509 remaining model-window evaluations.

No subset was selected by canary outcome.

## Complete estate

Completion produces exactly the original frozen Phase-5E estate:

- 732 fault shards;
- 366 clean caches;
- 61 subjects;
- all 5 folds;
- seeds 42, 123, and 2025;
- FP32 and PTQ-v7;
- transient and persistent fault modes;
- 20,170,008 outer execution identities;
- 1,642,980 clean model-window evaluations;
- 29,798,820 faulted model-window evaluations;
- 31,441,800 total model-window evaluations.

## Execution architecture requirement

Phase 5P authorizes completion but does not itself provide a fleet executor.

Before another outer shard may execute, a new fleet executor must be
implemented and qualified.

That executor must bind the Phase-5P gate SHA and the unchanged Phase-5E plan.

It must support both model variants and both persistence modes while reusing
the already accepted canary artifacts.

## Required mechanics

The fleet executor must preserve:

- Phase-5D deterministic sampling and identity derivation;
- Phase-5A mutation-specification `fault_id`;
- parent-bound `sampling_instance_id`;
- parent/variant/seed-bound `outer_instance_id`;
- Phase-5K fault-only execution semantics;
- Phase-5F atomic/resume semantics;
- exact forward accounting;
- PTQ persistent-weight reset at trial boundaries;
- no cross-trial persistent state.

## Prohibited adaptation

Full-fleet authorization does not permit changing:

- seed;
- fold;
- shard;
- target;
- element;
- bit;
- onset;
- persistence;
- threshold;
- operating point;
- multiplicity;
- replicate.

No best/worst member selection is permitted.

Outer outcomes cannot reconfigure the estate.

## Scientific boundary

No additional outer payload is read in Phase 5P.

No additional model forward or fault execution occurs.

Only one fault shard remains executed at this freeze.

No aggregate CC result exists yet.

CSC remains Phase 6.

No commit or push is part of this freeze.
