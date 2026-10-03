# Phase 4H — Outer Execution Plan Qualification

**Status:** QUALIFIED_PRE_OUTER_EXECUTION_PLAN

The complete held-out P0 C0/CS execution matrix is now frozen before any
outer model prediction or outer fault result.

Static qualification verified:

- 61 unique outer subjects in one fold each;
- 6,309 trials and 273,830 windows;
- 42,766,632 unique sensor-fault instances;
- 479,750,160 model-window evaluations;
- 793 deterministic subject × execution-block shards;
- all three checkpoint seeds and both model variants inside every shard;
- all three frozen operating points retained and reported separately;
- all 12 families, all severities, all variants, and all stochastic
  replicates preserved;
- compact subject-level sufficient-statistics output rather than raw
  hundreds-of-millions-row probability persistence;
- immutable/hash-guarded resume semantics.

No model was loaded, no model prediction was computed, and no sensor
fault was executed during planning or qualification.

The next step is to implement/freeze the executor against this exact
plan before executing any held-out shard.
