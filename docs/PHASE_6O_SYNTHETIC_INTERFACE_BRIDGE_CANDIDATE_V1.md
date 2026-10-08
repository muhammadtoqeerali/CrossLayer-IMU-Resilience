# Phase6O synthetic CSC execution-interface bridge candidate v1

## Purpose

Move beyond metadata-only request and stream qualification by
exercising the real frozen Phase6J interface chain with synthetic
sensor operators, synthetic rewindowing, synthetic model-bundle
placeholders, and a synthetic Phase5 executor.

The qualification uses 37 existing representative frozen CSC pairs,
expanded to exactly 189 validated Phase6G requests and organized
into 69 model-member streams.

## Frozen interfaces exercised

1. condition_sensor_parent — the actual frozen sensor-parent
   conditioning control flow, with fixture-only operator, exposure
   and runner dependencies.
2. synthetic_execute_pair_member — the actual frozen sensor-reference
   and compute-fault record assembly and call sequencing.
3. execute_bound_compute_fault_sequence — the actual frozen
   Phase5-executor routing and active-mask validation, but with a
   synthetic executor operating entirely on string tokens.
4. run_model_member_stream — the actual frozen stream grouping,
   per-stream synthetic bundle load, and release lifecycle.

The frozen Phase6G builder and validator independently verify each
request's model eligibility, seed, persistence, target routing,
temporal overlap, and request identity.

## Scientific boundaries

The synthetic fixture returns its previously frozen expected exposure
indices. It does not prove actual Phase6I exposure calculation.

The synthetic sensor operator performs string manipulation only. It
does not apply real sensor corruption.

The synthetic Phase5 executor performs no PyTorch tensor conversion,
no model forward, no mutation, and no real persistent-weight reset.

The record constructors receive artificial output summaries. Those
rows do not represent predictions, probabilities or scientific CSC
results and are never written as Phase6H outputs.

A synthetic release callback does not establish real CUDA/torch
resource cleanup, PTQ restoration, or production model lifecycle.

## Provenance and output boundary

This candidate adds new source, tests, documentation and a byte-exact
copy of the prior Phase6O static interface survey. The new synthetic
qualification report is stored outside the repository until separately
reviewed.

Frozen Phase6J, Phase6K, Phase6L, Phase6M and Phase6N files
remain byte-identical.

No Phase6H _SUCCESS marker or actual CSC artifact is created.
The original Phase6E digest serialization is not reproduced.

## Next implementation

After this synthetic bridge is qualified, a separate prospective
execution body still needs independently pinned real loaders,
Phase4H runtime dependencies, Phase5 model/injection/cleanup
qualification, bounded Phase6H output lifecycle and an explicit
reviewed CSC canary execution gate.

The frozen Phase6J execute_shard entrypoint remains blocked.
The new Phase6O execute_shard entrypoint is also unconditionally
blocked.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
