# Phase 5AA — Compute-Fault CC Analyzer Core Qualification V1

## Status

**QUALIFIED_CC_ANALYZER_CORE_PRE_OUTCOME**

## Purpose

Phase 5AA implements and qualifies the pure analysis core required by the
frozen Phase-5Z prospective compute-only (`CC`) outcome-analysis protocol.

Qualification is strictly pre-outcome.

No prospective Phase-5 prediction JSONL is opened, no outer-test label array
is loaded, no threshold is applied to an outer prediction, and no CC outcome
metric is computed.

## Qualified behavior

The analyzer core qualifies all of the following using synthetic fixtures and
frozen-source equivalence checks:

- exact 45-row frozen threshold lookup;
- historical Activity/Falling label normalization;
- direct Falling-probability extraction from the stored two-class softmax;
- exact threshold comparator and consecutive-trigger behavior;
- first-valid-trigger semantics;
- exact historical event-metric semantics;
- Phase-5M transient-record reconstruction using `parent.window_index`;
- Phase-5R persistent reconstruction using `execution_window_index`;
- exact persistent onset-to-trial-end suffix coverage;
- fatal rejection of persistent gaps;
- fatal rejection of duplicate execution-window indices;
- paired clean counterfactual construction;
- paired degradation sign convention;
- equal weighting of seeds 42, 123, and 2025 inside a subject;
- equal subject weighting;
- deterministic subject-cluster percentile bootstrap;
- explicit non-finite fault-output counting.

## Historical evaluator equivalence

The qualification compares the Phase-5AA implementation against the frozen
historical evaluator for:

- label normalization;
- trigger episodes;
- first valid trigger;
- event metrics.

The historical evaluator itself is hash-bound through the Phase-5AA
qualification configuration.

## Floating-point qualification repair

The first qualification invocation reached the paired-degradation synthetic
check and failed because the test compared `0.9 - 0.7` to decimal `0.2` using
exact Python floating-point equality.

The analyzer implementation was not modified.

Only that synthetic qualifier assertion was changed to an absolute-tolerance
comparison using `math.isclose(..., rel_tol=0, abs_tol=1e-12)`, consistent
with the qualifier's existing numerical-equivalence policy.

After that focused repair:

- the analyzer remained byte-identical;
- all 16 qualification checks passed;
- no prospective outer payload was accessed;
- no scientific parameter changed.

## Mixed Phase-5M / Phase-5R provenance

Both accepted record formats expose:

- `clean_softmax_values`;
- `faulted_softmax_values`.

Falling probability is element 1.

No logits reconstruction or canary-specific approximation is needed.

## Persistent-fault integrity

A persistent scenario is valid only when one `outer_instance_id` contains one
record for every execution window from the frozen onset through the end of the
parent trial.

Missing suffix rows, duplicate execution indices, inconsistent onset values,
or out-of-range execution indices are fatal.

## Statistical primitives

The generic qualified aggregation primitive requires exactly the three frozen
checkpoint seeds per subject.

Seed values receive equal weight within subject.

Subject estimates then receive equal weight.

The subject-cluster percentile bootstrap is deterministic for a frozen RNG
seed.

Non-finite subject-level scalar values are rejected rather than silently
dropped or imputed. Metric-specific missing-timing handling remains an
explicit responsibility of the later protocol-qualified prospective runner.

## Qualification boundary

The qualified Phase-5AA module is a pure analysis core.

It does not implement prospective artifact I/O.

It contains no prospective result-root path, prediction JSONL reader, label
array loader, model loader, model forward, or fault execution.

A separate prospective CC I/O runner must be implemented and statically
qualified before accepted outer predictions or labels may be opened.

At Phase-5AA completion:

- prospective outer result root accessed: false;
- clean prediction JSONL opened: false;
- fault prediction JSONL opened: false;
- prediction JSON deserialized: false;
- outer label array loaded: false;
- threshold applied to outer: false;
- outer CC metric computed: false;
- aggregate CC result generated: false;
- CSC result generated: false;
- model loaded: false;
- model forward executed: false;
- fault execution executed: false;
- OnField used: false.
