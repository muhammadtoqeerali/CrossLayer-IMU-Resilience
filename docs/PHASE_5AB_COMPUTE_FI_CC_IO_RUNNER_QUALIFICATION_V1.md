# Phase 5AB — Prospective CC I/O Runner Qualification V1

## Status

**QUALIFIED_PROSPECTIVE_CC_IO_RUNNER_PRE_OUTCOME**

## Purpose

Phase 5AB implements and qualifies the filesystem/I/O boundary required to
apply the frozen Phase-5Z compute-only (`CC`) outcome-analysis protocol using
the qualified Phase-5AA analyzer.

Qualification remains pre-outcome.

The accepted prospective outer result estate is not accessed during
qualification.

## Qualified integrity behavior

### Phase-5R artifacts

Before any record JSONL may be parsed, the runner requires:

- `_SUCCESS.json`;
- `metadata.json`;
- the expected record file;
- `status == PASS`;
- the expected frozen plan SHA;
- the expected executor SHA;
- metadata SHA matching the success marker;
- record-file SHA matching the success marker.

Only after those integrity checks pass may record JSON be deserialized.

### Accepted Phase-5M canary

The earlier accepted canary is handled through exact externally frozen hashes
for:

- `_SUCCESS.json`;
- `metadata.json`;
- the record file.

The historical canary is not rewritten into the later Phase-5R success
schema.

## Qualified adapters

Phase 5AB qualifies:

- complete clean-trial reconstruction from `parent.window_index`;
- fatal rejection of duplicate or missing clean windows;
- grouping fault rows by `outer_instance_id`;
- Phase-5M transient execution-window recovery from `parent.window_index`;
- Phase-5R persistent execution-window recovery from
  `execution_window_index`;
- delegation of persistent onset-to-end suffix integrity to the qualified
  Phase-5AA analyzer.

## Labels

The prospective runner contains a trial-local `labels.npy` loader.

Qualification uses only a temporary synthetic `labels.npy`.

The label length must exactly equal the frozen trial window count.

Labels are normalized with the historical-equivalent Phase-5AA rule:

- Activity = 0;
- Falling = 1.

## Risk / timing join

The runner provides a frozen risk-index CSV loader keyed by:

`(subject, task, trial)`.

The event-row adapter preserves the historical timing reconstruction:

- sampling rate 100 Hz;
- window length 30 samples;
- stride 15 samples;
- Activity windows first;
- Falling windows anchored to `fall_start_position`;
- true fall if Falling labels exist or a frozen risk record exists.

This preserves the historical `KFALL_106_T27_R05` truth exception without
rewriting its stored labels.

## Qualification fixtures

Qualification uses only a temporary synthetic filesystem containing:

- synthetic clean JSONL;
- synthetic transient fault JSONL;
- synthetic persistent fault JSONL;
- synthetic success markers;
- synthetic metadata;
- synthetic `labels.npy`;
- synthetic risk CSV.

No accepted prospective artifact is used.

## Integrity ordering

The qualification explicitly proves hash-before-parse behavior.

One synthetic JSONL is deliberately malformed and also deliberately assigned
an incorrect expected hash. The runner must fail on the hash mismatch before
JSON parsing.

A separate fixture has a correct file hash but malformed JSON. That artifact
must pass integrity verification and then fail during JSON parsing.

## Check-count repair

The first synthetic qualification invocation successfully reached the final
bookkeeping assertion but expected 13 recorded checks.

The qualifier actually records 12 named checks:

- clean sequence join;
- historical fallback truth rule;
- historical window-end reconstruction;
- label-length guard;
- malformed-JSON abort;
- Phase-5M frozen-hash adapter;
- Phase-5M transient adapter;
- Phase-5R clean hash-before-parse;
- Phase-5R persistent adapter;
- risk-index join;
- stored-label normalization;
- tamper abort before parse.

Only the expected bookkeeping count was corrected from 13 to 12.

The I/O runner remained byte-identical and no scientific semantics changed.

## Boundary

At Phase-5AB completion:

- prospective CC I/O runner implemented: true;
- prospective CC I/O runner qualified: true;
- qualification filesystem synthetic only: true;
- accepted outer result root accessed: false;
- accepted clean JSONL opened: false;
- accepted fault JSONL opened: false;
- accepted outer label array loaded: false;
- outer prediction deserialized: false;
- outer threshold applied: false;
- outer CC metric computed: false;
- aggregate CC result generated: false;
- CSC result generated: false;
- model loaded: false;
- model forward executed: false;
- fault execution executed: false;
- OnField used: false.

The next step is a separate prospective CC outcome execution gate binding the
frozen Phase-5Z protocol, qualified Phase-5AA analyzer, and qualified Phase-5AB
I/O runner before accepted prediction or label payloads are opened.
