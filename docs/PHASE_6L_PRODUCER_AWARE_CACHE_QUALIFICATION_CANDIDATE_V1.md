# Phase6L producer-aware cache qualification candidate v1

## Verified prospective scope

Two separately recorded metadata-only Phase6L validation audits passed for
all 366 frozen Phase5 clean caches. The accepted Phase5M canary accounts for
one clean cache and Phase5R full fleet for the remaining 365.

Both audits used the frozen Phase6J clean-cache resolver and Phase5 success
marker validator. Across each audit, 732 output-file SHA-256 checks covered
366 clean-cache artifacts and 1,642,980 clean-cache window evaluations.
The audits did not parse prediction rows or execute model forwards.

Independent reconciliation confirms identical 366 cache identities,
producer hashes, window counts, and six model-variant/seed cache identities
for each of the 61 subjects.

## Difference between v1 and v2

Phase6L v1 delegates to a caller-provided resolver and validator core.
The v1 audit independently verified their frozen source identity, but
the v1 API itself does not enforce caller-supplied validator provenance.

Phase6L v2 removes caller-supplied resolver and core parameters. It checks
the expected module source paths, file SHA-256 digests, module ownership,
and function code origin before invoking the frozen validators. Tests
reject ordinary monkeypatch replacement of the resolver, core validator,
and v1 delegate. These checks do not amount to a comprehensive proof
against sophisticated in-memory code substitution.

## Unresolved scientific and execution boundaries

The original Phase6E historical digest serialization remains unresolved.
Exact workload numerical agreement does not reproduce those original
historical stream digests.

Neither Phase6L adapter modifies or completes the frozen Phase6J execution
body. Neither version is an execution entrypoint, nor does this candidate
qualify a real CSC integration, authorize model loading, run CSC faults,
or permit forward evaluation.

The prior Phase6K evidence freeze is unchanged.

## Evidence governance

The two JSON reports in the adjacent evidence directory are preserved
byte-for-byte, each verified against its original SHA-256 digest.
The qualification candidate binds both reports, both adapter versions,
their tests, and the relevant frozen dependencies.

The known Phase5R historical canary-only regression assertion is explicitly
deselected from the wider suite; it is not edited or silently suppressed.

This remains an uncommitted candidate, not an execution authorization gate.
A separate formal qualification freeze and Git review are required.
