# Phase 4H Outer Subject Aggregation v1r1

Status: **technical repair of frozen v1; no scientific change**.

The frozen v1 runner failed before producing a final result because the
immutable executor serializes explicitly undefined metrics as JSON `null`,
while the already-qualified aggregation helper executes `float(value)` and
therefore cannot ingest Python `None`.

A read-only diagnostic established that:

- every `null` was explicitly named in `undefined_metric_fields`;
- no finite value was incorrectly declared undefined;
- the qualified helper already preserves `NaN` as missing and excludes it
  from finite equal-weight means;
- no performance value was printed or interpreted.

v1r1 therefore makes exactly one representation repair:

`JSON null / Python None -> floating NaN`

immediately before invoking the unchanged qualified aggregation helper.

There is no conversion to zero and no change to metric definitions,
replicate/variant/seed weighting, subject inference, fault selection,
severity selection, operating points, model selection, thresholds,
bootstrap rules, direction rules, or reporting policy.
