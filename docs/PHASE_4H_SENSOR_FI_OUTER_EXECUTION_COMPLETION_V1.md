# Phase 4H Sensor-FI Outer Execution Completion v1

Status: **COMPLETE_IMMUTABLE_OUTER_EXECUTION**

This record freezes completion evidence for the Phase 4H held-out
sensor-fault execution before statistical aggregation or robustness
reporting.

## Frozen scope

- Partition: outer test only.
- Regimes executed: C0 and CS.
- Models: prospective FP32 300 ms and qualified static PTQ v7.
- Operating points: balanced, low_false_alarm, timely_150ms.
- OnField was not used.
- Outer outcomes were not used for model, threshold, severity, fault,
  protection, or reporting-rule selection.

## Completed execution

- Shards: 793
- Subjects: 61
- Subject-condition rows: 320616
- Unique fault instances: 42766632
- Model-window evaluations: 479750160
- C0 shards: 61
- CS shards: 732
- Fold subject counts: {1: 13, 2: 12, 3: 12, 4: 12, 5: 12}
- Dataset subject counts: {'KFALL': 32, 'UNIVR': 29}

The frozen plan duplicates its aggregate expectations at
`$.expected_totals_from_shards`; independently summing the immutable
793 shard records gives the same totals.

## Integrity

Every shard has a PASS success marker. Every output hash declared by
each marker was reverified. No temporary shard directories remain and
no raw model probabilities were persisted.

Success-marker index digest:

`0683f407f68e6a9c1f1a3a93e5a2a6a5c2387cdeeea867c859dacc440b017dee`

Because each success marker contains the hashes of the corresponding
coverage and subject-condition files, this digest binds the complete
execution set without copying performance values into this record.

## Operational recovery

Plan index 292
`p4h-o2-f2-s120-stuck_channel-3dc8bd0e9771` was recovered from an empty stale temporary directory
using the already-frozen executor option `--recompute-partial`.

Final success-marker SHA256:

`bf75fdd20e3b1d398b0d6e6293663dd354d8c9510ed59ee52918f6927568e49e`

This was an operational resume event only. It did not modify scientific
parameters and no performance value was used to choose or gate the
recovery.

## Next authorized step

Frozen aggregation is authorized. Statistical results must be produced
using the already-qualified aggregation/reporting contracts. Outer
results remain prohibited as a source of prospective retuning.
