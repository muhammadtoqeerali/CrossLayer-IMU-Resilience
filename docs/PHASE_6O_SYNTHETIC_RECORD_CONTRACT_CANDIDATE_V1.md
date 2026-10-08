# Phase6O synthetic frozen record-contract candidate v1

## Objective

Qualify the structure and relational semantics of Phase6H CSC
output records before prospective production output integration.

This work is strictly additive, and preserves all previously frozen
Phase6J, Phase6K, Phase6L, Phase6M, and Phase6N evidence.

## Qualified synthetic record surface

The existing 37-pair Phase6M witness is regenerated from frozen
metadata and expanded to 189 validated Phase6G requests.

The actual frozen Phase6J synthetic record constructors produce
the pair-member, sensor-reference, and CSC-fault-window rows.

The new validator checks required fields against the frozen Phase6H
architecture configuration and frozen Phase6J record definitions.

Per-request checks cover parent and model identity, clean-cache ID,
sensor-reference cache ID, expected row counts, window indices,
sensor/compute temporal overlap and reference-kind provenance.

All three JSONL record sets are written in the frozen canonical JSON
encoding, re-read, and revalidated in an automatically deleted
temporary directory.

The fourth file is an explicitly synthetic qualification-only
metadata.json file, not real production metadata.

No Phase6H _SUCCESS marker is produced by this record-contract stage.

## Deliberate limitations

All output summaries contain synthetic, non-scientific values.
This work does not validate real float32 model output, probability
values, genuine input/output SHA-256 payloads, or PTQ state.

The 189 requests are representative only; they are not the complete
21,793,038 planned variant/seed pair-members.

These 189 record bundles are not one complete 732-shard fleet.
Therefore the full per-shard frozen coverage contract has not been
qualified by this stage.

The frozen success-marker validator checks marker identity and
output-file hashes. A prospective production readiness check must
also independently enforce exact frozen coverage/cardinality fields.

The complete Phase6H output schema and genuine runtime loading,
model inference, fault mutation, weight restoration and artifact
publication require separate qualification.

The original historical Phase6E digest serialization remains
unreproduced and unchanged.

## Governance

This candidate and its qualification report do not authorize
experimental CSC execution.

No existing frozen tracked source is modified, no commit is created,
and no push is performed.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
