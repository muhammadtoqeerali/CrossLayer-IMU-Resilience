# Phase 3M-R Label Semantics and UniVR Time Repair

## Status

PASS

## Canonical event binding

Annotated events:

2919

Dataset counts:

`{'KFALL': 2346, 'UNIVR': 573}`

Processed trials missing:

0

The UniVR event-index namespace mismatch is resolved by canonical
dataset-qualified subject identity.

## Historical label semantics

The 150-ms safety deadline is not interpreted as part of historical
training-label generation.

Annotated fall events with no Falling-labelled processed window:

`['KFALL_106_T27_R05']`

Annotated events with at least one Falling-labelled window:

2918

The frozen labels remain unchanged.

## Start-label replay diagnostic

Exact trial replays:

0

Nonexact trial replays:

2919

Window-level mismatches:

5406

This replay is diagnostic lineage evidence. It is not used to rewrite the
frozen labels.

## Safety timing geometry

Events with a complete post-onset 300-ms window before the 150-ms deadline:

2388

Events with any complete historical grid window before the deadline:

2919

Events with an existing Falling-labelled window ending before the deadline:

2918

Events with an onset-overlapping grid window ending before the deadline:

2919

These are timing-geometry counts only.

No classifier outcome is involved.

## UniVR legacy timing

Events audited:

573

Original files missing:

0

Legacy parse failures:

0

Original/oriented row-count mismatches:

573

Event positions out of range:

0

Median legacy timestamp-step summary:

`{'count': 573, 'max': 10.0, 'median': 10.0, 'min': 10.0}`

Legacy backwards timestamp steps:

1

Legacy duplicate timestamp steps:

120971

Legacy event-duration absolute error:

`{'count': 573, 'max': 140.0, 'median': 70.0, 'min': 30.0}`

Oriented event-duration absolute error:

`{'count': 566, 'max': 1218640.0, 'median': 599320.0, 'min': 259700.0}`

## Scientific separation

Historical classification labels and safety-deadline event timing are two
different protocol layers.

The first Phase-3M strict equivalence hypothesis is rejected.

The canonical event-to-trial mapping is retained.

Physical lead-time semantics remain unfrozen until the remaining timing
evidence is reviewed.
