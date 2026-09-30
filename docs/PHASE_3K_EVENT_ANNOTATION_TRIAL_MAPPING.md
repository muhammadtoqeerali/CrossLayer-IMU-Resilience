# Phase 3K Event Annotation and Trial-Mapping Audit

## Status

PASS

## Primary processed trial population

Total trials: 6,309

Dataset trial counts: `{'KFALL': 5075, 'UNIVR': 1234}`

Processed Falling trial counts: `{'KFALL': 2345, 'UNIVR': 573}`

## Annotation mapping

Complete annotation events: `{'KFALL': 480, 'UNIVR': 574}`

Mapped annotation events: `{'KFALL': 460, 'UNIVR': 566}`

Annotation events without processed trial: 28

Duplicate annotation trial keys: 0

Invalid onset >= impact rows: 1

Processed Falling trials without complete event annotation: 2868

Mapped events whose processed trial has no Falling windows: 976

## UniVR event-count issue

Local complete onset+impact rows: 574

Public summary fall events: 573

Status: `OPEN_ONE_EVENT_DIFFERENCE`

## Timing boundary

The 300-ms classifier is causal only when a decision is timestamped at the final sample available to its input window.

Candidate lead time is `impact_time - decision_available_time`.

However annotation frame numbers are not yet automatically treated as 100-Hz sensor sample indices.

Phase 3L must qualify the annotation-to-sensor synchronization before physical pre-impact timing is frozen.

## Phase-3L interpretation correction

The Phase-3K direct workbook-row join is retained as an audit diagnostic only.

It is not the authoritative event-to-trial mapping.

KFall annotation workbook rows can represent multiple trial instances, so
reducing each workbook row to one scalar Trial ID undercounts the true
event population.

The prospective event lineage instead uses the separately audited expanded
curated event index, subject to Phase-3L validation.
