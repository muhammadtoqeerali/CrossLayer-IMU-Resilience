# Phase 3L Curated Event and Sensor-Time Lineage Audit

## Status

PASS

## Phase-3K clarification

The direct workbook-row join from Phase 3K is diagnostic only and is not the authoritative event mapping.

The expanded curated event index is now audited independently.

## Curated event population

Events: 2919

Dataset counts: `{'KFALL': 2346, 'UNIVR': 573}`

## Processed fall-trial comparison

Processed fall-positive trials: 2918

Exact event/fall intersection: 2345

Event records not represented as processed fall-positive trials: 574

Processed fall-positive trials without curated event: 573

## Raw lineage

Missing sensor files: 0

Event positions outside sensor-file bounds: 0

KFall FrameCounter exact matches: 2346

KFall FrameCounter mismatches: 0

## Timing qualification

KFall: `QUALIFIED`

UniVR: `PENDING_ORIGINAL_TIMESTAMP_SYNCHRONIZATION_AUDIT`

Full physical lead-time semantics remain unfrozen until UniVR timestamp/synchronization lineage is resolved.
