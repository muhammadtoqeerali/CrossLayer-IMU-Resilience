# Phase 3H Primary 300-ms Dataset Binding

## Status

PASS

## Primary protocol

- Window: 300 ms
- Sampling rate: 100 Hz
- Samples per window: 30
- Overlap: 50 percent
- Stride: 15 samples
- Stride time: 150 ms
- Stored channels: 9

## Primary population

- UniVRFall: 29 subjects
- KFall: 32 subjects
- Total: 61 subjects
- OnField in primary folds: no

## Primary dataset root

`/mnt/hdd16T/protechto/data/UniVrFall_KFall_NoOF/segments/300ms_50ov_npseg_filt_binary`

## Exact source binding

All 61 merged subjects were compared against
their separately generated UniVRFall or KFall processed source roots.

Result:

`EXACT`

The existing 61-subject root therefore does not need to be regenerated merely
to establish 300-ms / 50-percent-overlap primary-data provenance.

## Task arrays

Trials:

6,309

Windows:

273,830

Labels:

`{'Activity': 264024, 'Falling': 9806}`

Empty trials:

0

## Subject identity

Scientific identity is dataset-qualified.

Examples:

- `UNIVR_09`
- `KFALL_06`

Existing numerical directory names remain storage identifiers only.

They are not treated as cross-dataset scientific subject identifiers.

## Excluded lineage

95-percent-overlap experiments are excluded from the primary CrossLayer
protocol.

Old manually combined 71/73-subject folders are not used as the primary
evaluation population.

## 400-ms condition

400 ms remains available under exactly the same 50-percent-overlap policy.

At 100 Hz:

- 40 samples/window
- 20-sample stride
- 200-ms stride time

It will be used as historical reference and, if needed, a controlled
window-duration sensitivity experiment.
