# Phase 3A Dataset Provenance Audit

## Status

PASS

Audit mode: READ ONLY

No split was created.

No dataset file was modified.

## Historical 400-ms segment roots

| Dataset root | Exists | Subjects | Ordinary subjects | Segment files | Windows | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| `KFall_oriented` | True | 32 | 32 | 5075 | 150045 | 438230240 |
| `UniVrFall_oriented` | True | 29 | 29 | 1234 | 52226 | 152398016 |
| `UniVrFall_KFall_OF` | True | 73 | 71 | 6327 | 1190235 | 3467584032 |
| `OnField` | True | 12 | 10 | 18 | 987964 | 2876955776 |
| `back_UniVrFall` | True | 37 | 35 | 1118 | 1035150 | 3014643008 |
| `back_UniVrFall_KFall` | True | 69 | 67 | 6193 | 1187326 | 3459078720 |

## Special historical subject IDs

### UniVrFall_KFall_OF

- `1000`: 206146 windows across 1 segment files
- `999`: 14326 windows across 1 segment files

### OnField

- `1000`: 206146 windows across 1 segment files
- `999`: 14326 windows across 1 segment files

### back_UniVrFall

- `1000`: 206146 windows across 1 segment files
- `999`: 14326 windows across 1 segment files

### back_UniVrFall_KFall

- `1000`: 206146 windows across 1 segment files
- `999`: 14326 windows across 1 segment files

## Companion-file patterns

### KFall_oriented

- `labels.npy`: 5075
- `segments.npy`: 5075

### UniVrFall_oriented

- `labels.npy`: 1234
- `segments.npy`: 1234

### UniVrFall_KFall_OF

- `labels.npy`: 6327
- `segments.npy`: 6327

### OnField

- `labels.npy`: 18
- `segments.npy`: 18

### back_UniVrFall

- `labels.npy`: 1118
- `segments.npy`: 1118

### back_UniVrFall_KFall

- `labels.npy`: 6193
- `segments.npy`: 6193

## Historical source-code references

- `RC-RGD-IMU_publish`: 51 matching source files
- `IMU_Reliability`: 69 matching source files
- `Protechto-master`: 250 matching source files
- `Protechto_master`: 250 matching source files

## Duplicate candidates across roots

7445

These are duplicate candidates only.

Large-array sampled fingerprints are not authoritative full hashes.

## Historical split status

REFERENCE ONLY

The preserved historical split used subject-level splitting with random state 42.

Phase 3 does not automatically reuse that split.

## Questions to resolve before split freeze

- Which physical datasets constitute the primary development population?
- What do path levels below subject encode: trial, activity, event, direction or sequence?
- Which companion arrays define labels and timing?
- What are the authoritative fall-onset/contact time semantics for each dataset?
- Are combined trees copies, transformed derivatives or merged views of the source datasets?
- What should be development/calibration, held-out confirmation and external-generalization populations?
