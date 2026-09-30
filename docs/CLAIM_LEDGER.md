# Claim and Evidence Ledger

All entries begin as hypotheses.

| ID | Working proposition | Current state |
|---|---|---|
| H1 | Sensor and compute faults produce different vulnerability patterns | HYPOTHESIS |
| H2 | Combined faults reveal failure regimes missed by isolated evaluation | HYPOTHESIS |
| H3 | Lightweight runtime evidence can identify important silent failures | HYPOTHESIS |
| H4 | Adaptive protection reduces unsafe or silent failures | HYPOTHESIS |
| H5 | Adaptive protection has lower normal-path cost than always-on protection | HYPOTHESIS |
| H6 | Protection can preserve useful pre-impact timing | HYPOTHESIS |
| H7 | The framework generalizes beyond one backbone or dataset | HYPOTHESIS |

## Evidence states

HYPOTHESIS

Planned proposition without final experimental evidence.

DEVELOPMENT_EVIDENCE

Observed during development but not valid as final held-out evidence.

CALIBRATED

Operating point fixed using permitted calibration/validation data.

CONFIRMED

Supported by frozen held-out evaluation.

EXTERNAL_CONFIRMED

Supported by an independent compatible architecture/dataset/platform.

## Rule

Every numerical publication claim must eventually trace to:

- Git commit
- configuration
- data manifest
- model identity
- run manifest
- metric implementation
- result artifact
