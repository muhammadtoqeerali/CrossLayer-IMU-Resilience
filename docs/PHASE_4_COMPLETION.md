# Phase 4 Completion

Status: **COMPLETE**

Phase 4 is the **Sensor Fault Engine** stage of the governed research roadmap.
The next phase is **Phase 5 — Compute Fault Engine**. The four-regime
cross-layer characterization (`C0`, `CS`, `CC`, `CSC`) belongs to Phase 6
after the compute fault engine exists.

## Completed Phase-4 scope

Phase 4 completed:

- a prospective 300-ms FP32 CNN baseline under the frozen Phase-3 protocol;
- qualification of the static PTQ v7 mixed-precision comparator across all
  15 seed×fold checkpoints;
- a deterministic P0 sensor-fault injection protocol with 12 fault families;
- held-out outer sensor-fault execution over all 61 primary subjects;
- subject-level reporting with frozen 10,000-replicate bootstrap uncertainty;
- frozen interpretation of the held-out sensor-fault results without
  retuning;
- an independent retained OnField Activity-only external evaluation over
  subjects 1001–1010;
- frozen Activity-side external interpretation without model, threshold,
  checkpoint, or operating-point selection.

## Sensor-FI execution scale

The immutable held-out sensor-FI execution contains:

- 320,616 subject-condition rows;
- 42,766,632 unique fault instances;
- 479,750,160 model-window evaluations;
- 4,536 family×severity inferential reporting records.

`NO_DIRECTIONAL_CONCLUSION` is not interpreted as equivalence. No global
binary robustness label is generated.

## OnField external evidence

The retained external cohort contains:

- 10 subjects;
- 16 trials;
- 1,023,337 Activity windows;
- zero Falling windows.

The external run executed 30,700,110 model-window evaluations and 92,100,330
threshold applications. It supports Activity-side false-alarm reporting only.
It cannot support fall recall, event recall, missed-fall rate, lead time,
recovery, or external fall-detection-effectiveness claims.

## Quantization boundary

The Phase-4G v7 artifact is a qualified **mixed-precision static PTQ**
comparator. It is not represented as a fully INT8 model. Historical Phase-2
`project.yaml` baseline fields remain unchanged where they describe the older
reference contract.

## Phase boundary

Phase 4 does not claim completion of compute fault injection or the combined
sensor+compute regime.

Those remain governed future work:

- Phase 5: Compute Fault Engine;
- Phase 6: Cross-Layer Vulnerability Characterization over `C0`, `CS`, `CC`,
  and `CSC`.

## Scientific safeguards

Phase-4 held-out outer results and OnField external results were not used to
retune model weights, thresholds, operating points, fault severities,
calibration, checkpoint selection, or statistical/reporting rules.

No MCU or physical-hardware claim is made.
