# Phase 4H — Dev/Cal Execution Runner Qualification

**Status:** QUALIFIED_TRAINING_CALIBRATION_EXECUTION_RUNNER

The pre-result-frozen runner passed all execution-integration gates on
the frozen training-calibration partition.

Qualification executed model forward passes and sensor faults, but did
not use performance outcomes to accept, reject, or modify any protocol
choice.

Verified properties include:

- all 15 FP32 and 15 paired PTQ-v7 artifacts load and execute;
- the same corrupted tensor is supplied to paired model variants;
- all five stored-window fault families execute;
- all seven sequence fault families execute through the historical
  filtering/windowing path;
- one real activity-only source trial in each fold reconstructs stored
  windows exactly;
- validation, outer-test, and OnField are excluded;
- the 45 frozen threshold rows are not applied or reselected;
- no robustness metric or model/fault ordering is an acceptance gate.

This qualification does not establish robustness.

Before held-out execution, one final operational manifest must freeze
the exact outer-test matrix and deterministic sharding/resume artifact
plan.
