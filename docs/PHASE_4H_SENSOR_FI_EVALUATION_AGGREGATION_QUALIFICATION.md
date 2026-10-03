# Phase 4H — Evaluation Aggregation Qualification

**Status:** QUALIFIED_SYNTHETIC_HIERARCHY

The frozen pre-result aggregation/uncertainty protocol passed every
synthetic hierarchy and bootstrap sanity gate.

The earlier NumPy boolean JSON failure is retained as a technical
serialization failure and did not alter the scientific protocol.

Qualified properties include:

- subject-level primary inference;
- no independence assumption for overlapping windows;
- no independence assumption for stochastic replicates;
- no independence assumption for model seeds or folds;
- equal-weight nested aggregation;
- paired C0-to-CS degradation;
- deterministic 10,000-replicate subject bootstrap;
- fixed 29 UniVR / 32 KFall strata;
- exclusion of OnField from faulted evaluation.

No model was loaded and no sensor fault was executed.

Only the robustness acceptance/reporting policy remained unfrozen at
the time of this qualification.
