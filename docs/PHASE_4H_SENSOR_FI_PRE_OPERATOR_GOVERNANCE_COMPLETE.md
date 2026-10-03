# Phase 4H — Pre-Operator Governance Complete

**Status:** PRE_OPERATOR_GOVERNANCE_COMPLETE

The Phase-4H sensor fault-injection design is now frozen through the
operator-semantics boundary.

Authoritative components are:

- input/injection-layer contract;
- prospective severity protocol;
- qualified sampling v3 with causal frame loss;
- qualified subject-level aggregation and sampling-v3 rebind;
- qualified reporting v2 bound to sampling v3;
- qualified executable jitter semantics.

There are no remaining governance choices to make before implementing
the complete 12-family sensor-fault operator engine.

No model prediction, validation outcome, outer-test outcome, or OnField
outcome was used to choose these rules.

The next step is implementation and operator-level qualification only.
