# Phase 4H Outer Subject Aggregation v1

This frozen stage is the first analysis that reads held-out Phase 4H metric
values.

It performs only the already-qualified within-subject hierarchy:

`replicate -> variant -> checkpoint seed -> subject`

It materializes paired C0/CS subject values for:

- family/severity macros,
- individual fault variants,
- checkpoint-seed-specific detail,
- subject timing summaries,
- source denominator/coverage records.

No cross-subject bootstrap is executed in this stage. No direction label or
global robustness label is produced. No result is permitted to alter any
scientific parameter, selection rule, threshold, severity, family, variant,
operating point, model, or reporting rule.

The conceptual `sensor_lead_ms` metric is bound by the separately qualified
adapter to `median_sensor_lead_ms`; q25/q75 remain descriptive timing fields.
