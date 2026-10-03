"""Phase-4H robustness reporting policy helpers.

No model inference or fault execution.

There is intentionally no arbitrary practical-loss threshold and no
global robust/not-robust classifier.
"""

from __future__ import annotations

import math


DIRECTION_STATUSES = {
    "DEGRADATION_SUPPORTED",
    "IMPROVEMENT_SUPPORTED",
    "NO_DIRECTIONAL_CONCLUSION",
    "UNRESOLVED",
}


def classify_direction(
    *,
    point,
    ci95_low,
    ci95_high,
    coverage_complete=True,
):
    point = float(point)
    ci95_low = float(ci95_low)
    ci95_high = float(ci95_high)

    if not coverage_complete:
        return "UNRESOLVED"

    if not all(
        math.isfinite(x)
        for x in [
            point,
            ci95_low,
            ci95_high,
        ]
    ):
        return "UNRESOLVED"

    if ci95_low > ci95_high:
        raise ValueError(
            "ci95_low must be <= ci95_high"
        )

    if ci95_low > 0.0:
        return "DEGRADATION_SUPPORTED"

    if ci95_high < 0.0:
        return "IMPROVEMENT_SUPPORTED"

    return "NO_DIRECTIONAL_CONCLUSION"


def execution_interpretability(
    gates,
):
    """Return protocol-compliance status, not robustness status."""

    required = {
        "frozen_input_contract_hash_matches",
        "frozen_severity_protocol_hash_matches",
        "qualified_sampling_v2_hash_matches",
        "qualified_aggregation_protocol_hash_matches",
        "all_required_checkpoints_and_model_variants_accounted_for",
        "all_fault_instances_have_valid_replay_ids",
        "no_outer_test_tuning_occurred",
        "no_OnField_fault_generation_occurred",
        "all_required_result_dimensions_present",
        "all_missing_or_undefined_metrics_explicitly_accounted_for",
    }

    missing = required - set(
        gates
    )

    if missing:
        raise ValueError(
            f"missing interpretability gates: {sorted(missing)}"
        )

    if all(
        bool(
            gates[key]
        )
        for key in required
    ):
        return "EXECUTION_INTERPRETABLE"

    return "EXECUTION_NOT_INTERPRETABLE"


def global_robustness_label(*args, **kwargs):
    raise RuntimeError(
        "Phase-4H v1 reporting policy prohibits a binary global "
        "robust/not-robust label because no validated practical "
        "performance margin is available."
    )
