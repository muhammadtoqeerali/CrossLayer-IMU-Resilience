import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SUMMARY = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_interpretation_v1.json"
)


def load():
    return json.loads(
        SUMMARY.read_text()
    )


def test_status_and_reporting_lineage():
    x = load()

    assert x["status"] == (
        "FROZEN_INTERPRETATION_OF_HELD_OUT_OUTER_RESULTS"
    )

    assert x["source_reporting"][
        "bootstrap_replicates"
    ] == 10000

    assert x["source_reporting"][
        "execution_interpretability"
    ] == "EXECUTION_INTERPRETABLE"


def test_global_direction_counts():
    x = load()

    assert x[
        "direction_counts_all_strata"
    ] == {
        "DEGRADATION_SUPPORTED": 2677,
        "IMPROVEMENT_SUPPORTED": 268,
        "NO_DIRECTIONAL_CONCLUSION": 1555,
        "UNRESOLVED": 36,
    }


def test_overall_direction_counts():
    x = load()

    assert x[
        "direction_counts_by_stratum"
    ][
        "overall_61_subject"
    ] == {
        "DEGRADATION_SUPPORTED": 928,
        "IMPROVEMENT_SUPPORTED": 96,
        "NO_DIRECTIONAL_CONCLUSION": 476,
        "UNRESOLVED": 12,
    }


def test_incomplete_records_are_only_axis_loss_L3_precision_and_timing():
    x = load()

    checks = x[
        "predeclared_failure_pattern_checks"
    ]

    assert checks[
        "incomplete_record_count"
    ] == 36

    observed = {
        (
            row["fault_family"],
            row["severity_level"],
            row["metric"],
        )
        for row in checks[
            "incomplete_record_structures"
        ]
    }

    assert observed == {
        (
            "axis_loss",
            "L3",
            "precision",
        ),
        (
            "axis_loss",
            "L3",
            "sensor_lead_ms",
        ),
    }


def test_axis_loss_L3_recall_and_balanced_accuracy_are_degradation_supported():
    x = load()

    axis = x[
        "predeclared_failure_pattern_checks"
    ][
        "axis_loss_L3_overall"
    ]

    for model in axis.values():
        for op in model.values():
            assert op[
                "falling_recall"
            ][
                "direction_status"
            ] == "DEGRADATION_SUPPORTED"

            assert op[
                "balanced_accuracy"
            ][
                "direction_status"
            ] == "DEGRADATION_SUPPORTED"

            assert op[
                "precision"
            ][
                "direction_status"
            ] == "UNRESOLVED"

            assert op[
                "sensor_lead_ms"
            ][
                "direction_status"
            ] == "UNRESOLVED"


def test_governance_boundary_prohibits_feedback():
    x = load()

    g = x[
        "governance_boundary"
    ]

    assert g[
        "outer_results_may_be_reported"
    ] is True

    assert g[
        "outer_results_may_be_interpreted"
    ] is True

    assert g[
        "outer_results_may_be_used_for_retuning"
    ] is False

    assert g[
        "outer_results_may_change_thresholds"
    ] is False

    assert g[
        "outer_results_may_change_fault_severities"
    ] is False

    assert g[
        "outer_results_may_change_operating_points"
    ] is False

    assert g[
        "outer_results_may_change_reporting_rules"
    ] is False

    assert g[
        "NO_DIRECTIONAL_CONCLUSION_means_equivalence"
    ] is False

    assert g[
        "binary_global_robustness_label_generated"
    ] is False


def test_dataset_specific_statuses_are_retained():
    x = load()

    d = x[
        "dataset_direction_status_disagreement_counts"
    ]

    assert d[
        "prospective_fp32_300ms"
    ][
        "timely_150ms"
    ][
        "sensor_lead_ms"
    ] == 14

    assert d[
        "qualified_static_ptq_v7"
    ][
        "low_false_alarm"
    ][
        "sensor_lead_ms"
    ] == 19
