from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/interpretation.json")
AGGREGATE = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5af_compute_fi_cc_outcome_execution_v1/aggregate_cc_outcome.json")

CLEAN_CSV = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/clean_absolute_metrics.csv")
STRATUM_CSV = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/cc_primary_strata.csv")
MACRO_CSV = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/cc_equal_target_macro_summary.csv")
COMMON_CSV = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/common_target_fp32_ptq_comparison.csv")


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status_and_frozen_aggregate():
    x = load(RESULT)

    assert (
        x["status"]
        == "SCIENTIFICALLY_INTERPRETED_FROZEN_CC_AGGREGATE_WITH_UNAVAILABLE_TIMING_PRESERVED"
    )

    assert (
        x["accepted_aggregate_sha256"]
        == sha(AGGREGATE)
        == "608a9f7c67b07a0c7b58df6ba6967943857d35e6c0856def8179fb6847317006"
    )


def test_availability_accounting():
    x = load(RESULT)[
        "availability_accounting"
    ]

    assert x["clean_available"] == 24
    assert x["clean_unavailable_without_imputation"] == 6
    assert x["CC_available"] == 576
    assert x["CC_unavailable_without_imputation"] == 144

    assert (
        x["unavailable_metric"]
        == "median_trigger_lead_ms"
    )


def test_all_unavailable_rows_are_lead_time():
    x = load(RESULT)

    unavailable = [
        row
        for row in x[
            "CC_primary_strata"
        ]
        if (
            row[
                "aggregate_status"
            ]
            == "UNAVAILABLE_WITHOUT_IMPUTATION"
        )
    ]

    assert len(unavailable) == 144

    assert {
        row["metric"]
        for row in unavailable
    } == {
        "median_trigger_lead_ms"
    }

    assert all(
        row[
            "degradation_estimate"
        ] is None
        and row["lower"] is None
        and row["upper"] is None
        for row in unavailable
    )


def test_available_rows_have_frozen_direction_semantics():
    x = load(RESULT)

    for row in x[
        "CC_primary_strata"
    ]:
        if (
            row[
                "aggregate_status"
            ]
            != "AVAILABLE"
        ):
            continue

        value = row[
            "degradation_estimate"
        ]

        if value > 0:
            assert (
                row[
                    "point_direction"
                ]
                == "adverse"
            )

        elif value < 0:
            assert (
                row[
                    "point_direction"
                ]
                == "favorable"
            )

        else:
            assert (
                row[
                    "point_direction"
                ]
                == "no_point_difference"
            )


def test_lead_macros_remain_unavailable():
    x = load(RESULT)

    rows = [
        row
        for row in x[
            "CC_equal_target_descriptive_macros"
        ]
        if (
            row["metric"]
            == "median_trigger_lead_ms"
        )
    ]

    assert len(rows) == 12

    assert all(
        row[
            "macro_status"
        ]
        == "UNAVAILABLE_WITHOUT_IMPUTATION"
        and row[
            "descriptive_equal_target_macro_degradation"
        ] is None
        for row in rows
    )


def test_lead_common_target_comparisons_remain_unavailable():
    x = load(RESULT)

    rows = [
        row
        for row in x[
            "common_target_FP32_PTQ_descriptive_comparison"
        ]
        if (
            row["metric"]
            == "median_trigger_lead_ms"
        )
    ]

    assert len(rows) == 6

    assert all(
        row[
            "comparison_status"
        ]
        == "UNAVAILABLE_WITHOUT_IMPUTATION"
        and row[
            "ptq_minus_fp32_macro_degradation"
        ] is None
        for row in rows
    )


def test_target_sets():
    x = load(RESULT)[
        "target_sets"
    ]

    assert len(
        x["common_targets"]
    ) == 10

    assert len(
        x["ptq_only_targets"]
    ) == 4

    assert x[
        "fp32_only_targets"
    ] == []


def test_no_drop_or_imputation():
    b = load(RESULT)[
        "interpretation_boundary"
    ]

    assert b[
        "unavailable_timing_preserved"
    ] is True

    assert b[
        "unavailable_values_dropped"
    ] is False

    assert b[
        "unavailable_values_imputed"
    ] is False


def test_no_new_inference():
    b = load(RESULT)[
        "interpretation_boundary"
    ]

    assert b[
        "existing_subject_bootstrap_CI_interpreted"
    ] is True

    assert b[
        "new_subject_bootstrap_performed"
    ] is False

    assert b[
        "new_timing_event_bootstrap_performed"
    ] is False

    assert b[
        "new_inferential_CI_created"
    ] is False


def test_no_selection_feedback_or_new_execution():
    b = load(RESULT)[
        "interpretation_boundary"
    ]

    for key in (
        "scalar_robustness_score",
        "rank_based_selection",
        "best_target_selected",
        "best_family_selected",
        "best_variant_selected",
        "threshold_retuning",
        "checkpoint_selection",
        "fault_resampling",
        "protocol_change",
        "CSC_generated",
        "OnField_used",
        "model_forward_executed",
        "new_fault_execution",
    ):
        assert b[key] is False


def test_csv_outputs_exist():
    for path in (
        CLEAN_CSV,
        STRATUM_CSV,
        MACRO_CSV,
        COMMON_CSV,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0
