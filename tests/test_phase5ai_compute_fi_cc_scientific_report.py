from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ai_compute_fi_cc_scientific_report_v1/scientific_report.json")
REPORT_MD = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ai_compute_fi_cc_scientific_report_v1/SCIENTIFIC_REPORT.md")
AH_RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/interpretation.json")


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status_and_source_hash():
    x = load(RESULT)

    assert (
        x["status"]
        == "FROZEN_REPORTED_PHASE5AH_CC_SCIENTIFIC_FINDINGS"
    )

    assert (
        x[
            "source_phase5ah_result_sha256"
        ]
        == sha(AH_RESULT)
        == "3841ceb4a1af3ca2f1c32a7d1126ab3f94e6debbb46222f2635e694cec6012fb"
    )


def test_availability_preserved():
    x = load(RESULT)[
        "availability"
    ]

    assert x[
        "clean_available"
    ] == 24

    assert x[
        "clean_timing_unavailable_without_imputation"
    ] == 6

    assert x[
        "CC_available"
    ] == 576

    assert x[
        "CC_timing_unavailable_without_imputation"
    ] == 144

    assert (
        x[
            "unavailable_metric"
        ]
        == "median_trigger_lead_ms"
    )


def test_persistent_vs_transient_pattern():
    x = load(RESULT)[
        "persistent_vs_transient_descriptive"
    ]

    assert x[
        "available_comparison_count"
    ] == 24

    assert x[
        "unavailable_timing_comparison_count"
    ] == 6

    assert x[
        "persistent_more_adverse_count"
    ] == 24

    assert x[
        "persistent_less_adverse_count"
    ] == 0

    assert x[
        "equal_count"
    ] == 0


def test_common_target_variant_pattern():
    x = load(RESULT)[
        "common_target_FP32_PTQ_descriptive"
    ]

    assert x[
        "common_target_count"
    ] == 10

    assert x[
        "available_comparison_count"
    ] == 24

    assert x[
        "unavailable_timing_comparison_count"
    ] == 6

    assert x[
        "ptq_less_adverse_count"
    ] == 21

    assert x[
        "ptq_more_adverse_count"
    ] == 3

    assert x[
        "equal_count"
    ] == 0


def test_three_nonuniform_ptq_cells():
    x = load(RESULT)[
        "common_target_FP32_PTQ_descriptive"
    ]

    rows = x[
        "ptq_more_adverse_cells"
    ]

    assert {
        (
            row[
                "persistence"
            ],
            row[
                "operating_point"
            ],
            row[
                "metric"
            ],
        )
        for row in rows
    } == {
        (
            "transient_one_inference",
            "timely_150ms",
            "fall_recall",
        ),
        (
            "persistent_from_onset_until_trial_end",
            "low_false_alarm",
            "false_triggers_per_activity_hour",
        ),
        (
            "persistent_from_onset_until_trial_end",
            "timely_150ms",
            "false_triggers_per_activity_hour",
        ),
    }


def test_no_universal_variant_claim():
    b = load(RESULT)[
        "reporting_boundary"
    ]

    assert b[
        "universal_variant_superiority_claim"
    ] is False

    assert b[
        "model_selected"
    ] is False


def test_no_new_inference_or_feedback():
    b = load(RESULT)[
        "reporting_boundary"
    ]

    for key in (
        "new_scientific_outcome_analysis",
        "new_bootstrap",
        "new_inferential_CI",
        "new_significance_test",
        "unavailable_values_dropped",
        "unavailable_values_imputed",
        "scalar_robustness_score",
        "model_selected",
        "target_selected",
        "fault_family_selected",
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


def test_report_markdown_exists():
    assert REPORT_MD.is_file()
    assert REPORT_MD.stat().st_size > 0


def test_report_mentions_timing_unavailable():
    text = REPORT_MD.read_text()

    assert (
        "UNAVAILABLE_WITHOUT_IMPUTATION"
        in text
    )

    assert (
        "no subject-primary median-lead robustness conclusion"
        in text.lower()
    )


def test_report_refuses_universal_superiority():
    text = REPORT_MD.read_text()

    assert (
        "does not claim universal PTQ superiority"
        in text
    )
