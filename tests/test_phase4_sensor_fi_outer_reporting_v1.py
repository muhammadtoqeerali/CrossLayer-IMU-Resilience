import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_reporting_v1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_outer_reporting_v1",
        RUNNER,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def records():
    rows = []

    for i in range(29):
        rows.append({
            "subject": 1000 + i,
            "dataset": "UNIVR",
            "C0_subject_value": 0.9,
            "CS_subject_value": 0.8,
            "paired_degradation": 0.1,
        })

    for i in range(32):
        rows.append({
            "subject": 2000 + i,
            "dataset": "KFALL",
            "C0_subject_value": 0.9,
            "CS_subject_value": 0.8,
            "paired_degradation": 0.1,
        })

    return rows


def identity():
    return {
        "model_variant":
            "prospective_fp32_300ms",
        "operating_point":
            "balanced",
        "fault_family":
            "bias",
        "severity_level":
            "L1",
        "metric":
            "falling_recall",
    }


def empty_denominator():
    fields = (
        "TP",
        "FN",
        "TN",
        "FP",
        "eligible_event_count",
        "detected_event_count",
        "missed_event_count",
        "false_trigger_episode_count",
        "activity_seconds",
    )

    return {
        "interpretation":
            "noninferential_repeated_source_condition_totals",
        "subject_count": 61,
        "C0": {
            "source_row_count": 0,
            **{
                x: 0.0
                if x == "activity_seconds"
                else 0
                for x in fields
            },
        },
        "CS": {
            "source_row_count": 0,
            **{
                x: 0.0
                if x == "activity_seconds"
                else 0
                for x in fields
            },
        },
    }


def test_family_bootstrap_positive_degradation():
    m = load_runner()

    out = m.family_bootstrap_task(
        (
            identity(),
            records(),
            empty_denominator(),
            "overall_61_subject",
            100,
        )
    )

    assert out[
        "paired_degradation_point"
    ] == pytest.approx(0.1)

    assert out[
        "direction_status"
    ] == "DEGRADATION_SUPPORTED"

    assert out[
        "eligible_subject_count"
    ] == 61

    assert out[
        "coverage_complete"
    ] is True


def test_dataset_strata_exact_subject_counts():
    m = load_runner()

    for stratum, n in (
        ("UNIVR", 29),
        ("KFALL", 32),
    ):
        out = m.family_bootstrap_task(
            (
                identity(),
                records(),
                empty_denominator(),
                stratum,
                100,
            )
        )

        assert out[
            "total_subject_count"
        ] == n

        assert out[
            "eligible_subject_count"
        ] == n


def test_incomplete_paired_coverage_is_unresolved():
    m = load_runner()

    x = records()
    x[0]["paired_degradation"] = float("nan")

    out = m.family_bootstrap_task(
        (
            identity(),
            x,
            empty_denominator(),
            "overall_61_subject",
            100,
        )
    )

    assert out[
        "coverage_complete"
    ] is False

    assert out[
        "bootstrap_replicates"
    ] == 0

    assert math.isnan(
        out[
            "paired_degradation_ci95_low"
        ]
    )

    assert math.isnan(
        out[
            "paired_degradation_ci95_high"
        ]
    )

    assert out[
        "direction_status"
    ] == "UNRESOLVED"


def test_descriptive_detail_has_no_inferential_ci():
    m = load_runner()

    out = m.point_detail_record(
        rows=records(),
        stratum="UNIVR",
        identity={
            **identity(),
            "detail_type":
                "individual_fault_variant",
            "variant_id":
                "variant0",
        },
    )

    assert out[
        "inferential_ci_applied"
    ] is False

    assert "paired_degradation_ci95_low" not in out
    assert "direction_status" not in out


def test_no_global_binary_label_call():
    source = RUNNER.read_text()

    assert "global_robustness_label(" not in source
