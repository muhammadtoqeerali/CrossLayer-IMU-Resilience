import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_subject_aggregation_v1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_outer_subject_aggregation_v1",
        RUNNER,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_row(seed, variant, replicate, value):
    return {
        "checkpoint_seed": seed,
        "variant_id": variant,
        "replicate_index": replicate,
        "falling_recall": value,
        "activity_specificity": value,
        "balanced_accuracy": value,
        "precision": value,
        "f1": value,
        "event_recall": value,
        "median_sensor_lead_ms": value,
        "q25_sensor_lead_ms": value - 1,
        "q75_sensor_lead_ms": value + 1,
        "TP": 1,
        "FN": 2,
        "TN": 3,
        "FP": 4,
        "eligible_event_count": 5,
        "detected_event_count": 6,
        "missed_event_count": 7,
        "false_trigger_episode_count": 8,
        "activity_seconds": 9.0,
    }


def test_metric_source_binding_uses_qualified_adapter():
    m = load_runner()

    assert m.metric_source(
        "sensor_lead_ms"
    ) == "median_sensor_lead_ms"

    assert m.metric_source(
        "falling_recall"
    ) == "falling_recall"


def test_nested_subject_aggregate_equal_weights_variants():
    m = load_runner()

    rows = [
        base_row(42, "A", 0, 0.0),
        base_row(42, "A", 1, 0.0),
        base_row(42, "A", 2, 0.0),
        base_row(42, "B", 0, 1.0),
    ]

    x = m.metric_subject_aggregate(
        rows,
        "falling_recall",
    )

    assert x["value"] == pytest.approx(
        0.5
    )


def test_nested_subject_aggregate_equal_weights_seeds():
    m = load_runner()

    rows = [
        base_row(42, "A", 0, 0.0),
        base_row(123, "A", 0, 0.5),
        base_row(2025, "A", 0, 1.0),
    ]

    x = m.metric_subject_aggregate(
        rows,
        "balanced_accuracy",
    )

    assert x["value"] == pytest.approx(
        0.5
    )

    assert x["eligible_seed_count"] == 3


def test_denominator_summary_is_labeled_raw_source_sum():
    m = load_runner()

    rows = [
        base_row(42, "A", 0, 0.1),
        base_row(123, "A", 0, 0.2),
    ]

    x = m.denominator_summary(
        rows
    )

    assert x["source_row_count"] == 2

    sums = x[
        "noninferential_repeated_source_row_sums"
    ]

    assert sums["TP"] == 2
    assert sums["FN"] == 4
    assert sums["activity_seconds"] == pytest.approx(
        18.0
    )


def test_stage_does_not_call_global_robustness_label():
    source = RUNNER.read_text()

    assert "global_robustness_label(" not in source
    assert "stratified_subject_bootstrap(" not in source
    assert "one_stratum_subject_bootstrap(" not in source
