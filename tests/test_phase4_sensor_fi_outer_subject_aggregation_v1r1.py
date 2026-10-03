import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_subject_aggregation_v1r1.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_outer_subject_aggregation_v1r1",
        RUNNER,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def row(value):
    return {
        "checkpoint_seed": 42,
        "variant_id": "A",
        "replicate_index": 0,
        "precision": value,
        "median_sensor_lead_ms": value,
    }


def test_none_is_converted_to_nan_only_at_helper_boundary():
    m = load_runner()

    original = row(None)

    normalized = m.normalize_missing_for_qualified_helper(
        [original],
        value_key="precision",
    )

    assert original["precision"] is None
    assert math.isnan(
        normalized[0]["precision"]
    )


def test_none_precision_remains_undefined_after_aggregation():
    m = load_runner()

    x = m.metric_subject_aggregate(
        [row(None)],
        "precision",
    )

    assert math.isnan(
        x["value"]
    )

    assert x["eligible_seed_count"] == 0
    assert x["source_row_total"] == 1
    assert x["source_finite_row_count"] == 0


def test_none_timing_remains_undefined_after_aggregation():
    m = load_runner()

    x = m.timing_subject_aggregate(
        [row(None)],
        "median_sensor_lead_ms",
    )

    assert math.isnan(
        x["value"]
    )

    assert x["eligible_seed_count"] == 0
    assert x["source_finite_row_count"] == 0


def test_none_and_nan_have_identical_qualified_semantics():
    m = load_runner()

    none_result = m.metric_subject_aggregate(
        [row(None)],
        "precision",
    )

    nan_result = m.metric_subject_aggregate(
        [row(float("nan"))],
        "precision",
    )

    assert math.isnan(
        none_result["value"]
    )

    assert math.isnan(
        nan_result["value"]
    )

    assert (
        none_result["eligible_seed_count"]
        == nan_result["eligible_seed_count"]
        == 0
    )


def test_finite_values_are_unchanged():
    m = load_runner()

    x = m.metric_subject_aggregate(
        [row(0.375)],
        "precision",
    )

    assert x["value"] == pytest.approx(
        0.375
    )

    assert x["eligible_seed_count"] == 1


def test_nanmean_behavior_preserved_for_mixed_missingness():
    m = load_runner()

    rows = [
        {
            "checkpoint_seed": 42,
            "variant_id": "A",
            "replicate_index": 0,
            "precision": None,
            "median_sensor_lead_ms": None,
        },
        {
            "checkpoint_seed": 42,
            "variant_id": "A",
            "replicate_index": 1,
            "precision": 0.8,
            "median_sensor_lead_ms": 12.0,
        },
    ]

    x = m.metric_subject_aggregate(
        rows,
        "precision",
    )

    assert x["value"] == pytest.approx(
        0.8
    )

    assert x["source_row_total"] == 2
    assert x["source_finite_row_count"] == 1


def test_repair_does_not_add_cross_subject_inference():
    source = RUNNER.read_text()

    assert "stratified_subject_bootstrap(" not in source
    assert "one_stratum_subject_bootstrap(" not in source
    assert "global_robustness_label(" not in source
