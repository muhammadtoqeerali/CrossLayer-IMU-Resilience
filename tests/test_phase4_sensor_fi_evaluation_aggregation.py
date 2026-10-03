import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_aggregation_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_aggregation.py"
)


def load_config():
    return json.loads(
        CONFIG.read_text()
    )


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_aggregation",
        IMPL,
    )

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m


def test_subject_is_primary_inferential_unit():
    d = load_config()

    assert d["primary_inferential_unit"] == "subject"

    assert d["raw_prediction_unit"]["independent_for_uncertainty"] is False

    assert d["seed_aggregation"]["seeds_independent_for_uncertainty"] is False

    assert d["fold_aggregation"]["folds_independent_for_uncertainty"] is False

    assert d["uncertainty"]["window_bootstrap_prohibited"] is True


def test_exact_dataset_strata():
    d = load_config()
    x = d["dataset_strata"]

    assert x["UNIVR_subject_count"] == 29
    assert x["KFALL_subject_count"] == 32
    assert x["total_subject_count"] == 61
    assert x["bootstrap_stratified_by_dataset"] is True


def test_exact_bootstrap_contract():
    d = load_config()
    u = d["uncertainty"]

    assert u["method"] == "stratified_paired_subject_bootstrap_percentile"
    assert u["confidence_level"] == 0.95
    assert u["bootstrap_replicates"] == 10000
    assert u["resample_unit"] == "subject"
    assert u["resample_sizes"] == {
        "UNIVR": 29,
        "KFALL": 32,
    }
    assert u["paired_clean_fault_resampling"] is True


def test_confusion_metric_definitions():
    m = load_impl()

    x = m.confusion_metrics(
        tp=8,
        fn=2,
        tn=18,
        fp=2,
    )

    assert x["falling_recall"] == pytest.approx(0.8)
    assert x["activity_specificity"] == pytest.approx(0.9)
    assert x["balanced_accuracy"] == pytest.approx(0.85)


def test_undefined_metric_remains_nan():
    m = load_impl()

    x = m.confusion_metrics(
        tp=0,
        fn=0,
        tn=10,
        fp=0,
    )

    assert math.isnan(
        x["falling_recall"]
    )

    assert math.isnan(
        x["balanced_accuracy"]
    )


def test_nested_equal_weight_prevents_replicate_inflation():
    m = load_impl()

    rows = []

    # Seed 42: variant A has three replicates around 0;
    # variant B has one replicate at 1.
    # Equal variant macro must be 0.5, not raw-row mean 0.25.
    for r in [0, 1, 2]:
        rows.append({
            "checkpoint_seed": 42,
            "variant_id": "A",
            "replicate_index": r,
            "value": 0.0,
        })

    rows.append({
        "checkpoint_seed": 42,
        "variant_id": "B",
        "replicate_index": 0,
        "value": 1.0,
    })

    x = m.aggregate_nested_equal(
        rows,
        value_key="value",
    )

    assert x["subject_value"] == pytest.approx(0.5)


def test_three_seed_equal_weighting():
    m = load_impl()

    rows = [
        {
            "checkpoint_seed": 42,
            "variant_id": "A",
            "replicate_index": 0,
            "value": 0.0,
        },
        {
            "checkpoint_seed": 123,
            "variant_id": "A",
            "replicate_index": 0,
            "value": 0.5,
        },
        {
            "checkpoint_seed": 2025,
            "variant_id": "A",
            "replicate_index": 0,
            "value": 1.0,
        },
    ]

    x = m.aggregate_nested_equal(
        rows,
        value_key="value",
    )

    assert x["subject_value"] == pytest.approx(0.5)
    assert x["eligible_seed_count"] == 3


def test_positive_degradation_means_worse_fault_result():
    m = load_impl()

    d = m.paired_degradation(
        metric="balanced_accuracy",
        clean=0.95,
        fault=0.80,
    )

    assert d == pytest.approx(0.15)


def test_bootstrap_is_deterministic_and_subject_stratified():
    m = load_impl()

    rows = []

    for i in range(29):
        rows.append({
            "subject": f"U{i:02d}",
            "dataset": "UNIVR",
            "value": 0.01 * i,
        })

    for i in range(32):
        rows.append({
            "subject": f"K{i:02d}",
            "dataset": "KFALL",
            "value": 0.02 * i,
        })

    a = m.stratified_subject_bootstrap(
        rows,
        value_key="value",
        replicates=500,
        seed_parts=(
            "test",
            "bias",
            "L1",
        ),
    )

    b = m.stratified_subject_bootstrap(
        rows,
        value_key="value",
        replicates=500,
        seed_parts=(
            "test",
            "bias",
            "L1",
        ),
    )

    assert a == b
    assert a["univr_subject_count"] == 29
    assert a["kfall_subject_count"] == 32
    assert a["total_subject_count"] == 61
    assert a["ci95_low"] <= a["point_estimate"] <= a["ci95_high"]


def test_bootstrap_rejects_duplicate_subject_rows():
    m = load_impl()

    rows = [
        {
            "subject": "U01",
            "dataset": "UNIVR",
            "value": 1.0,
        },
        {
            "subject": "U01",
            "dataset": "UNIVR",
            "value": 2.0,
        },
    ]

    with pytest.raises(ValueError):
        m.stratified_subject_bootstrap(
            rows,
            value_key="value",
            replicates=10,
        )


def test_pooled_windows_are_descriptive_only():
    d = load_config()

    assert d["raw_prediction_unit"]["pooled_window_metrics"] == "descriptive_only"
    assert d["classification_metrics"]["pooled_window_level"] == "descriptive_only"
    assert d["activity_false_alarm_reporting"]["pooled_false_positive_window_rate"] == "descriptive_only"


def test_onfield_excluded_from_faulted_evaluation():
    d = load_config()

    assert d["onfield_policy"]["part_of_phase4h_fault_generation"] is False
    assert d["onfield_policy"]["part_of_phase4h_faulted_evaluation"] is False


def test_only_acceptance_criteria_remain_unfrozen():
    d = load_config()

    assert d["still_not_frozen"] == [
        "robustness acceptance criteria"
    ]
