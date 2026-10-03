import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

ADAPTER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_aggregation_adapter_v1.py"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_aggregation_adapter_v1.json"
)


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_outer_aggregation_adapter_v1",
        ADAPTER,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_config():
    return json.loads(
        CONFIG.read_text()
    )


def make_rows(dataset, n, *, value_fn):
    return [
        {
            "subject": f"{dataset}-{i:02d}",
            "dataset": dataset,
            "value": float(value_fn(i)),
        }
        for i in range(n)
    ]


def manual_one_stratum_reference(
    m,
    rows,
    *,
    dataset,
    replicates,
    seed_parts,
):
    seed = m.AGG.bootstrap_seed(
        *seed_parts
    )

    rng = np.random.default_rng(seed)

    x = np.asarray(
        [row["value"] for row in rows],
        dtype=float,
    )

    estimates = np.empty(
        replicates,
        dtype=float,
    )

    for i in range(replicates):
        estimates[i] = float(
            np.mean(
                rng.choice(
                    x,
                    size=len(x),
                    replace=True,
                )
            )
        )

    low, high = np.quantile(
        estimates,
        [0.025, 0.975],
    )

    return {
        "point": float(np.mean(x)),
        "low": float(low),
        "high": float(high),
        "seed": int(seed),
    }


def test_config_declares_no_scientific_change():
    d = load_config()

    assert d["scientific_change"] is False
    assert d["statistical_rule_change"] is False
    assert d[
        "outer_performance_values_used_to_define_adapter"
    ] is False

    assert all(
        value is False
        for value in d["selection_changes"].values()
    )


def test_timing_metric_maps_to_predeclared_subject_median():
    m = load_adapter()
    d = load_config()

    assert m.source_field_for_metric(
        "sensor_lead_ms"
    ) == "median_sensor_lead_ms"

    assert m.source_field_for_metric(
        "falling_recall"
    ) == "falling_recall"

    assert d["timing_binding"][
        "inferential_outer_row_field"
    ] == "median_sensor_lead_ms"

    assert m.timing_descriptive_fields() == (
        "q25_sensor_lead_ms",
        "median_sensor_lead_ms",
        "q75_sensor_lead_ms",
    )


@pytest.mark.parametrize(
    "dataset,n",
    [
        ("UNIVR", 29),
        ("KFALL", 32),
    ],
)
def test_one_stratum_bootstrap_matches_manual_specialization(
    dataset,
    n,
):
    m = load_adapter()

    rows = make_rows(
        dataset,
        n,
        value_fn=lambda i: 0.01 * (i + 1),
    )

    seed_parts = (
        "synthetic",
        "prospective_fp32_300ms",
        "balanced",
        "bias",
        "L1",
        "falling_recall",
        dataset,
    )

    got = m.one_stratum_subject_bootstrap(
        rows,
        value_key="value",
        dataset=dataset,
        replicates=500,
        seed_parts=seed_parts,
    )

    ref = manual_one_stratum_reference(
        m,
        rows,
        dataset=dataset,
        replicates=500,
        seed_parts=seed_parts,
    )

    assert got["point_estimate"] == pytest.approx(
        ref["point"]
    )

    assert got["ci95_low"] == pytest.approx(
        ref["low"]
    )

    assert got["ci95_high"] == pytest.approx(
        ref["high"]
    )

    assert got["seed"] == ref["seed"]

    assert got["eligible_subject_count"] == n
    assert got["total_subject_count"] == n
    assert got["coverage_complete"] is True


def test_overall_dispatch_is_exact_parent_qualified_helper():
    m = load_adapter()

    rows = (
        make_rows(
            "UNIVR",
            29,
            value_fn=lambda i: 0.001 * i,
        )
        + make_rows(
            "KFALL",
            32,
            value_fn=lambda i: 0.002 * i,
        )
    )

    seed_parts = (
        "synthetic",
        "qualified_static_ptq_v7",
        "timely_150ms",
        "orientation",
        "L3",
        "balanced_accuracy",
        "overall_61_subject",
    )

    got = m.inferential_subject_bootstrap(
        rows,
        value_key="value",
        stratum="overall_61_subject",
        replicates=500,
        seed_parts=seed_parts,
    )

    ref = m.AGG.stratified_subject_bootstrap(
        rows,
        value_key="value",
        replicates=500,
        seed_parts=seed_parts,
    )

    for key in (
        "point_estimate",
        "ci95_low",
        "ci95_high",
        "bootstrap_replicates",
        "seed",
    ):
        assert got[key] == ref[key]

    assert got["eligible_subject_count"] == 61
    assert got["total_subject_count"] == 61
    assert got["coverage_complete"] is True


def test_incomplete_coverage_does_not_bootstrap_or_impute():
    m = load_adapter()

    rows = make_rows(
        "UNIVR",
        29,
        value_fn=lambda i: 0.5,
    )

    rows[4]["value"] = float("nan")

    got = m.inferential_subject_bootstrap(
        rows,
        value_key="value",
        stratum="UNIVR",
        replicates=500,
        seed_parts=("synthetic", "missingness"),
    )

    assert got["eligible_subject_count"] == 28
    assert got["total_subject_count"] == 29
    assert got["coverage_complete"] is False
    assert got["bootstrap_replicates"] == 0
    assert got["seed"] is None
    assert math.isnan(got["ci95_low"])
    assert math.isnan(got["ci95_high"])
    assert got["point_estimate"] == pytest.approx(0.5)


def test_duplicate_subject_rejected():
    m = load_adapter()

    rows = make_rows(
        "UNIVR",
        29,
        value_fn=lambda i: i,
    )

    rows[-1]["subject"] = rows[0]["subject"]

    with pytest.raises(ValueError):
        m.one_stratum_subject_bootstrap(
            rows,
            value_key="value",
            dataset="UNIVR",
            replicates=10,
        )


def test_wrong_dataset_size_rejected():
    m = load_adapter()

    rows = make_rows(
        "KFALL",
        31,
        value_fn=lambda i: i,
    )

    with pytest.raises(ValueError):
        m.one_stratum_subject_bootstrap(
            rows,
            value_key="value",
            dataset="KFALL",
            replicates=10,
        )


def test_adapter_contains_no_outer_result_path_or_cli():
    source = ADAPTER.read_text()

    assert "phase4h_sensor_fi_outer_v2_execution_v1" not in source
    assert "subject_condition_metrics.jsonl" not in source
    assert "argparse" not in source
    assert "__main__" not in source
