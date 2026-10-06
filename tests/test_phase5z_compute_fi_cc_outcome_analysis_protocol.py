import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5z_compute_fi_cc_outcome_analysis_protocol_v1.json"
)


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def test_status_and_boundary():
    x = load(CONFIG)

    assert (
        x["status"]
        == "FROZEN_PROSPECTIVE_CC_OUTCOME_ANALYSIS_PROTOCOL"
    )

    assert x[
        "analysis_boundary"
    ][
        "allowed_regimes"
    ] == [
        "C0",
        "CC",
    ]

    assert x[
        "analysis_boundary"
    ][
        "CSC_allowed"
    ] is False

    assert all(
        value is False
        for value in x[
            "freeze_boundary"
        ].values()
    )


def test_frozen_dependency_hashes():
    x = load(CONFIG)

    for rec in x[
        "frozen_dependencies"
    ].values():
        path = Path(
            rec["path"]
        )

        if not path.is_absolute():
            path = ROOT / path

        assert path.is_file()
        assert sha(path) == rec["sha256"]


def test_exact_threshold_matrix():
    x = load(CONFIG)

    matrix = x[
        "threshold_protocol"
    ][
        "matrix"
    ]

    assert len(matrix) == 45

    keys = {
        (
            row["seed"],
            row["fold"],
            row["operating_point"],
        )
        for row in matrix
    }

    assert keys == {
        (
            seed,
            fold,
            operating_point,
        )
        for seed in (
            42,
            123,
            2025,
        )
        for fold in range(
            1,
            6,
        )
        for operating_point in (
            "balanced",
            "low_false_alarm",
            "timely_150ms",
        )
    }


def test_embedded_thresholds_equal_frozen_csv():
    x = load(CONFIG)

    rec = x[
        "frozen_dependencies"
    ][
        "validation_selected_thresholds"
    ]

    path = Path(
        rec["path"]
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    expected = sorted(
        [
            {
                "seed":
                    int(row["seed"]),

                "fold":
                    int(row["fold"]),

                "operating_point":
                    row["operating_point"],

                "threshold":
                    float(row["threshold"]),

                "required_consecutive":
                    int(
                        row[
                            "required_consecutive"
                        ]
                    ),
            }
            for row in rows
        ],
        key=lambda row: (
            row["seed"],
            row["fold"],
            row["operating_point"],
        ),
    )

    assert (
        x[
            "threshold_protocol"
        ][
            "matrix"
        ]
        == expected
    )


def test_class_probability_adapter_is_identical():
    x = load(CONFIG)

    c = x[
        "class_semantics"
    ]

    assert c[
        "falling_class_index"
    ] == 1

    assert c[
        "phase5m_clean_field"
    ] == "clean_softmax_values"

    assert c[
        "phase5m_fault_field"
    ] == "faulted_softmax_values"

    assert c[
        "phase5r_clean_field"
    ] == "clean_softmax_values"

    assert c[
        "phase5r_fault_field"
    ] == "faulted_softmax_values"

    assert c[
        "softmax_reconstruction_from_logits"
    ] is False


def test_mixed_provenance_cardinality():
    x = load(CONFIG)

    m = x[
        "mixed_executor_adapter"
    ]

    assert m[
        "phase5m_fault_shards"
    ] == 1

    assert m[
        "phase5m_clean_caches"
    ] == 1

    assert m[
        "phase5r_fault_shards"
    ] == 731

    assert m[
        "phase5r_clean_caches"
    ] == 365

    assert m[
        "probability_semantics_identical"
    ] is True


def test_fault_scenario_is_one_outer_identity():
    x = load(CONFIG)

    s = x[
        "scenario_reconstruction"
    ]

    assert s[
        "fault_scenario"
    ][
        "primary_identity"
    ] == "outer_instance_id"

    assert s[
        "fault_scenario"
    ][
        "mixing_different_outer_instance_ids_in_one_trial"
    ] is False

    assert s[
        "paired_clean_counterfactual"
    ][
        "required"
    ] is True


def test_exact_trigger_semantics():
    x = load(CONFIG)

    assert x[
        "threshold_protocol"
    ][
        "comparison_operator"
    ] == ">="

    assert x[
        "event_decision_semantics"
    ][
        "confirmation_index"
    ] == (
        "trigger_start + "
        "required_consecutive - 1"
    )


def test_historical_truth_exception_is_preserved():
    x = load(CONFIG)

    g = x[
        "label_and_truth_protocol"
    ]

    assert (
        g[
            "historical_exception"
        ]
        == "KFALL_106_T27_R05"
    )

    assert g[
        "historical_exception_stored_labels_rewritten"
    ] is False

    assert g[
        "historical_exception_truth_from_frozen_risk_record"
    ] is True


def test_timing_contract():
    x = load(CONFIG)

    t = x[
        "trial_time_reconstruction"
    ]

    assert t[
        "sampling_rate_hz"
    ] == 100.0

    assert t[
        "window_samples"
    ] == 30

    assert t[
        "stride_samples"
    ] == 15

    assert t[
        "framecounter_authoritative"
    ] is True

    assert t[
        "annotated_fall_event_count"
    ] == 2919

    assert t[
        "source_faithful_event_count"
    ] == 2918

    assert t[
        "historical_fallback_event_count"
    ] == 1

    assert t[
        "runtime_adjusted_lead_in_phase5_cc"
    ] is False


def test_primary_metric_set_and_no_composite():
    x = load(CONFIG)

    m = x[
        "metric_contract"
    ]

    assert m[
        "primary_reporting_metrics"
    ] == [
        "balanced_accuracy",
        "fall_recall",
        "false_triggers_per_activity_hour",
        "recall_by_150ms",
        "median_trigger_lead_ms",
    ]

    assert m[
        "composite_robustness_score"
    ] is False


def test_aggregation_prevents_record_count_weighting():
    x = load(CONFIG)

    a = x[
        "aggregation_contract"
    ]

    assert a[
        "window_independence_assumed"
    ] is False

    assert a[
        "fault_instance_independence_for_inference_assumed"
    ] is False

    assert a[
        "aggregate_CC_summary"
    ][
        "scalar_score"
    ] is False

    assert a[
        "aggregate_CC_summary"
    ][
        "ptq_only_targets_in_cross_variant_comparison"
    ] is False


def test_subject_cluster_uncertainty():
    x = load(CONFIG)

    u = x[
        "uncertainty_contract"
    ]

    assert (
        u["primary_unit"]
        == "outer subject"
    )

    assert (
        u["bootstrap_replicates"]
        == 10000
    )

    assert (
        u["confidence_level"]
        == 0.95
    )

    assert (
        u["rng_seed"]
        == 20261006
    )

    assert (
        u["paired_delta_bootstrap"]
        is True
    )


def test_complete_estate_cardinality():
    x = load(CONFIG)

    i = x[
        "integrity_and_abort_rules"
    ]

    assert i[
        "expected_fault_shards"
    ] == 732

    assert i[
        "expected_clean_caches"
    ] == 366

    assert i[
        "expected_artifact_directories"
    ] == 1098

    assert i[
        "expected_outer_instance_ids"
    ] == 20170008

    assert i[
        "expected_total_records"
    ] == 31441800

    assert i[
        "skip_bad_records"
    ] is False


def test_no_phase5z_outcome_execution():
    x = load(CONFIG)

    boundary = x[
        "freeze_boundary"
    ]

    assert boundary[
        "phase5_prediction_jsonl_opened"
    ] is False

    assert boundary[
        "phase5_prediction_json_deserialized"
    ] is False

    assert boundary[
        "outer_label_array_loaded"
    ] is False

    assert boundary[
        "threshold_applied_to_outer"
    ] is False

    assert boundary[
        "CC_metric_computed"
    ] is False

    assert boundary[
        "aggregate_CC_result_generated"
    ] is False

    assert boundary[
        "CSC_result_generated"
    ] is False

    assert boundary[
        "model_forward_executed"
    ] is False

    assert boundary[
        "fault_execution_executed"
    ] is False
