import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_execution_v2_historical_truth.json"
)

SHARDS = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json"
)


def cfg():
    return json.loads(
        CONFIG.read_text()
    )


def plan():
    return json.loads(
        SHARDS.read_text()
    )


def test_v2_preserves_v1_and_changes_only_truth_contract():
    d = cfg()

    assert d[
        "supersedes"
    ][
        "v1_preserved_unchanged"
    ] is True

    assert d[
        "ground_truth_contract"
    ][
        "mismatch_count"
    ] == 1


def test_historical_truth_counts():
    d = cfg()

    truth = d[
        "ground_truth_contract"
    ][
        "authoritative_truth_counts"
    ]

    assert truth == {
        "Activity":
            3390,

        "Falling":
            2919,

        "total":
            6309,
    }


def test_stored_label_counts_are_explicitly_non_authoritative():
    d = cfg()

    stored = d[
        "outer_population"
    ][
        "stored_label_trial_inventory"
    ]

    assert stored[
        "Activity"
    ] == 3391

    assert stored[
        "Falling"
    ] == 2918


def test_exact_exception():
    d = cfg()

    x = d[
        "ground_truth_contract"
    ][
        "mismatch_trial"
    ]

    assert x[
        "event_id"
    ] == "KFALL_106_T27_R05"

    assert x[
        "outer_fold"
    ] == 4

    assert x[
        "retained_falling_window_count"
    ] == 0

    assert x[
        "risk_record_present"
    ] is True

    assert x[
        "historical_evaluator_true_event"
    ] == "Falling"


def test_exception_has_no_valid_trigger_under_retained_labels():
    d = cfg()

    x = d[
        "historical_event_rule"
    ][
        "known_exception_trigger_semantics"
    ]

    assert x[
        "true_event"
    ] == "FALLING"

    assert x[
        "retained_segment_labels"
    ] == "all Activity"

    assert x[
        "detected_event"
    ] is False

    assert x[
        "sensor_lead_ms"
    ] is None

    assert x[
        "excluded_from_truth_denominator"
    ] is False

    assert x[
        "relabeled"
    ] is False


def test_v2_shard_cardinality_and_workload_unchanged():
    d = plan()

    assert d[
        "shard_count"
    ] == 793

    assert d[
        "subject_count"
    ] == 61

    assert d[
        "expected_totals_from_shards"
    ] == {
        "model_window_evaluations":
            479750160,

        "subject_condition_rows":
            320616,

        "unique_fault_instances":
            42766632,
    }


def test_v2_plan_truth_counts():
    d = plan()

    assert d[
        "global_stored_label_trial_inventory"
    ] == {
        "Activity":
            3391,

        "Falling":
            2918,

        "total":
            6309,
    }

    assert d[
        "global_historical_evaluator_trial_truth_inventory"
    ] == {
        "Activity":
            3390,

        "Falling":
            2919,

        "total":
            6309,
    }


def test_no_outer_execution_in_static_plan():
    d = plan()

    b = d[
        "scientific_boundary"
    ]

    assert b[
        "model_loaded"
    ] is False

    assert b[
        "model_predictions_computed"
    ] is False

    assert b[
        "fault_injection_executed"
    ] is False

    assert b[
        "outer_model_outcomes_seen"
    ] is False

    assert b[
        "outer_fault_outcomes_seen"
    ] is False
