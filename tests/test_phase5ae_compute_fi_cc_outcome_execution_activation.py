import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACTIVATION = (
    ROOT
    / "configs/evaluation/"
    "phase5ae_compute_fi_cc_outcome_execution_activation_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_activation_status():
    x = load(
        ACTIVATION
    )

    assert (
        x["status"]
        == "FROZEN_FINAL_CC_OUTCOME_EXECUTION_ACTIVATION"
    )

    assert (
        x["activation_kind"]
        == "one_way_prospective_outcome_activation"
    )


def test_all_file_dependencies_are_hash_bound():
    x = load(
        ACTIVATION
    )

    for rec in x[
        "frozen_dependencies"
    ].values():
        if (
            not isinstance(
                rec,
                dict,
            )
            or "path" not in rec
            or "sha256" not in rec
        ):
            continue

        path = Path(
            rec["path"]
        )

        if not path.is_absolute():
            path = ROOT / path

        assert path.is_file()

        assert (
            sha(path)
            == rec["sha256"]
        )


def test_exact_executor_is_activated():
    x = load(
        ACTIVATION
    )

    dep = x[
        "frozen_dependencies"
    ][
        "phase5ad_final_executor"
    ]

    auth = x[
        "authorized_execution"
    ]

    assert (
        auth["executor_path"]
        == dep["path"]
    )

    assert (
        auth["executor_sha256"]
        == dep["sha256"]
    )


def test_only_frozen_outcome_operations_are_authorized():
    x = load(
        ACTIVATION
    )

    a = x[
        "authorized_execution"
    ]

    assert a[
        "accepted_prediction_read"
    ] is True

    assert a[
        "accepted_outer_label_load"
    ] is True

    assert a[
        "frozen_threshold_application"
    ] is True

    assert a[
        "CC_metric_computation"
    ] is True

    assert a[
        "aggregate_CC_result_generation"
    ] is True

    assert a[
        "CSC_result_generation"
    ] is False

    assert a[
        "OnField_access"
    ] is False

    assert a[
        "model_load"
    ] is False

    assert a[
        "model_forward"
    ] is False

    assert a[
        "new_fault_execution"
    ] is False

    assert a[
        "threshold_retuning"
    ] is False

    assert a[
        "checkpoint_selection"
    ] is False

    assert a[
        "fault_resampling"
    ] is False


def test_exact_estate_is_frozen():
    x = load(
        ACTIVATION
    )

    c = x[
        "accepted_estate"
    ]

    assert c[
        "subjects"
    ] == 61

    assert c[
        "trials"
    ] == 6309

    assert c[
        "stored_windows"
    ] == 273830

    assert c[
        "clean_caches"
    ] == 366

    assert c[
        "fault_shards"
    ] == 732

    assert c[
        "artifact_directories"
    ] == 1098

    assert c[
        "outer_instance_ids"
    ] == 20170008

    assert c[
        "total_model_window_records"
    ] == 31441800


def test_shard_order_is_frozen():
    x = load(
        ACTIVATION
    )

    order = x[
        "execution_order"
    ]

    assert (
        order["shard_source"]
        == "Phase-5E shards list"
    )

    assert (
        order["order"]
        == "exact Phase-5E manifest order"
    )

    assert (
        order["expected_shard_count"]
        == 732
    )

    assert order[
        "aggregate_only_after_all_shards"
    ] is True


def test_resume_cannot_be_outcome_driven():
    x = load(
        ACTIVATION
    )

    r = x[
        "resume_policy"
    ]

    assert r[
        "reuse_only_if_success_marker_valid"
    ] is True

    assert r[
        "partial_or_unverified_output_reuse"
    ] is False

    assert r[
        "manual_metric_based_shard_skipping"
    ] is False

    assert r[
        "protocol_change_on_resume"
    ] is False


def test_one_way_rule_is_frozen():
    x = load(
        ACTIVATION
    )

    rule = x[
        "one_way_rule"
    ]

    assert rule[
        "outcome_disappointment_is_not_a_repair_reason"
    ] is True

    assert x[
        "authorized_execution"
    ][
        "protocol_change_after_first_payload_read"
    ] is False

    assert x[
        "authorized_execution"
    ][
        "output_driven_rerun_with_changed_rules"
    ] is False


def test_activation_itself_did_not_cross_outcome_boundary():
    x = load(
        ACTIVATION
    )

    assert all(
        value is False
        for value in x[
            "activation_boundary"
        ].values()
    )
