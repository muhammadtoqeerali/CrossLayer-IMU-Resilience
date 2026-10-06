import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5ac_compute_fi_cc_pre_outcome_execution_gate_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status():
    x = load(CONFIG)

    assert (
        x["status"]
        == "FROZEN_PRE_OUTCOME_CC_EXECUTION_GATE"
    )


def test_all_dependencies_are_hash_bound():
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

        assert (
            sha(path)
            == rec["sha256"]
        )


def test_scope_is_c0_cc_only():
    x = load(CONFIG)

    scope = x[
        "analysis_scope"
    ]

    assert scope[
        "regimes"
    ] == [
        "C0",
        "CC",
    ]

    assert scope[
        "CSC"
    ] is False

    assert scope[
        "OnField"
    ] is False


def test_exact_estate_cardinality():
    x = load(CONFIG)

    c = x[
        "accepted_estate_cardinality"
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
        "clean_model_window_records"
    ] == 1642980

    assert c[
        "faulted_model_window_records"
    ] == 29798820

    assert c[
        "total_model_window_records"
    ] == 31441800


def test_mixed_executor_provenance_cardinality():
    x = load(CONFIG)

    c = x[
        "accepted_estate_cardinality"
    ]

    assert c[
        "phase5m_canary_clean_caches"
    ] == 1

    assert c[
        "phase5m_canary_fault_shards"
    ] == 1

    assert c[
        "phase5r_clean_caches"
    ] == 365

    assert c[
        "phase5r_fault_shards"
    ] == 731


def test_future_input_paths_are_frozen_but_unaccessed():
    x = load(CONFIG)

    outer = x[
        "future_input_roots"
    ][
        "accepted_outer_result_root"
    ]

    assert outer[
        "accessed_during_phase5ac"
    ] is False

    assert outer[
        "stat_performed_during_phase5ac"
    ] is False

    assert outer[
        "listed_during_phase5ac"
    ] is False

    dataset = x[
        "future_input_roots"
    ][
        "outer_dataset_root"
    ]

    assert dataset[
        "payload_accessed_during_phase5ac"
    ] is False

    assert dataset[
        "label_array_loaded_during_phase5ac"
    ] is False


def test_required_executor_integrity_rules():
    x = load(CONFIG)

    rules = x[
        "required_executor_behavior"
    ]

    required_true = (
        "must_bind_exact_phase5z_protocol",
        "must_bind_exact_phase5aa_analyzer",
        "must_bind_exact_phase5ab_io_runner",
        "must_verify_artifact_hashes_before_jsonl_parse",
        "must_verify_phase5m_canary_against_frozen_acceptance",
        "must_verify_phase5r_artifacts_against_success_metadata",
        "must_group_faults_by_outer_instance_id",
        "must_not_mix_outer_instance_ids",
        "must_enforce_persistent_onset_to_end_suffix",
        "must_apply_only_frozen_validation_thresholds",
        "must_preserve_phase4_event_metric_semantics",
        "must_produce_paired_clean_counterfactual",
        "must_use_equal_seed_weight_within_subject",
        "must_use_subject_primary_uncertainty",
        "must_abort_on_integrity_failure",
        "must_not_skip_bad_records",
    )

    for key in required_true:
        assert rules[key] is True


def test_phase5ac_does_not_authorize_first_outcome_read():
    x = load(CONFIG)

    lifecycle = x[
        "execution_lifecycle"
    ]

    assert lifecycle[
        "final_outcome_executor_implemented"
    ] is False

    assert lifecycle[
        "final_outcome_executor_qualified"
    ] is False

    assert lifecycle[
        "accepted_outer_read_authorized_by_phase5ac_alone"
    ] is False

    assert lifecycle[
        "accepted_outer_label_load_authorized_by_phase5ac_alone"
    ] is False

    assert lifecycle[
        "outcome_executor_implementation_authorized"
    ] is True

    assert lifecycle[
        "outcome_executor_synthetic_static_qualification_authorized"
    ] is True


def test_activation_is_required_before_accepted_payload_read():
    x = load(CONFIG)

    policy = x[
        "first_outcome_read_policy"
    ]

    assert policy[
        "activation_after_executor_qualification_required"
    ] is True

    assert (
        policy[
            "first_accepted_prediction_row_read"
        ]
        == "not authorized in Phase 5AC"
    )

    assert (
        policy[
            "first_outer_label_array_load"
        ]
        == "not authorized in Phase 5AC"
    )

    assert (
        policy[
            "first_CC_metric"
        ]
        == "not authorized in Phase 5AC"
    )


def test_phase5ac_boundary_all_false():
    x = load(CONFIG)

    assert all(
        value is False
        for value in x[
            "phase5ac_boundary"
        ].values()
    )
