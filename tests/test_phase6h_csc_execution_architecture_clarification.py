from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/phase6h_csc_execution_architecture_clarification_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_6h_csc_execution_architecture_clarification_freeze_v1.json"
)


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_status_is_preimplementation_and_execution_disabled():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg["status"]
        == "FROZEN_PRE_IMPLEMENTATION_EXECUTION_ARCHITECTURE_CLARIFICATION"
    )

    assert cfg[
        "scientific_change"
    ] is False

    assert cfg[
        "execution_authorized"
    ] is False


def test_frozen_scientific_cardinalities_unchanged():
    cfg = load_json(
        CONFIG
    )

    surface = cfg[
        "frozen_scientific_surface"
    ]

    assert surface[
        "model_independent_pair_count"
    ] == 4237835

    assert surface[
        "seed_variant_expanded_pair_member_count"
    ] == 21793038

    assert surface[
        "sensor_reference_member_window_count"
    ] == 35167107

    assert surface[
        "compute_faulted_member_window_count"
    ] == 411540372

    assert surface[
        "simultaneous_csc_overlap_member_window_count"
    ] == 19926021

    assert surface[
        "zero_temporal_overlap_pair_member_count"
    ] == 5234355

    assert surface[
        "subject_family_shard_count"
    ] == 732


def test_runtime_work_unit_is_subject_family_shard():
    cfg = load_json(
        CONFIG
    )

    work = cfg[
        "runtime_work_unit"
    ]

    assert (
        work["unit"]
        == "one subject x one sensor-family shard"
    )

    assert work[
        "count"
    ] == 732

    assert work[
        "all_severities_in_same_shard"
    ] is True

    assert work[
        "shard_is_unit_of_concurrency"
    ] is True

    assert work[
        "intra_shard_model_mutation_concurrency"
    ] is False


def test_clean_reference_is_phase5_reuse_only():
    cfg = load_json(
        CONFIG
    )

    clean = cfg[
        "clean_reference_ownership"
    ]

    assert clean[
        "phase6_c0_model_forwards"
    ] == 0

    assert clean[
        "owner"
    ] == "frozen Phase-5 clean-cache estate"

    assert (
        clean[
            "missing_or_invalid_clean_cache"
        ]
        == "ABORT_SHARD_NO_PHASE6_RECOMPUTE"
    )

    assert clean[
        "copy_clean_records_into_phase6_artifact"
    ] is False


def test_sensor_reference_cache_scope_and_windows():
    cfg = load_json(
        CONFIG
    )

    sensor = cfg[
        "sensor_reference_ownership"
    ]

    assert (
        sensor[
            "cache_scope"
        ]
        == "one selected sensor instance x one model member"
    )

    assert sensor[
        "reference_forward_occurs_before_compute_fault_sequence"
    ] is True

    assert sensor[
        "reference_forward_uses_sensor_conditioned_input"
    ] is True

    assert (
        "sensor-exposed"
        in sensor[
            "reference_windows"
        ]
    )


def test_reference_rule_depends_only_on_sensor_activity_at_window():
    cfg = load_json(
        CONFIG
    )

    rule = cfg[
        "per_compute_window_reference_rule"
    ]

    assert (
        "Phase-6 sensor-reference"
        in rule[
            "sensor_active_at_compute_window"
        ]
    )

    assert (
        "Phase-5 clean-cache"
        in rule[
            "sensor_inactive_at_compute_window"
        ]
    )


def test_composition_order_is_sensor_then_compute():
    cfg = load_json(
        CONFIG
    )

    composition = cfg[
        "sensor_compute_composition"
    ]

    assert composition[
        "sensor_fault_applied_before_compute_fault"
    ] is True

    assert composition[
        "no_sensor_relocation"
    ] is True

    assert composition[
        "no_compute_relocation"
    ] is True

    assert (
        "sensor-conditioned"
        in composition[
            "compute_fault_input"
        ]
    )


def test_model_lifecycle_is_one_bundle_per_member_stream():
    cfg = load_json(
        CONFIG
    )

    lifecycle = cfg[
        "model_lifecycle"
    ]

    assert (
        lifecycle[
            "scope"
        ]
        == "one model bundle per shard x model_variant x checkpoint_seed stream"
    )

    assert lifecycle[
        "pair_members_executed_sequentially_within_model_stream"
    ] is True

    assert lifecycle[
        "ptq_weight_sequence_must_reset_clean_state_in_finally"
    ] is True

    assert lifecycle[
        "cross_shard_model_state_reuse"
    ] is False


def test_atomic_resume_contract_has_no_row_level_resume():
    cfg = load_json(
        CONFIG
    )

    atomic = cfg[
        "atomic_resume_contract"
    ]

    assert (
        atomic[
            "final_directory"
        ]
        == "{output_root}/shards/{shard_id}"
    )

    assert (
        atomic[
            "temporary_directory"
        ]
        == "{output_root}/shards/{shard_id}.partial"
    )

    assert atomic[
        "success_marker"
    ] == "_SUCCESS.json"

    assert atomic[
        "row_level_resume"
    ] is False

    assert (
        atomic[
            "recompute_partial_true"
        ]
        .startswith(
            "delete invalid final/partial artifact"
        )
    )


def test_output_file_set_and_uncompressed_canonical_jsonl():
    cfg = load_json(
        CONFIG
    )

    files = cfg[
        "output_files"
    ]

    assert files[
        "pair_members"
    ] == "pair_members.jsonl"

    assert files[
        "sensor_references"
    ] == "sensor_reference.jsonl"

    assert files[
        "csc_fault_windows"
    ] == "csc_fault.jsonl"

    assert files[
        "metadata"
    ] == "metadata.json"

    assert files[
        "success"
    ] == "_SUCCESS.json"

    assert files[
        "compression"
    ] == "none"


def test_record_schemas_bind_reference_and_overlap_fields():
    cfg = load_json(
        CONFIG
    )

    pair_fields = set(
        cfg[
            "pair_member_record_schema"
        ][
            "required_fields"
        ]
    )

    assert {
        "execution_request_id",
        "sensor_reference_cache_id",
        "phase5_clean_cache_id",
        "simultaneous_overlap_window_indices",
        "zero_temporal_overlap",
    }.issubset(
        pair_fields
    )

    fault_fields = set(
        cfg[
            "csc_fault_record_schema"
        ][
            "required_fields"
        ]
    )

    assert {
        "sensor_active_at_execution_window",
        "simultaneous_sensor_compute_active",
        "reference_kind",
        "mutation",
    }.issubset(
        fault_fields
    )


def test_zero_overlap_pairs_are_executed_and_recorded():
    cfg = load_json(
        CONFIG
    )

    zero = cfg[
        "zero_overlap_record_contract"
    ]

    assert zero[
        "pair_retained"
    ] is True

    assert zero[
        "pair_member_record_always_written"
    ] is True

    assert zero[
        "compute_fault_execution_still_runs"
    ] is True

    assert zero[
        "csc_fault_rows_written"
    ] is True

    assert zero[
        "simultaneous_overlap_window_indices"
    ] == []

    assert zero[
        "reference_kind_for_every_compute_window"
    ] == "phase5_clean_cache"

    for key in (
        "filter",
        "resample",
        "relocate",
        "replace",
        "rebalance",
    ):
        assert zero[
            key
        ] is False


def test_future_runtime_stays_separate_and_gate_required():
    cfg = load_json(
        CONFIG
    )

    interface = cfg[
        "future_runtime_interface"
    ]

    assert interface[
        "runtime_must_be_separate_new_file"
    ] is True

    assert interface[
        "must_not_modify_phase6e_metadata_executor"
    ] is True

    assert interface[
        "must_not_modify_phase6g_adapter"
    ] is True

    gate = cfg[
        "pre_execution_gate_required"
    ]

    assert gate[
        "required"
    ] is True

    assert gate[
        "must_exist_before_any_model_forward"
    ] is True

    assert gate[
        "outer_performance_outcomes_may_inform_gate"
    ] is False


def test_governance_is_pre_forward():
    cfg = load_json(
        CONFIG
    )

    governance = cfg[
        "governance"
    ]

    for key in (
        "outer_performance_outcomes_used",
        "validation_used",
        "onfield_used",
        "threshold_retuning",
        "checkpoint_selection",
        "resampling",
        "rebalancing",
        "pair_files_materialized",
        "outer_arrays_read",
        "model_loaded",
        "sensor_fault_executed",
        "compute_fault_executed",
        "csc_model_forward",
        "metrics_computed",
        "commit_created",
        "push_performed",
    ):
        assert governance[
            key
        ] is False


def test_upstream_frozen_hashes():
    cfg = load_json(
        CONFIG
    )

    for record in cfg[
        "frozen_dependencies"
    ].values():
        path = (
            ROOT
            / record[
                "path"
            ]
        )

        assert sha256_file(
            path
        ) == record[
            "sha256"
        ]


def test_manifest_binds_clarification_files():
    manifest = load_json(
        MANIFEST
    )

    assert (
        manifest[
            "status"
        ]
        == "FROZEN_PRE_IMPLEMENTATION_EXECUTION_ARCHITECTURE_CLARIFICATION"
    )

    assert manifest[
        "execution_authorized"
    ] is False

    for record in manifest[
        "frozen_files"
    ]:
        path = (
            ROOT
            / record[
                "path"
            ]
        )

        assert sha256_file(
            path
        ) == record[
            "sha256"
        ]
