import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_execution_v1.json"
)

PLANNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_execution_plan.py"
)

SHARDS = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_shards_v1.json"
)


def config():
    return json.loads(
        CONFIG.read_text()
    )


def shards():
    return json.loads(
        SHARDS.read_text()
    )


def test_partition_is_outer_test_only():
    d = config()

    assert d[
        "partition"
    ] == "outer_test"

    assert d[
        "scientific_boundary"
    ][
        "outer_model_predictions_seen"
    ] is False

    assert d[
        "scientific_boundary"
    ][
        "outer_fault_results_seen"
    ] is False


def test_full_fault_matrix():
    d = config()

    f = d[
        "fault_matrix"
    ]

    assert len(
        f[
            "fault_families_in_order"
        ]
    ) == 12

    assert f[
        "severity_levels"
    ] == [
        "L1",
        "L2",
        "L3",
    ]

    assert f[
        "expected_window_instances_per_parent"
    ] == 153

    assert f[
        "expected_sequence_instances_per_trial"
    ] == 138

    assert f[
        "expected_unique_fault_instances_total"
    ] == 42766632


def test_all_operating_points_preserved():
    d = config()

    o = d[
        "operating_points"
    ]

    assert o[
        "required_names"
    ] == [
        "balanced",
        "low_false_alarm",
        "timely_150ms",
    ]

    assert o[
        "all_three_evaluated"
    ] is True

    assert o[
        "selection_or_averaging_across_operating_points"
    ] is False

    assert o[
        "no_outer_threshold_reselection"
    ] is True


def test_trial_level_historical_rule():
    d = config()

    h = d[
        "historical_event_rule"
    ]

    assert h[
        "confusion_unit"
    ] == "trial"

    assert h[
        "consecutive_rule_resets_at_trial_boundary"
    ] is True

    assert h[
        "pooled_window_confusion_as_primary"
    ] is False


def test_timing_route_separates_slice_and_physical_position():
    d = config()

    t = d[
        "timing_contract"
    ]

    assert "fall_start_frame" in t[
        "source_slice_route"
    ]

    assert "zero-based" in t[
        "physical_timing_position"
    ]

    assert t[
        "trigger_lead_uses_confirmation_window"
    ] is True


def test_subject_family_sharding():
    d = config()

    s = d[
        "sharding"
    ]

    assert s[
        "execution_blocks_per_subject"
    ] == 13

    assert s[
        "expected_shard_count"
    ] == 793

    assert s[
        "all_three_checkpoint_seeds_inside_same_shard"
    ] is True

    assert s[
        "both_model_variants_inside_same_shard"
    ] is True


def test_compact_output_preserves_required_statistics():
    d = config()

    o = d[
        "output_policy"
    ]

    assert o[
        "raw_per_window_probability_persistence_required"
    ] is False

    for field in [
        "TP",
        "FN",
        "TN",
        "FP",
    ]:
        assert field in o[
            "condition_row_classification_fields"
        ]

    for field in [
        "eligible_event_count",
        "detected_event_count",
        "missed_event_count",
        "median_sensor_lead_ms",
        "q25_sensor_lead_ms",
        "q75_sensor_lead_ms",
    ]:
        assert field in o[
            "condition_row_event_fields"
        ]


def test_resume_is_hash_guarded():
    d = config()

    r = d[
        "resume_semantics"
    ]

    assert r[
        "success_marker_written_last"
    ] is True

    assert r[
        "final_shard_directory_is_immutable"
    ] is True

    assert r[
        "scientific_parameter_change_on_resume"
    ] is False


def test_no_heldout_retuning():
    d = config()

    h = d[
        "heldout_governance"
    ]

    for key in [
        "outer_result_may_change_fault_severity",
        "outer_result_may_change_fault_sampling",
        "outer_result_may_change_threshold",
        "outer_result_may_change_aggregation",
        "outer_result_may_change_reporting",
        "outer_result_may_select_fault_families",
        "outer_result_may_select_operating_point",
    ]:
        assert h[
            key
        ] is False


def test_static_shard_plan_cardinality():
    d = shards()

    assert d[
        "status"
    ] == "FROZEN_PRE_OUTER_EXECUTION"

    assert d[
        "shard_count"
    ] == 793

    assert d[
        "subject_count"
    ] == 61

    assert len(
        d[
            "shards"
        ]
    ) == 793

    assert len({
        row[
            "shard_id"
        ]
        for row
        in d[
            "shards"
        ]
    }) == 793


def test_each_subject_has_exactly_13_blocks():
    d = shards()

    counts = {}

    for row in d[
        "shards"
    ]:
        counts.setdefault(
            row[
                "subject"
            ],
            set(),
        ).add(
            row[
                "block"
            ]
        )

    assert len(
        counts
    ) == 61

    assert all(
        len(
            blocks
        )
        == 13
        for blocks
        in counts.values()
    )


def test_static_workload_totals():
    d = shards()

    totals = d[
        "expected_totals_from_shards"
    ]

    assert totals[
        "unique_fault_instances"
    ] == 42766632

    assert totals[
        "model_window_evaluations"
    ] == 479750160

    assert totals[
        "subject_condition_rows"
    ] == 320616


def test_plan_itself_did_not_execute_outer_models_or_faults():
    d = shards()
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
        "OnField_used"
    ] is False
