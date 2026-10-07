from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase6d_csc_pairing_protocol_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def protocol():
    return load(
        CONFIG
    )


def test_status_and_phase():
    x = protocol()

    assert x["phase"] == "6D"

    assert (
        x["status"]
        == "FROZEN_PROSPECTIVE_CSC_PAIRING_PROTOCOL_PRE_EXECUTION_PLAN"
    )


def test_frozen_parent_hashes():
    x = protocol()[
        "frozen_dependencies"
    ]

    expected = {
        "configs/faults/phase4h_sensor_fi_instance_sampling_v3_causal_frame_loss.json":
            "543b648a09091225cfc80eaac5cd03d835a1727118f7730d00697150db205518",

        "configs/faults/phase4h_sensor_fi_severity_protocol_v1.json":
            "48b8927628f2580642c93ed9b492cf562a4ad40a21c7815bc3d5a93ca970b009",

        "configs/evaluation/phase4h_sensor_fi_outer_execution_v2_historical_truth.json":
            "5167985edd3f4bf03fbbb4cc8d196407ac5b172a65477613813a7eaaed53243b",

        "manifests/phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json":
            "a43df702c2694854e1ce79712af50a9bf91d0f24b40989241f604405f460024d",

        "manifests/phase_4h_sensor_fi_outer_execution_completion_v1.json":
            "587d608f1f9ff1addbdcec312e3312591e008f05459253b60f754667f209b077",

        "configs/faults/phase5a_compute_fi_representation_protocol_v1.json":
            "279d5c9ebd865ae0d147ec62d0cd04d2ffb5d58d9c80544ee6c913f9a1964bd5",

        "configs/evaluation/phase5d_compute_fi_outer_protocol_v1.json":
            "0ba30c9d336433748378f8d3eed1ed45b673fb9690250b1bd3bbbc79f88c9c36",

        "manifests/phase_5e_compute_fi_outer_execution_plan_v1.json":
            "95aecd14b4aa8dce70492f0c4d6d50a8bf5a8dafb2ec962aff9c033b8472bb54",

        "configs/evaluation/phase5aj_phase5_compute_fi_final_synthesis_v1.json":
            "57e6011b2e6a9d43876be96313b6ac222f537b497afe9804316e27ffca9cb1cd",
    }

    assert x == expected

    for path, expected_sha in expected.items():
        assert sha(
            ROOT / path
        ) == expected_sha


def test_sensor_surface_is_complete():
    x = protocol()[
        "sensor_surface"
    ]

    assert x[
        "family_count"
    ] == 12

    assert x[
        "severity_levels"
    ] == [
        "L1",
        "L2",
        "L3",
    ]

    assert len(
        x[
            "sequence_families"
        ]
    ) == 7

    assert len(
        x[
            "window_value_families"
        ]
    ) == 5


def test_exact_sparse_sensor_cardinality():
    x = protocol()[
        "sensor_surface"
    ][
        "exact_selected_instance_cardinality_before_compute_assignment"
    ]

    assert x[
        "stored_window_parent_instances"
    ] == 4107450

    assert x[
        "source_trial_parent_instances"
    ] == 132489

    assert x[
        "total"
    ] == 4239939


def test_compute_surface_is_complete():
    x = protocol()[
        "compute_surface"
    ]

    assert x[
        "target_count"
    ] == 14

    assert x[
        "common_fp32_ptq_target_count"
    ] == 10

    assert x[
        "ptq_only_target_count"
    ] == 4

    assert x[
        "persistence_order"
    ] == [
        "transient_one_inference",
        "persistent_from_onset_until_trial_end",
    ]

    assert x[
        "compute_stratum_count"
    ] == 28

    assert len(
        x[
            "compute_strata"
        ]
    ) == 28


def test_compute_assignment_is_prospective_and_not_rebalanced():
    x = protocol()[
        "compute_stratum_assignment"
    ]

    assert x[
        "all_28_strata_must_have_nonzero_plan_coverage"
    ] is True

    assert x[
        "plan_must_report_exact_count_per_stratum"
    ] is True

    assert x[
        "rebalancing_after_counts_are_seen"
    ] is False

    assert x[
        "outer_performance_based_reassignment"
    ] is False


def test_causal_join_and_exact_cc_coordinate_reuse():
    x = protocol()[
        "causal_parent_join"
    ]

    assert x[
        "no_cross_subject_pairing"
    ] is True

    assert x[
        "no_cross_fold_pairing"
    ] is True

    assert x[
        "no_cross_trial_pairing"
    ] is True

    assert x[
        "sensor_corruption_precedes_compute_fault"
    ] is True

    assert x[
        "sensor_corrupted_parent_must_generate_the_compute_faulted_inference"
    ] is True

    assert x[
        "persistent_compute_pairing"
    ][
        "temporal_overlap_filtering_allowed"
    ] is False

    assert x[
        "persistent_compute_pairing"
    ][
        "temporal_overlap_status_must_be_recorded"
    ] is True


def test_four_regime_comparator_contract():
    x = protocol()[
        "comparator_contract"
    ]

    for key in (
        "C0",
        "CS",
        "CC",
        "CSC",
    ):
        assert key in x

    assert x[
        "same_checkpoint_required"
    ] is True

    assert x[
        "same_fold_required"
    ] is True

    assert x[
        "same_parent_required"
    ] is True

    assert x[
        "same_validation_selected_threshold_required"
    ] is True

    assert x[
        "threshold_retuning_allowed"
    ] is False


def test_subject_is_primary_uncertainty_unit():
    x = protocol()[
        "analysis_governance"
    ]

    assert x[
        "primary_uncertainty_unit"
    ] == "outer_subject"

    assert x[
        "fault_pairs_are_independent_subjects"
    ] is False

    assert x[
        "same_subject_equal_weighting_required"
    ] is True


def test_no_outer_feedback_selection():
    x = protocol()[
        "analysis_governance"
    ]

    forbidden = (
        "outer_result_based_model_selection",
        "outer_result_based_sensor_family_selection",
        "outer_result_based_sensor_severity_selection",
        "outer_result_based_compute_target_selection",
        "outer_result_based_bit_selection",
        "outer_result_based_persistence_selection",
        "outer_result_based_pair_resampling",
        "threshold_retuning",
        "checkpoint_selection",
    )

    for key in forbidden:
        assert x[key] is False


def test_no_execution_or_physical_claim():
    x = protocol()

    assert x[
        "claim_boundary"
    ][
        "CSC_outer_execution_authorized_by_this_protocol"
    ] is False

    assert x[
        "claim_boundary"
    ][
        "execution_plan_must_be_frozen_first"
    ] is True

    assert x[
        "claim_boundary"
    ][
        "OnField_used"
    ] is False

    assert x[
        "claim_boundary"
    ][
        "MCU_fault_equivalence_claim"
    ] is False

    assert x[
        "claim_boundary"
    ][
        "physical_fault_equivalence_claim"
    ] is False

    assert x[
        "freeze_boundary"
    ][
        "CSC_fault_pairs_materialized"
    ] is False

    assert x[
        "freeze_boundary"
    ][
        "CSC_model_forward_executed"
    ] is False
