import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

V3 = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_instance_sampling_v3_qualification_v1.json"
)

AGG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_aggregation_v1.json"
)

AGG_REBIND = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_evaluation_aggregation_sampling_v3_rebind_v1.json"
)

REPORT_V1 = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_robustness_reporting_v1.json"
)

REPORT_V2 = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_robustness_reporting_v2_sampling_v3.json"
)

REPORT_IMPL_V2 = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_reporting_v2.py"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_reporting_v2",
        REPORT_IMPL_V2,
    )

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m


def test_sampling_v3_is_qualified():
    d = load(V3)

    assert d["status"] == (
        "QUALIFIED_METADATA_REPLAY_CAUSAL_FRAME_LOSS"
    )

    assert d["acceptance"]["v3_frame_loss_onset0_observed"] == 0


def test_aggregation_is_unchanged_and_rebound():
    d = load(AGG_REBIND)

    assert d["status"] == (
        "QUALIFIED_UNCHANGED_AGGREGATION_REBOUND_TO_SAMPLING_V3"
    )

    c = d["scientific_change"]

    assert c["aggregation_math_changed"] is False
    assert c["primary_inferential_unit_changed"] is False
    assert c["bootstrap_changed"] is False
    assert c["metric_definitions_changed"] is False
    assert c["variant_weighting_changed"] is False
    assert c["seed_weighting_changed"] is False
    assert c["fold_handling_changed"] is False
    assert c["OnField_policy_changed"] is False
    assert c["only_lineage_binding_changed"] is True


def test_reporting_v2_gate_points_to_sampling_v3():
    d = load(REPORT_V2)

    gates = d[
        "execution_interpretability_gate"
    ][
        "required"
    ]

    assert "qualified_sampling_v3_hash_matches" in gates
    assert "qualified_sampling_v2_hash_matches" not in gates


def test_reporting_single_change_only():
    d = load(REPORT_V2)
    c = d["single_change_from_v1"]

    assert c["v1_value"] == (
        "qualified_sampling_v2_hash_matches"
    )

    assert c["v2_value"] == (
        "qualified_sampling_v3_hash_matches"
    )

    for key in [
        "performance_margin_policy_changed",
        "required_model_variants_changed",
        "fault_matrix_changed",
        "primary_metrics_changed",
        "secondary_metrics_changed",
        "direction_status_rules_changed",
        "multiple_condition_policy_changed",
        "mandatory_reporting_changed",
        "clean_reference_reporting_changed",
        "missingness_policy_changed",
        "claim_policy_changed",
        "heldout_governance_changed",
        "OnField_policy_changed",
        "publication_record_changed",
        "arbitrary_practical_margin_introduced",
    ]:
        assert c[key] is False


def test_reporting_science_matches_v1():
    v1 = load(REPORT_V1)
    v2 = load(REPORT_V2)

    for key in [
        "performance_margin_policy",
        "required_model_variants",
        "required_fault_matrix",
        "primary_metrics",
        "secondary_metrics",
        "condition_level_direction_status",
        "important_interpretation_rule",
        "multiple_condition_policy",
        "mandatory_reporting",
        "clean_reference_reporting",
        "missingness_policy",
        "claim_policy",
        "heldout_governance",
        "OnField_policy",
        "publication_minimum_record",
    ]:
        assert v2[key] == v1[key]


def test_reporting_helper_requires_v3_gate():
    m = load_impl()

    gates = {
        "frozen_input_contract_hash_matches": True,
        "frozen_severity_protocol_hash_matches": True,
        "qualified_sampling_v3_hash_matches": True,
        "qualified_aggregation_protocol_hash_matches": True,
        "all_required_checkpoints_and_model_variants_accounted_for": True,
        "all_fault_instances_have_valid_replay_ids": True,
        "no_outer_test_tuning_occurred": True,
        "no_OnField_fault_generation_occurred": True,
        "all_required_result_dimensions_present": True,
        "all_missing_or_undefined_metrics_explicitly_accounted_for": True,
    }

    assert m.execution_interpretability(
        gates
    ) == "EXECUTION_INTERPRETABLE"


def test_old_sampling_v2_gate_is_rejected_as_incomplete():
    m = load_impl()

    gates = {
        "frozen_input_contract_hash_matches": True,
        "frozen_severity_protocol_hash_matches": True,
        "qualified_sampling_v2_hash_matches": True,
        "qualified_aggregation_protocol_hash_matches": True,
        "all_required_checkpoints_and_model_variants_accounted_for": True,
        "all_fault_instances_have_valid_replay_ids": True,
        "no_outer_test_tuning_occurred": True,
        "no_OnField_fault_generation_occurred": True,
        "all_required_result_dimensions_present": True,
        "all_missing_or_undefined_metrics_explicitly_accounted_for": True,
    }

    with pytest.raises(ValueError):
        m.execution_interpretability(
            gates
        )


def test_only_jitter_semantics_remain():
    d = load(REPORT_V2)

    assert d["remaining_pre_operator_governance"] == [
        "executable jitter semantics"
    ]

    assert d[
        "preexecution_governance_complete_after_qualification"
    ] is False
