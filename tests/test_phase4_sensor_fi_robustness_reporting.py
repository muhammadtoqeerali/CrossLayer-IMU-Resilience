import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_robustness_reporting_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_reporting.py"
)


def load_config():
    return json.loads(CONFIG.read_text())


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_reporting",
        IMPL,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_no_arbitrary_global_margin():
    d = load_config()
    p = d["performance_margin_policy"]

    assert p["application_level_practical_margin_available"] is False
    assert p["P2_P3_validated_margin_available"] is False
    assert p["arbitrary_numeric_robustness_margin_allowed"] is False
    assert p["binary_global_robust_not_robust_label_allowed"] is False


def test_exact_required_fault_matrix():
    d = load_config()
    m = d["required_fault_matrix"]

    assert len(m["fault_families"]) == 12
    assert m["severity_levels"] == ["L1", "L2", "L3"]
    assert m["family_severity_macro_count_per_model_variant"] == 36
    assert m["all_three_checkpoint_seeds_required"] is True
    assert m["all_five_outer_folds_required_for_final_evaluation"] is True
    assert m["all_61_outer_test_subjects_required_for_final_evaluation"] is True


def test_required_current_model_variants():
    d = load_config()
    models = d["required_model_variants"]

    assert set(models) == {
        "prospective_fp32_300ms",
        "qualified_static_ptq_v7",
    }

    assert models["prospective_fp32_300ms"]["checkpoint_count"] == 15
    assert models["qualified_static_ptq_v7"]["checkpoint_count"] == 15
    assert models["qualified_static_ptq_v7"]["fully_int8_claim"] is False
    assert models["qualified_static_ptq_v7"]["MCU_execution_claim"] is False


@pytest.mark.parametrize(
    "low,high,expected",
    [
        (0.01, 0.20, "DEGRADATION_SUPPORTED"),
        (-0.20, -0.01, "IMPROVEMENT_SUPPORTED"),
        (-0.10, 0.10, "NO_DIRECTIONAL_CONCLUSION"),
        (0.0, 0.10, "NO_DIRECTIONAL_CONCLUSION"),
        (-0.10, 0.0, "NO_DIRECTIONAL_CONCLUSION"),
    ],
)
def test_direction_classification(low, high, expected):
    m = load_impl()

    assert m.classify_direction(
        point=(low + high) / 2.0,
        ci95_low=low,
        ci95_high=high,
        coverage_complete=True,
    ) == expected


def test_incomplete_coverage_is_unresolved():
    m = load_impl()

    assert m.classify_direction(
        point=0.2,
        ci95_low=0.1,
        ci95_high=0.3,
        coverage_complete=False,
    ) == "UNRESOLVED"


def test_global_robustness_label_is_prohibited():
    m = load_impl()

    with pytest.raises(RuntimeError):
        m.global_robustness_label()


def test_interpretability_gate():
    m = load_impl()

    required = {
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

    assert m.execution_interpretability(
        required
    ) == "EXECUTION_INTERPRETABLE"

    required[
        "all_required_result_dimensions_present"
    ] = False

    assert m.execution_interpretability(
        required
    ) == "EXECUTION_NOT_INTERPRETABLE"


def test_no_directional_conclusion_not_equivalence():
    d = load_config()

    assert "must never be rewritten" in d[
        "important_interpretation_rule"
    ]


def test_heldout_cannot_retune_protocol():
    d = load_config()
    h = d["heldout_governance"]

    assert h["outer_test_results_may_change_severity"] is False
    assert h["outer_test_results_may_change_sampling"] is False
    assert h["outer_test_results_may_change_aggregation"] is False
    assert h["outer_test_results_may_change_reporting_rules"] is False
    assert h["outer_test_results_may_select_fault_families"] is False
    assert h["outer_test_results_may_select_only_favorable_metrics"] is False


def test_onfield_claim_boundaries():
    d = load_config()
    o = d["OnField_policy"]

    assert o["faulted_evaluation_allowed"] is False
    assert o["fall_recall_claim_allowed"] is False
    assert o["fall_timing_claim_allowed"] is False
    assert o["recovery_claim_allowed"] is False


def test_preexecution_completion_flag():
    d = load_config()

    assert d["preexecution_governance_complete_after_qualification"] is True
