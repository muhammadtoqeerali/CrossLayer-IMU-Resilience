import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(p):
    return json.loads((ROOT / p).read_text())

def test_phase4g_protocol_is_pre_result_frozen():
    d = load("configs/baseline/prospective_300ms_static_ptq_v1.json")
    assert d["status"] == "FROZEN_PRE_RESULT"
    assert d["method"]["primary_method"] == "static_post_training_quantization"
    assert d["method"]["framework"] == "PyTorch FX"
    assert d["method"]["runtime_qconfig"]["backend"] == "x86"
    assert d["selection_policy"]["candidate_count"] == 1
    assert d["selection_policy"]["candidate_sweep"] is False

def test_phase4g_calibration_is_training_only():
    d = load("configs/baseline/prospective_300ms_static_ptq_v1.json")
    c = d["calibration"]
    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False
    assert len(c["replay_audit"]) == 5

def test_phase4g_quantizer_selection_cannot_use_heldout_or_fault_data():
    d = load("configs/baseline/prospective_300ms_static_ptq_v1.json")
    s = d["selection_policy"]
    assert s["selection_by_validation_accuracy"] is False
    assert s["selection_by_outer_test_accuracy"] is False
    assert s["selection_by_onfield"] is False
    assert s["selection_by_sensor_fault_robustness"] is False
    assert s["selection_by_compute_fault_robustness"] is False
    assert s["selection_by_combined_fault_robustness"] is False

def test_phase4g_numerical_gates_are_frozen():
    d = load("configs/baseline/prospective_300ms_static_ptq_v1.json")
    r = d["numerical_acceptance"]["required"]
    assert r["all_outputs_finite"] is True
    assert r["output_shape_exactly_preserved"] is True
    assert r["argmax_agreement_fraction_min"] == 0.99
    assert r["mean_absolute_probability_error_max"] == 0.01
    assert r["p99_absolute_probability_error_max"] == 0.05

def test_phase4g_precision_claim_requires_operator_audit():
    d = load("configs/baseline/prospective_300ms_static_ptq_v1.json")
    s = d["structural_acceptance"]
    assert "Conv1d" in s["required_weighted_operator_quantization"]
    assert "Linear" in s["required_weighted_operator_quantization"]
    assert len(s["full_int8_claim_allowed_only_if"]) >= 4
    assert "mixed-precision" in s["mixed_precision_acceptance"]

def test_phase4g_manifest_records_no_pre_freeze_quantized_results():
    d = load("manifests/phase_4g_static_ptq_protocol_freeze_v1.json")
    assert d["status"] == "FROZEN_PRE_RESULT"
    p = d["pre_result_freeze"]
    assert p["quantized_model_created"] is False
    assert p["quantized_outputs_observed"] is False
    assert p["validation_used"] is False
    assert p["outer_test_used"] is False
    assert p["onfield_used"] is False
    assert p["fault_data_used"] is False
