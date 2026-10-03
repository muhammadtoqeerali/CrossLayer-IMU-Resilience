import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/prospective_300ms_static_ptq_v2_prelu_fp32.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_4g_static_ptq_v2_prelu_fp32_protocol_freeze_v1.json"
)

def test_v2_is_frozen_before_result():
    d=json.loads(CONFIG.read_text())
    assert d["status"] == "FROZEN_PRE_RESULT"
    b=d["scientific_boundary"]
    assert b["v2_quantized_model_created_before_freeze"] is False
    assert b["v2_quantized_outputs_observed_before_freeze"] is False

def test_v2_has_exactly_one_targeted_change():
    d=json.loads(CONFIG.read_text())
    c=d["v2_change_control"]
    assert c["change_count"] == 1
    assert c["single_targeted_change"] == "exclude torch.nn.PReLU from quantization"
    assert d["method"]["object_type_overrides"]["torch.nn.PReLU"] is None
    assert d["method"]["prelu_precision"] == "FP32"

def test_v2_retains_v1_numerical_gates():
    d=json.loads(CONFIG.read_text())
    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v2_remains_training_only():
    d=json.loads(CONFIG.read_text())
    c=d["calibration"]
    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v2_is_explicitly_mixed_precision():
    d=json.loads(CONFIG.read_text())
    s=d["structural_acceptance"]
    assert s["precision_classification"] == "MIXED_PRECISION_STATIC_PTQ"
    assert s["fully_int8_claim_allowed"] is False
    assert d["scientific_boundary"]["fully_int8_claim"] is False

def test_v2_change_is_supported_by_preserved_v1_operator_failure():
    d=json.loads(MANIFEST.read_text())
    v=d["v1_failure_preserved"]
    assert v["result"] == "FAIL"
    assert v["operator_microtest_classification"] == (
        "QUANTIZED_PRELU_POSITIVE_BRANCH_OPERATOR_CONTRACT_FAILURE"
    )
    assert v["positive_failures"] == 25
    assert v["positive_cases"] == 25
    assert v["negative_failures"] == 0
    assert v["negative_cases"] == 22

def test_v2_does_not_use_heldout_or_fault_data_for_design():
    d=json.loads(CONFIG.read_text())
    b=d["scientific_boundary"]
    assert b["validation_used_for_v2_design"] is False
    assert b["outer_test_used_for_v2_design"] is False
    assert b["onfield_used_for_v2_design"] is False
    assert b["fault_data_used_for_v2_design"] is False
