import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/prospective_300ms_static_ptq_v3_prelu_wrapper.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_4g_static_ptq_v3_prelu_wrapper_protocol_freeze_v1.json"
)

IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/ptq_v3_prelu_wrapper.py"
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_v3_is_frozen_before_real_data_result():
    d=json.loads(CONFIG.read_text())

    assert d["status"] == "FROZEN_PRE_RESULT"

    b=d["scientific_boundary"]

    assert b["real_calibration_fidelity_observed_before_v3_freeze"] is False
    assert b["validation_used_for_v3_design"] is False
    assert b["outer_test_used_for_v3_design"] is False
    assert b["onfield_used_for_v3_design"] is False
    assert b["fault_data_used_for_v3_design"] is False
    assert b["task_metric_used_for_v3_design"] is False

def test_v3_has_single_mechanism_change_from_v2():
    d=json.loads(CONFIG.read_text())

    c=d["v3_change_control"]

    assert c["change_count"] == 1
    assert c["not_a_task_metric_selection"] is True
    assert "FloatPReLU" in c["single_change_from_v2"]

def test_v3_implementation_is_hash_frozen():
    d=json.loads(CONFIG.read_text())
    m=json.loads(MANIFEST.read_text())

    expected=d["implementation"]["sha256"]

    assert sha(IMPLEMENTATION) == expected
    assert m["implementation"]["sha256"] == expected

def test_v3_keeps_prelus_float_and_weighted_layers_quantized():
    d=json.loads(CONFIG.read_text())

    assert d["method"]["quantized_weighted_operators_required"] == [
        "Conv1d",
        "Linear",
    ]

    assert d["method"]["prelu_execution"] == "FP32 FloatPReLU wrapper"

    assert d["method"]["precision_classification"] == (
        "MIXED_PRECISION_STATIC_PTQ"
    )

    assert d["method"]["fully_int8_claim_allowed"] is False

def test_v3_retains_frozen_numerical_gates():
    d=json.loads(CONFIG.read_text())

    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v3_calibration_is_still_training_only():
    d=json.loads(CONFIG.read_text())
    c=d["calibration"]

    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v3_requires_exact_wrapper_structure():
    d=json.loads(CONFIG.read_text())

    required=d["structural_acceptance"]["required"]

    assert "exactly two QuantizedConv1d modules" in required
    assert "exactly two QuantizedLinear modules" in required
    assert "exactly three FloatPReLU wrappers" in required
    assert "zero QuantizedPReLU modules" in required
    assert "all three FloatPReLU weights exactly equal FP32 checkpoint slopes" in required

def test_v3_capability_basis_used_no_real_samples():
    d=json.loads(MANIFEST.read_text())

    e=d["synthetic_capability_evidence"]

    assert e["float_wrapper_structural_ok"] is True
    assert e["real_dataset_samples_used"] is False
    assert e["verdict"] == (
        "FLOAT_PRELU_WRAPPER_WITH_EXPLICIT_QDQ_IS_STRUCTURALLY_VIABLE"
    )
