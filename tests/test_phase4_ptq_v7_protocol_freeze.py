import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/"
    "prospective_300ms_static_ptq_v7_fc1_fp32.json"
)

MANIFEST = (
    ROOT
    / "manifests/"
    "phase_4g_static_ptq_v7_fc1_fp32_protocol_freeze_v1.json"
)

IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/"
    "ptq_v7_fc1_fp32.py"
)

V6_IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/"
    "ptq_v6_postconv2_floattail.py"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def test_v7_frozen_before_real_result():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert d["status"] == "FROZEN_PRE_RESULT"
    assert b["real_v7_calibration_fidelity_observed_before_freeze"] is False
    assert b["real_v7_task_output_observed_before_freeze"] is False
    assert b["validation_used_for_v7_design"] is False
    assert b["outer_test_used_for_v7_design"] is False
    assert b["onfield_used_for_v7_design"] is False
    assert b["fault_data_used_for_v7_design"] is False

def test_v7_one_change():
    d = json.loads(CONFIG.read_text())
    c = d["v7_change_control"]

    assert c["change_count"] == 1
    assert "fc.1" in c["single_change_from_v6"]
    assert "fc.2" in c["single_change_from_v6"]

def test_v7_hashes_frozen():
    d = json.loads(CONFIG.read_text())
    m = json.loads(MANIFEST.read_text())

    assert sha(IMPLEMENTATION) == d["implementation"]["sha256"]
    assert sha(V6_IMPLEMENTATION) == (
        d["implementation"][
            "parent_v6_implementation"
        ]["sha256"]
    )

    assert m["implementation"]["sha256"] == sha(IMPLEMENTATION)

def test_v7_preserves_numerical_gates():
    d = json.loads(CONFIG.read_text())

    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v7_calibration_training_only():
    d = json.loads(CONFIG.read_text())
    c = d["calibration"]

    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v7_precision_contract():
    d = json.loads(CONFIG.read_text())
    m = d["method"]

    assert m["quantized_weighted_modules_required"] == [
        "conv_2.0",
    ]

    assert m["quantized_linear_modules_required"] == []

    assert m["float_regions_required"] == [
        "front_end",
        "conv_2.2",
        "conv_2.3",
        "conv_2.4",
        "fc.0",
        "fc.1",
        "fc.2",
        "fc.4",
    ]

    assert m["fc_quantization_boundary"] == (
        "after fc.2 / before fc.3"
    )

    assert m["fully_int8_claim_allowed"] is False

def test_v7_root_cause_is_fc1_input_quantization():
    d = json.loads(CONFIG.read_text())
    r = d["v7_change_control"]["reason"]

    assert r["root_cause_classification"] == (
        "FC1_INPUT_ACTIVATION_QUANTIZATION_PUSHES_P99_OVER_GATE"
    )

    assert r["before_fc1_quantization_p99"] < r["frozen_p99_limit"]
    assert r["fc1_input_quant_dequant_p99"] > r["frozen_p99_limit"]
    assert r["fc1_input_activation_quantization_increment"] > 0

def test_v7_synthetic_probe_passed():
    d = json.loads(CONFIG.read_text())
    e = d["v7_change_control"]["pre_result_capability_evidence"]

    assert e["structural_pass"] is True
    assert e["quantized_conv"] == ["conv_2.0"]
    assert e["quantized_linear"] == []
    assert e["flatten_to_fc1_direct"] is True
    assert e["fc1_to_fc2_direct"] is True
    assert e["quantize_immediately_after_fc2"] is True
    assert e["fc1_weight_exact"] is True
    assert e["fc1_bias_exact"] is True
    assert e["real_dataset_samples_used"] is False

def test_v7_preserves_v6_failure():
    d = json.loads(MANIFEST.read_text())
    p = d["preserved_v6_failure"]

    assert p["pilot_status"] == "FAIL"
    assert p["structural_status"] == "PASS"
    assert p["numerical_status"] == "FAIL"
    assert p["serialization_status"] == "PASS"
    assert p["p99_observed"] > p["p99_limit"]

def test_v7_no_gate_or_threshold_change():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert b["numerical_gate_changed"] is False
    assert b["decision_threshold_changed"] is False
