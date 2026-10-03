import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/"
    "prospective_300ms_static_ptq_v6_postconv2_floattail.json"
)

MANIFEST = (
    ROOT
    / "manifests/"
    "phase_4g_static_ptq_v6_postconv2_floattail_protocol_freeze_v1.json"
)

IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/"
    "ptq_v6_postconv2_floattail.py"
)

V5_IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/"
    "ptq_v5_float_frontend.py"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def test_v6_frozen_before_real_result():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert d["status"] == "FROZEN_PRE_RESULT"
    assert b["real_v6_calibration_fidelity_observed_before_freeze"] is False
    assert b["real_v6_task_output_observed_before_freeze"] is False
    assert b["validation_used_for_v6_design"] is False
    assert b["outer_test_used_for_v6_design"] is False
    assert b["onfield_used_for_v6_design"] is False
    assert b["fault_data_used_for_v6_design"] is False

def test_v6_one_boundary_change():
    d = json.loads(CONFIG.read_text())
    c = d["v6_change_control"]

    assert c["change_count"] == 1
    assert "conv_2.2" in c["single_change_from_v5"]
    assert "fc.0" in c["single_change_from_v5"]

def test_v6_hashes_frozen():
    d = json.loads(CONFIG.read_text())
    m = json.loads(MANIFEST.read_text())

    assert sha(IMPLEMENTATION) == d["implementation"]["sha256"]
    assert sha(V5_IMPLEMENTATION) == (
        d["implementation"][
            "parent_v5_implementation"
        ]["sha256"]
    )

    assert m["implementation"]["sha256"] == sha(IMPLEMENTATION)

def test_v6_preserves_numerical_gates():
    d = json.loads(CONFIG.read_text())

    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v6_calibration_training_only():
    d = json.loads(CONFIG.read_text())
    c = d["calibration"]

    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v6_precision_contract():
    d = json.loads(CONFIG.read_text())
    m = d["method"]

    assert m["quantized_weighted_modules_required"] == [
        "conv_2.0",
        "fc.1",
    ]

    assert m["float_regions_required"] == [
        "front_end",
        "conv_2.2",
        "conv_2.3",
        "conv_2.4",
        "fc.0",
        "fc.2",
        "fc.4",
    ]

    assert m["second_quantization_boundary"] == (
        "after fc.0 / before fc.1"
    )

    assert m["fully_int8_claim_allowed"] is False

def test_v6_root_cause_is_post_prelu_requantization():
    d = json.loads(CONFIG.read_text())
    r = d["v6_change_control"]["reason"]

    assert r["root_cause_classification"] == (
        "POST_CONV2_PRELU_ACTIVATION_REQUANTIZATION_PUSHES_P99_OVER_GATE_"
        "MAXPOOL_KERNEL_IS_EXACT_ON_QUANTIZED_CODES"
    )

    assert r["post_prelu_activation_quantization_increment"] > 0
    assert r["maxpool_integer_kernel_increment"] == 0.0

def test_v6_corrected_synthetic_probe_passed():
    d = json.loads(CONFIG.read_text())
    e = d["v6_change_control"]["pre_result_capability_evidence"]

    assert e["initial_probe"]["recorded_status"] == "FAIL"
    assert e["corrected_probe"]["structural_pass"] is True
    assert e["corrected_probe"]["candidate_mechanism_changed"] is False
    assert e["corrected_probe"]["prelu2_to_pool_direct"] is True
    assert e["corrected_probe"]["pool_to_dropout_direct"] is True
    assert e["corrected_probe"]["dropout_to_flatten_direct"] is True
    assert e["corrected_probe"]["quantize_immediately_after_flatten"] is True
    assert e["corrected_probe"]["real_dataset_samples_used"] is False

def test_v6_preserves_v5_failure():
    d = json.loads(MANIFEST.read_text())
    p = d["preserved_v5_failure"]

    assert p["pilot_status"] == "FAIL"
    assert p["structural_status"] == "PASS"
    assert p["numerical_status"] == "FAIL"
    assert p["serialization_status"] == "PASS"
    assert p["p99_observed"] > p["p99_limit"]

def test_v6_no_gate_or_threshold_change():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert b["numerical_gate_changed"] is False
    assert b["decision_threshold_changed"] is False
