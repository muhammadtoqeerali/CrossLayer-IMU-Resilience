import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/prospective_300ms_static_ptq_v5_float_frontend.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_4g_static_ptq_v5_float_frontend_protocol_freeze_v1.json"
)

IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/ptq_v5_float_frontend.py"
)

V4_IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/ptq_v4_float_classifier.py"
)

V3_IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/ptq_v3_prelu_wrapper.py"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def test_v5_frozen_before_real_result():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert d["status"] == "FROZEN_PRE_RESULT"
    assert b["real_v5_calibration_fidelity_observed_before_freeze"] is False
    assert b["real_v5_task_output_observed_before_freeze"] is False
    assert b["validation_used_for_v5_design"] is False
    assert b["outer_test_used_for_v5_design"] is False
    assert b["onfield_used_for_v5_design"] is False
    assert b["fault_data_used_for_v5_design"] is False

def test_v5_has_one_versioned_boundary_change():
    d = json.loads(CONFIG.read_text())
    c = d["v5_change_control"]

    assert c["change_count"] == 1
    assert "front_end" in c["single_change_from_v4"]
    assert "normalizer" in c["single_change_from_v4"]
    assert "conv_1" in c["single_change_from_v4"]

def test_v5_implementation_hashes_frozen():
    d = json.loads(CONFIG.read_text())
    m = json.loads(MANIFEST.read_text())

    assert sha(IMPLEMENTATION) == d["implementation"]["sha256"]
    assert sha(V4_IMPLEMENTATION) == (
        d["implementation"]["parent_v4_implementation"]["sha256"]
    )
    assert sha(V3_IMPLEMENTATION) == (
        d["implementation"]["parent_v3_implementation"]["sha256"]
    )

    assert m["implementation"]["sha256"] == sha(IMPLEMENTATION)

def test_v5_preserves_numerical_gates():
    d = json.loads(CONFIG.read_text())

    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v5_calibration_still_training_only():
    d = json.loads(CONFIG.read_text())
    c = d["calibration"]

    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v5_precision_contract():
    d = json.loads(CONFIG.read_text())
    m = d["method"]

    assert m["quantized_modules_required"] == [
        "conv_2.0",
        "fc.1",
    ]

    assert m["float_regions_required"] == [
        "front_end",
        "conv_2.2",
        "fc.2",
        "fc.4",
    ]

    assert m["fully_int8_claim_allowed"] is False

def test_v5_design_evidence_removed_upstream_quantization():
    d = json.loads(MANIFEST.read_text())
    e = d["v5_design_evidence"]

    assert e["simple_conv1_exclusion_structural_pass"] is False
    assert e["opaque_conv1_only_structural_pass"] is False
    assert e["opaque_float_frontend_structural_pass"] is True
    assert e["front_input_exact"] is True
    assert e["front_output_exact"] is True
    assert e["upstream_quantize_nodes"] == []
    assert e["upstream_dequantize_nodes"] == []
    assert e["real_dataset_samples_used_by_capability_probe"] is False

def test_v5_preserves_v4_failure():
    d = json.loads(MANIFEST.read_text())
    p = d["preserved_v4_failure"]

    assert p["pilot_status"] == "FAIL"
    assert p["structural_status"] == "PASS"
    assert p["numerical_status"] == "FAIL"
    assert p["serialization_status"] == "PASS"
    assert p["p99_observed"] > p["p99_limit"]

def test_v5_does_not_relax_gate_or_threshold():
    d = json.loads(CONFIG.read_text())
    b = d["scientific_boundary"]

    assert b["numerical_gate_changed"] is False
    assert b["decision_threshold_changed"] is False
