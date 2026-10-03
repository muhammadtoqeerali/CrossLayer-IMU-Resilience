import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/baseline/prospective_300ms_static_ptq_v4_float_classifier.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_4g_static_ptq_v4_float_classifier_protocol_freeze_v1.json"
)

IMPLEMENTATION = (
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

def test_v4_is_frozen_before_real_result():
    d=json.loads(CONFIG.read_text())

    assert d["status"] == "FROZEN_PRE_RESULT"

    b=d["scientific_boundary"]

    assert b["real_v4_calibration_fidelity_observed_before_freeze"] is False
    assert b["real_v4_task_output_observed_before_freeze"] is False
    assert b["validation_used_for_v4_design"] is False
    assert b["outer_test_used_for_v4_design"] is False
    assert b["onfield_used_for_v4_design"] is False
    assert b["fault_data_used_for_v4_design"] is False

def test_v4_has_exactly_one_change_from_v3():
    d=json.loads(CONFIG.read_text())
    c=d["v4_change_control"]

    assert c["change_count"] == 1
    assert "fc.4" in c["single_change_from_v3"]
    assert "FP32" in c["single_change_from_v3"]

def test_v4_implementation_and_parent_are_hash_frozen():
    d=json.loads(CONFIG.read_text())
    m=json.loads(MANIFEST.read_text())

    assert sha(IMPLEMENTATION) == d["implementation"]["sha256"]
    assert sha(V3_IMPLEMENTATION) == (
        d["implementation"][
            "parent_v3_implementation"
        ]["sha256"]
    )

    assert m["implementation"]["sha256"] == sha(IMPLEMENTATION)
    assert m["implementation"]["parent_v3_sha256"] == sha(V3_IMPLEMENTATION)

def test_v4_retains_unchanged_numerical_gates():
    d=json.loads(CONFIG.read_text())

    assert d["numerical_acceptance"]["required"] == {
        "all_outputs_finite": True,
        "output_shape_exactly_preserved": True,
        "argmax_agreement_fraction_min": 0.99,
        "mean_absolute_probability_error_max": 0.01,
        "p99_absolute_probability_error_max": 0.05,
    }

def test_v4_calibration_remains_training_only():
    d=json.loads(CONFIG.read_text())
    c=d["calibration"]

    assert c["windows_per_fold"] == 4096
    assert c["total_fold_window_instances"] == 20480
    assert c["source_partition"] == "training only"
    assert c["validation_allowed"] is False
    assert c["outer_test_allowed"] is False
    assert c["onfield_allowed"] is False
    assert c["fault_injected_allowed"] is False

def test_v4_precision_scope_is_explicit():
    d=json.loads(CONFIG.read_text())
    m=d["method"]

    assert m["quantized_modules_required"] == [
        "conv_1.0",
        "conv_2.0",
        "fc.1",
    ]

    assert m["float_modules_required"] == [
        "conv_1.2",
        "conv_2.2",
        "fc.2",
        "fc.4",
    ]

    assert m["final_classifier"] == "FP32 Linear"
    assert m["precision_classification"] == "MIXED_PRECISION_STATIC_PTQ"
    assert m["fully_int8_claim_allowed"] is False

def test_v4_design_is_supported_by_v3_lattice_evidence():
    d=json.loads(MANIFEST.read_text())
    e=d["v4_design_evidence"]

    assert e["worst25_lattice_match"] is True
    assert e["worst25_lattice_max_residual"] < 1e-6
    assert e["max_adjacent_probability_step"] > 0.05
    assert e["float_fc4_synthetic_structural_pass"] is True
    assert e["real_dataset_samples_used_by_structural_probe"] is False

def test_v4_preserves_v3_failure():
    d=json.loads(MANIFEST.read_text())
    p=d["preserved_v3_failure"]

    assert p["pilot_status"] == "FAIL"
    assert p["structural_status"] == "PASS"
    assert p["numerical_status"] == "FAIL"
    assert p["serialization_status"] == "PASS"
    assert p["failed_gate"] == "p99_absolute_probability_error_max"

def test_v4_does_not_relax_frozen_p99_gate():
    d=json.loads(CONFIG.read_text())

    c=d["v4_change_control"]

    assert c["frozen_p99_error_limit"] == 0.05
    assert c["v3_max_adjacent_probability_step"] > 0.05
    assert d["scientific_boundary"]["numerical_gate_changed"] is False
    assert d["scientific_boundary"]["decision_threshold_changed"] is False
