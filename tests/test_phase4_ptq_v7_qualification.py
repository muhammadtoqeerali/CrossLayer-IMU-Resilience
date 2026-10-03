import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4g_static_ptq_v7_all15_training_calibration_qualification_v1.json"
)

PROTOCOL = (
    ROOT
    / "configs/baseline/"
    "prospective_300ms_static_ptq_v7_fc1_fp32.json"
)

IMPLEMENTATION = (
    ROOT
    / "experiments/phase_04/"
    "ptq_v7_fc1_fp32.py"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def load():
    return json.loads(
        QUAL.read_text()
    )

def test_v7_all15_qualification_status():
    d = load()

    assert d["status"] == "QUALIFIED_TRAINING_CALIBRATION"
    assert d["candidate"] == "torch_fx_x86_fc1_fp32_v7"
    assert d["precision_classification"] == "MIXED_PRECISION_STATIC_PTQ"
    assert d["fully_int8_claim"] is False
    assert d["mcu_execution_claim"] is False

def test_v7_all15_acceptance_counts():
    d = load()
    a = d["acceptance"]

    assert a["all_15_pass"] is True
    assert a["overall_pass_count"] == 15
    assert a["overall_fail_count"] == 0
    assert a["structural_pass_count"] == 15
    assert a["numerical_pass_count"] == 15
    assert a["serialization_pass_count"] == 15

def test_v7_all15_exact_member_set():
    d = load()

    pairs = {
        (x["seed"], x["fold"])
        for x in d["members"]
    }

    assert pairs == {
        (seed, fold)
        for seed in [42, 123, 2025]
        for fold in [1, 2, 3, 4, 5]
    }

    assert len(d["members"]) == 15

def test_v7_all15_every_member_passed():
    d = load()

    for x in d["members"]:
        assert x["status"] == "PASS"
        assert x["structural_status"] == "PASS"
        assert x["numerical_status"] == "PASS"
        assert x["serialization_status"] == "PASS"

def test_v7_all15_frozen_p99_gate_passed():
    d = load()

    p99 = d[
        "aggregate_descriptive_metrics"
    ][
        "p99_absolute_probability_error"
    ]

    assert p99["limit"] == 0.05
    assert p99["max"] <= p99["limit"]

    for x in d["members"]:
        assert x["p99_absolute_probability_error"] <= 0.05

def test_v7_all15_protocol_and_impl_hashes():
    d = load()

    assert d["pre_result_protocol"]["sha256"] == sha(PROTOCOL)
    assert d["implementation"]["sha256"] == sha(IMPLEMENTATION)

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    assert protocol["status"] == "FROZEN_PRE_RESULT"

def test_v7_all15_training_calibration_only():
    d = load()
    s = d["qualification_scope"]

    assert s["checkpoint_count"] == 15
    assert s["calibration_windows_per_fold"] == 4096
    assert s["calibration_partition"] == "training only"
    assert s["validation_used"] is False
    assert s["outer_test_used"] is False
    assert s["onfield_used"] is False
    assert s["fault_data_used"] is False

def test_v7_all15_no_post_result_rule_change():
    d = load()
    s = d["scientific_boundary"]

    assert s["pre_result_protocol_rewritten"] is False
    assert s["candidate_changed_after_observing_results"] is False
    assert s["quantizer_changed"] is False
    assert s["numerical_gate_changed"] is False
    assert s["decision_threshold_changed"] is False

def test_v7_all15_claim_boundary():
    d = load()
    c = d["claim_boundary"]

    assert c["falling_class_p99_is_non_gating"] is True

    unsupported = set(
        c["not_supported"]
    )

    assert "fully INT8 execution" in unsupported
    assert "MCU execution" in unsupported
    assert "outer-test predictive performance" in unsupported
    assert "sensor-fault robustness" in unsupported
