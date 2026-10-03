import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_instance_sampling_v2_qualification_v1.json"
)

V1_PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v1.json"
)

V2_PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v2_fault_id.json"
)

EXPECTED_V1_SHA = (
    "8174885d78a8864ce1f1c9f5fb1faaa1b3e1a7d1f34538c8e4120c87ecd452f0"
)

EXPECTED_V2_SHA = (
    "6004868ec664981f5feb8fc03d70f7263b41e94be597805fa7dba5f9831a3cdd"
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load():
    return json.loads(
        QUAL.read_text()
    )


def test_status_and_scope():
    d = load()

    assert d["status"] == "QUALIFIED_METADATA_REPLAY"
    assert d["evidence_tier"] == "P0"
    assert d["physical_realism_claim"] is False


def test_v1_failure_preserved():
    d = load()

    assert d["parent_v1"]["preserved_unchanged"] is True
    assert d["parent_v1"]["failure_classification"] == (
        "FAULT_ID_NAMESPACE_COLLISION_ONLY"
    )

    assert d["parent_v1"]["failed_gates"] == {
        "2": ["no_cross_fold_fault_id_collision"],
        "3": ["no_cross_fold_fault_id_collision"],
        "4": ["no_cross_fold_fault_id_collision"],
    }

    assert sha(V1_PROTOCOL) == EXPECTED_V1_SHA


def test_v2_pre_result_protocol_preserved():
    d = load()

    assert d["v2_pre_result_protocol"]["preserved_unchanged"] is True
    assert d["v2_pre_result_protocol"]["sha256"] == EXPECTED_V2_SHA
    assert sha(V2_PROTOCOL) == EXPECTED_V2_SHA


def test_all_five_folds_pass():
    d = load()
    a = d["acceptance"]

    assert a["fold_count"] == 5
    assert a["pass_count"] == 5
    assert a["fail_count"] == 0
    assert a["all_folds_pass"] is True

    for fold in a["folds"]:
        assert fold["status"] == "PASS"
        assert fold["gate_count"] == fold["passed_gate_count"]
        assert fold["window_instance_count"] == 153
        assert fold["sequence_instance_count"] == 138
        assert fold["total_instance_count"] == 291


def test_single_change_remains_fault_id_only():
    d = load()
    c = d["single_change_from_v1"]

    assert c["field"] == "fault_id"

    for key in [
        "severity_changed",
        "target_policy_changed",
        "onset_policy_changed",
        "duration_policy_changed",
        "replicate_policy_changed",
        "seed_namespace_changed",
        "seed_derivation_changed",
        "partition_policy_changed",
        "replay_canonicalization_changed",
    ]:
        assert c[key] is False


def test_no_model_or_heldout_evidence():
    d = load()
    b = d["scientific_boundary"]

    assert b["metadata_only"] is True
    assert b["sensor_values_mutated"] is False
    assert b["fault_injection_executed"] is False
    assert b["model_loaded"] is False
    assert b["model_predictions_computed"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["physical_realism_claim"] is False
    assert b["v1_rewritten"] is False
    assert b["v2_pre_result_protocol_rewritten"] is False
    assert b["severity_changed"] is False
    assert b["sampling_changed_except_fault_id_namespace"] is False
    assert b["seed_derivation_changed"] is False


def test_only_two_preexecution_freezes_remain():
    d = load()

    assert set(d["still_not_frozen"]) == {
        "evaluation aggregation protocol",
        "robustness acceptance criteria",
    }
