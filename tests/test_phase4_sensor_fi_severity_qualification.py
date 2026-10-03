import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_severity_protocol_v1_sanity_qualification.json"
)

PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA = (
    "48b8927628f2580642c93ed9b492cf562a4ad40a21c7815bc3d5a93ca970b009"
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


def test_qualification_status():
    d = json.loads(QUAL.read_text())

    assert d["status"] == "QUALIFIED_INPUT_SPACE_SANITY"
    assert d["evidence_tier"] == "P0"
    assert d["physical_realism_claim"] is False


def test_pre_result_protocol_preserved():
    d = json.loads(QUAL.read_text())

    assert d["pre_result_protocol"]["preserved_unchanged"] is True
    assert d["pre_result_protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA
    assert sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA


def test_all_five_folds_pass():
    d = json.loads(QUAL.read_text())
    a = d["acceptance"]

    assert a["fold_count"] == 5
    assert a["pass_count"] == 5
    assert a["fail_count"] == 0
    assert a["all_folds_pass"] is True

    assert len(a["folds"]) == 5

    for fold in a["folds"]:
        assert fold["status"] == "PASS"
        assert fold["gate_count"] == fold["passed_gate_count"]
        assert (
            fold["largest_predeclared_temporal_requirement_samples"]
            < fold["shortest_parent_trial_samples"]
        )


def test_no_model_or_heldout_evidence_used():
    d = json.loads(QUAL.read_text())
    b = d["scientific_boundary"]

    assert b["model_predictions_computed"] is False
    assert b["model_checkpoint_loaded"] is False
    assert b["fault_injection_executed"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["physical_realism_claim"] is False
    assert b["cross_fold_pooled_reference_scale"] is False
    assert b["pre_result_protocol_rewritten"] is False


def test_unqualified_claims_remain_explicit():
    d = json.loads(QUAL.read_text())

    assert set(d["not_qualified_claims"]) == {
        "model robustness under sensor faults",
        "outer-test robustness",
        "OnField robustness",
        "physical sensor-fault realism",
        "hardware sensor-fault behavior",
        "HIL behavior",
        "MCU sensor-fault behavior",
        "P2 realism",
        "P3 realism",
    }


def test_remaining_freezes_are_explicit():
    d = json.loads(QUAL.read_text())

    assert set(d["still_not_frozen"]) == {
        "fault-instance onset sampling policy",
        "number of stochastic fault instances per parent sequence",
        "fault-instance seed namespace",
        "evaluation aggregation protocol",
        "robustness acceptance criteria",
    }
