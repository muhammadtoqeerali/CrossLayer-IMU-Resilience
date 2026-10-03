import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_evaluation_aggregation_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_aggregation_v1.json"
)

EXPECTED_CONFIG_SHA = (
    "ef95356b8e32d8f63bb41c967f35fcdfc24338e7d8ee59af40aa99c3424b0628"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load():
    return json.loads(QUAL.read_text())


def test_aggregation_qualification_status():
    d = load()

    assert d["status"] == "QUALIFIED_SYNTHETIC_HIERARCHY"
    assert d["evidence_tier"] == "P0"


def test_pre_result_aggregation_preserved():
    d = load()

    assert d["pre_result_config"]["preserved_unchanged"] is True
    assert d["pre_result_config"]["sha256"] == EXPECTED_CONFIG_SHA
    assert sha(CONFIG) == EXPECTED_CONFIG_SHA


def test_all_sanity_gates_pass():
    d = load()
    a = d["acceptance"]

    assert a["gate_count"] == a["passed_gate_count"]
    assert a["all_gates_pass"] is True


def test_no_real_result_evidence_used():
    d = load()
    b = d["scientific_boundary"]

    assert b["synthetic_or_metadata_only"] is True
    assert b["sensor_values_mutated"] is False
    assert b["fault_injection_executed"] is False
    assert b["model_loaded"] is False
    assert b["model_predictions_computed"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["physical_realism_claim"] is False


def test_only_reporting_policy_was_remaining():
    d = load()

    assert d["still_not_frozen"] == [
        "robustness acceptance criteria"
    ]
