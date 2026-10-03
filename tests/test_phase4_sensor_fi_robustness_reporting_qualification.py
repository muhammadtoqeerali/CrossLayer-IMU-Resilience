import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_robustness_reporting_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_robustness_reporting_v1.json"
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load():
    return json.loads(QUAL.read_text())


def test_reporting_qualification_status():
    d = load()

    assert d["status"] == "QUALIFIED_PREEXECUTION_REPORTING_POLICY"
    assert d["preexecution_governance_status"] == "COMPLETE"


def test_pre_result_reporting_policy_preserved():
    d = load()

    assert d["pre_result_config"]["preserved_unchanged"] is True
    assert d["pre_result_config"]["sha256"] == sha(CONFIG)


def test_all_reporting_gates_pass():
    d = load()
    a = d["acceptance"]

    assert a["gate_count"] == a["passed_gate_count"]
    assert a["all_gates_pass"] is True


def test_no_real_results_used():
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
    assert b["arbitrary_practical_margin_introduced"] is False


def test_no_governance_items_remain_before_operator_implementation():
    d = load()

    assert d[
        "remaining_governance_items_before_fault_operator_implementation"
    ] == []
