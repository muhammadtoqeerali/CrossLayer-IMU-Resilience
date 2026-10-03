import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLOSE = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_pre_operator_governance_complete_v1.json"
)


def load():
    return json.loads(
        CLOSE.read_text()
    )


def test_status():
    d = load()

    assert d["status"] == (
        "PRE_OPERATOR_GOVERNANCE_COMPLETE"
    )


def test_no_items_remain():
    d = load()

    assert d[
        "remaining_items_before_fault_operator_implementation"
    ] == []


def test_all_required_components_present():
    d = load()

    assert set(
        d["authoritative_components"]
    ) == {
        "input_contract",
        "severity_protocol",
        "sampling_v3",
        "aggregation_v1",
        "reporting_v2",
        "jitter_semantics_v1",
    }


def test_no_heldout_or_model_outcomes_selected_governance():
    d = load()
    b = d["scientific_boundary"]

    assert b["model_predictions_used_to_select_governance"] is False
    assert b["validation_used_to_select_governance"] is False
    assert b["outer_test_used_to_select_governance"] is False
    assert b["onfield_used_to_select_governance"] is False
    assert b["physical_realism_claim"] is False
    assert b["arbitrary_practical_margin_introduced"] is False
