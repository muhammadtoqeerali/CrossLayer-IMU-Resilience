import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_operator_engine_qualification_v1.json"
)

ENGINE = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_operators.py"
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load():
    return json.loads(
        QUAL.read_text()
    )


def test_status():
    d = load()

    assert d["status"] == (
        "QUALIFIED_12_FAMILY_OPERATOR_ENGINE"
    )


def test_pre_result_engine_preserved():
    d = load()

    assert d[
        "pre_result_engine"
    ][
        "preserved_unchanged"
    ] is True

    assert d[
        "pre_result_engine"
    ][
        "sha256"
    ] == sha(ENGINE)


def test_all_families_and_gates_pass():
    d = load()
    a = d["acceptance"]

    assert a["family_count"] == 12
    assert a["all_gates_pass"] is True
    assert a["gate_count"] == a["passed_gate_count"]


def test_real_smoke_is_reference_independent_only():
    d = load()

    assert set(
        d[
            "acceptance"
        ][
            "real_training_calibration_smoke_families"
        ]
    ) == {
        "stuck_channel",
        "dropout",
        "frame_loss",
        "jitter",
        "delay",
        "orientation",
    }


def test_no_model_or_heldout_evidence():
    d = load()
    b = d["scientific_boundary"]

    assert b["project_dataset_files_mutated"] is False
    assert b["model_loaded"] is False
    assert b["model_predictions_computed"] is False
    assert b["threshold_selected"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["physical_realism_claim"] is False
    assert b["reference_scales_reestimated"] is False


def test_ready_for_dev_cal_runner():
    d = load()

    assert d[
        "remaining_items_before_development_calibration_engine_execution"
    ] == []
