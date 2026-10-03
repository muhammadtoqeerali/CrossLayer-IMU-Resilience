import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_jitter_semantics_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_jitter_semantics_v1.json"
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
        "QUALIFIED_EXECUTABLE_JITTER_SEMANTICS"
    )


def test_pre_result_semantics_preserved():
    d = load()

    assert d[
        "pre_result_config"
    ][
        "preserved_unchanged"
    ] is True

    assert d[
        "pre_result_config"
    ][
        "sha256"
    ] == sha(CONFIG)


def test_all_levels_qualified():
    d = load()

    assert d["acceptance"]["all_gates_pass"] is True

    assert d["acceptance"]["levels_qualified"] == [
        "L1",
        "L2",
        "L3",
    ]


def test_no_real_dataset_or_model_execution():
    d = load()
    b = d["scientific_boundary"]

    assert b["synthetic_operator_exercised"] is True
    assert b["project_dataset_files_mutated"] is False
    assert b["real_dataset_fault_execution"] is False
    assert b["model_loaded"] is False
    assert b["model_predictions_computed"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["physical_realism_claim"] is False


def test_no_items_remain_before_operator_engine():
    d = load()

    assert d[
        "remaining_items_before_fault_operator_implementation"
    ] == []
