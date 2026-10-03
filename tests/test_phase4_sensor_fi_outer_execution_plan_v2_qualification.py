import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_plan_v2_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_execution_v2_historical_truth.json"
)

SHARDS = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json"
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


def test_status():
    d = load()

    assert d[
        "status"
    ] == (
        "QUALIFIED_PRE_OUTER_EXECUTION_PLAN_V2_HISTORICAL_TRUTH"
    )


def test_frozen_files_match():
    d = load()

    assert d[
        "config"
    ][
        "sha256"
    ] == sha(
        CONFIG
    )

    assert d[
        "shard_plan"
    ][
        "sha256"
    ] == sha(
        SHARDS
    )


def test_truth_contract():
    d = load()

    t = d[
        "truth_contract"
    ]

    assert t[
        "stored_label_trial_inventory"
    ] == {
        "Activity":
            3391,

        "Falling":
            2918,
    }

    assert t[
        "historical_evaluator_trial_truth"
    ] == {
        "Activity":
            3390,

        "Falling":
            2919,
    }

    assert t[
        "mismatch_count"
    ] == 1

    assert t[
        "historical_exception"
    ] == "KFALL_106_T27_R05"

    assert t[
        "historical_exception_valid_trigger_possible"
    ] is False

    assert t[
        "historical_exception_classification_consequence"
    ] == "FN"

    assert t[
        "historical_exception_event_consequence"
    ] == "missed"

    assert t[
        "historical_exception_sensor_lead_ms"
    ] is None

    assert t[
        "relabeling"
    ] is False


def test_execution_matrix_unchanged():
    d = load()

    m = d[
        "unchanged_execution_matrix"
    ]

    assert m[
        "subjects"
    ] == 61

    assert m[
        "trials"
    ] == 6309

    assert m[
        "windows"
    ] == 273830

    assert m[
        "shards"
    ] == 793

    assert m[
        "unique_fault_instances"
    ] == 42766632

    assert m[
        "model_window_evaluations"
    ] == 479750160

    assert m[
        "subject_condition_rows"
    ] == 320616


def test_all_gates_pass():
    d = load()

    a = d[
        "acceptance"
    ]

    assert a[
        "all_gates_pass"
    ] is True

    assert a[
        "gate_count"
    ] == a[
        "passed_gate_count"
    ]


def test_no_outer_execution():
    d = load()

    b = d[
        "scientific_boundary"
    ]

    assert b[
        "model_loaded"
    ] is False

    assert b[
        "model_predictions_computed"
    ] is False

    assert b[
        "fault_injection_executed"
    ] is False

    assert b[
        "outer_model_outcomes_seen"
    ] is False

    assert b[
        "outer_fault_outcomes_seen"
    ] is False
