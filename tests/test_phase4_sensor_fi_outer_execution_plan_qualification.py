import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_plan_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_execution_v1.json"
)

PLANNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_execution_plan.py"
)

SHARDS = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_shards_v1.json"
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
        "QUALIFIED_PRE_OUTER_EXECUTION_PLAN"
    )


def test_frozen_files_preserved():
    d = load()

    assert d[
        "config"
    ][
        "sha256"
    ] == sha(
        CONFIG
    )

    assert d[
        "planner"
    ][
        "sha256"
    ] == sha(
        PLANNER
    )

    assert d[
        "shard_plan"
    ][
        "sha256"
    ] == sha(
        SHARDS
    )


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


def test_exact_workload():
    d = load()

    w = d[
        "frozen_workload"
    ]

    assert w[
        "outer_subject_count"
    ] == 61

    assert w[
        "outer_trial_count"
    ] == 6309

    assert w[
        "outer_window_count"
    ] == 273830

    assert w[
        "unique_fault_instances"
    ] == 42766632

    assert w[
        "model_window_evaluations"
    ] == 479750160

    assert w[
        "immutable_shards"
    ] == 793

    assert w[
        "subject_condition_rows_expected"
    ] == 320616


def test_full_matrix_is_preserved():
    d = load()

    m = d[
        "execution_matrix"
    ]

    assert m[
        "checkpoint_seeds"
    ] == 3

    assert m[
        "model_variants"
    ] == 2

    assert m[
        "operating_points"
    ] == 3

    assert m[
        "fault_families"
    ] == 12

    assert m[
        "severity_levels"
    ] == 3

    assert m[
        "all_predeclared_variants"
    ] is True

    assert m[
        "all_stochastic_replicates"
    ] is True


def test_no_outer_execution_happened():
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

    assert b[
        "validation_used_for_new_selection"
    ] is False

    assert b[
        "OnField_used"
    ] is False
