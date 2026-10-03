import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_executor_v1_qualification.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_executor_v1.json"
)

EXECUTOR = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_executor_v1.py"
)

FREEZE = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_executor_v1_freeze.json"
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
        "QUALIFIED_PRE_OUTER_EXECUTION_EXECUTOR_DRY_RUN"
    )


def test_frozen_executor_preserved():
    d = load()

    assert d[
        "config"
    ][
        "sha256"
    ] == sha(
        CONFIG
    )

    assert d[
        "implementation"
    ][
        "sha256"
    ] == sha(
        EXECUTOR
    )

    assert d[
        "freeze"
    ][
        "sha256"
    ] == sha(
        FREEZE
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


def test_dry_run_coverage():
    d = load()

    c = d[
        "dry_run"
    ][
        "coverage"
    ]

    assert c[
        "shards_checked"
    ] == 793

    assert c[
        "subjects_checked"
    ] == 61

    assert c[
        "source_paths_checked"
    ] == 6309

    assert c[
        "threshold_rows_checked"
    ] == 45

    assert c[
        "fp32_artifacts_hashed"
    ] == 15

    assert c[
        "ptq_artifacts_hashed"
    ] == 15

    assert c[
        "historical_truth"
    ] == {
        "Activity":
            3390,

        "Falling":
            2919,
    }


def test_no_outer_execution_during_qualification():
    b = load()[
        "scientific_boundary"
    ]

    assert b[
        "model_weights_loaded"
    ] is False

    assert b[
        "model_forward_called"
    ] is False

    assert b[
        "fault_operator_called"
    ] is False

    assert b[
        "outer_performance_computed"
    ] is False

    assert b[
        "outer_model_outcomes_seen"
    ] is False

    assert b[
        "outer_fault_outcomes_seen"
    ] is False

    assert b[
        "OnField_used"
    ] is False


def test_executor_authorized_only_after_dry_run():
    d = load()

    assert d[
        "execution_authorization_state"
    ] == (
        "EXECUTOR_QUALIFIED_FOR_FROZEN_OUTER_SHARD_EXECUTION"
    )
