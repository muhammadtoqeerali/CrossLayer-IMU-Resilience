import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_devcal_runner_qualification_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_devcal_runner_v1.json"
)

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_devcal_runner.py"
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
        "QUALIFIED_TRAINING_CALIBRATION_EXECUTION_RUNNER"
    )


def test_pre_result_files_preserved():
    d = load()

    assert d[
        "pre_result_config"
    ][
        "preserved_unchanged"
    ] is True

    assert d[
        "pre_result_runner"
    ][
        "preserved_unchanged"
    ] is True

    assert d[
        "pre_result_config"
    ][
        "sha256"
    ] == sha(
        CONFIG
    )

    assert d[
        "pre_result_runner"
    ][
        "sha256"
    ] == sha(
        RUNNER
    )


def test_all_execution_counts():
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

    assert a[
        "clean_model_pair_count"
    ] == 15

    assert a[
        "window_fault_check_count"
    ] == 25

    assert a[
        "sequence_fault_check_count"
    ] == 35

    assert a[
        "fault_family_count"
    ] == 12


def test_qualification_used_training_calibration_only():
    d = load()
    b = d[
        "scientific_boundary"
    ]

    assert b[
        "model_predictions_computed"
    ] is True

    assert b[
        "sensor_fault_injection_executed"
    ] is True

    assert b[
        "training_calibration_used"
    ] is True

    assert b[
        "validation_used"
    ] is False

    assert b[
        "outer_test_used"
    ] is False

    assert b[
        "onfield_used"
    ] is False


def test_no_performance_or_threshold_tuning():
    d = load()
    b = d[
        "scientific_boundary"
    ]

    assert b[
        "thresholds_applied"
    ] is False

    assert b[
        "threshold_selection"
    ] is False

    assert b[
        "performance_metrics_computed"
    ] is False

    assert b[
        "probability_values_persisted"
    ] is False

    assert b[
        "robustness_comparison_made"
    ] is False

    assert b[
        "protocol_tuned_from_predictions"
    ] is False


def test_outer_execution_still_requires_manifest():
    d = load()

    assert len(
        d[
            "remaining_before_outer_test_execution"
        ]
    ) == 1

    assert (
        "outer-test execution manifest"
        in d[
            "remaining_before_outer_test_execution"
        ][0]
    )
