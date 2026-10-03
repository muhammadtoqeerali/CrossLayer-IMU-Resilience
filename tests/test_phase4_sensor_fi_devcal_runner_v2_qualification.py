import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_devcal_runner_v2_phase3_falling_route_qualification_v1.json"
)

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_devcal_runner_v2.py"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_devcal_runner_v2_phase3_falling_route.json"
)

V1 = (
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
        "QUALIFIED_PHASE3_FALLING_ROUTE_TRAINING_CALIBRATION"
    )


def test_v2_pre_result_preserved():
    d = load()

    assert d[
        "pre_result_v2"
    ][
        "preserved_unchanged"
    ] is True

    assert d[
        "pre_result_v2"
    ][
        "runner_sha256"
    ] == sha(
        RUNNER
    )

    assert d[
        "pre_result_v2"
    ][
        "config_sha256"
    ] == sha(
        CONFIG
    )


def test_v1_preserved():
    d = load()

    assert d[
        "v1_preserved"
    ][
        "rewritten"
    ] is False

    assert d[
        "v1_preserved"
    ][
        "runner_sha256"
    ] == sha(
        V1
    )


def test_all_falling_parents_exact():
    d = load()
    r = d[
        "route_acceptance"
    ]

    assert r[
        "unique_training_calibration_falling_parents"
    ] == 2269

    assert r[
        "exact_phase3_frame_route_count"
    ] == 2269

    assert r[
        "dataset_parent_counts"
    ] == {
        "KFALL":
            1813,

        "UNIVR":
            456,
    }


def test_position_diagnostic_matches_univr_only():
    d = load()

    assert d[
        "route_acceptance"
    ][
        "position_route_exact_count_diagnostic_only"
    ] == 456


def test_all_seven_sequence_families_all_folds():
    d = load()
    s = d[
        "fault_model_smoke_acceptance"
    ]

    assert s[
        "check_count"
    ] == 35

    assert s[
        "fold_count"
    ] == 5

    assert s[
        "family_count"
    ] == 7

    assert s[
        "same_corrupted_tensor_fp32_ptq"
    ] is True

    assert s[
        "performance_acceptance"
    ] is False


def test_no_heldout_or_tuning():
    d = load()
    b = d[
        "scientific_boundary"
    ]

    assert b[
        "validation_used"
    ] is False

    assert b[
        "outer_test_used"
    ] is False

    assert b[
        "onfield_used"
    ] is False

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
        "probabilities_persisted"
    ] is False

    assert b[
        "robustness_comparison_made"
    ] is False

    assert b[
        "protocol_tuned_from_model_outputs"
    ] is False


def test_outer_manifest_still_required():
    d = load()

    assert len(
        d[
            "remaining_before_outer_test_execution"
        ]
    ) == 1
