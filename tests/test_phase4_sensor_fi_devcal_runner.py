import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

RUNNER = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_devcal_runner.py"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_devcal_runner_v1.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_devcal_runner",
        RUNNER,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_config():
    return json.loads(
        CONFIG.read_text()
    )


def test_qualification_partition_only():
    m = load_runner()

    m.assert_partition(
        "training_calibration",
        execution_stage="qualification",
    )

    for partition in [
        "validation",
        "outer_test",
        "onfield",
    ]:
        with pytest.raises(
            ValueError
        ):
            m.assert_partition(
                partition,
                execution_stage="qualification",
            )


def test_final_outer_stage_rejects_non_outer():
    m = load_runner()

    m.assert_partition(
        "outer_test",
        execution_stage="final_outer_evaluation",
    )

    for partition in [
        "training_calibration",
        "validation",
        "onfield",
    ]:
        with pytest.raises(
            ValueError
        ):
            m.assert_partition(
                partition,
                execution_stage="final_outer_evaluation",
            )


def test_label_normalization():
    m = load_runner()

    assert m.normalise_binary_label(
        "Activity"
    ) == 0

    assert m.normalise_binary_label(
        "Falling"
    ) == 1

    assert m.normalise_binary_label(
        b"Falling"
    ) == 1


def test_activity_only_rewindow_geometry(monkeypatch):
    m = load_runner()

    monkeypatch.setattr(
        m,
        "low_pass_filter",
        lambda x, cutoff, rate: np.asarray(
            x
        ).copy(),
    )

    x = np.arange(
        100 * 9,
        dtype=float,
    ).reshape(
        100,
        9,
    )

    labels = np.zeros(
        5,
        dtype=np.int8,
    )

    windows = m.rewindow_sequence(
        x,
        labels,
        fall_start_position=None,
    )

    assert windows.shape == (
        5,
        30,
        9,
    )

    np.testing.assert_array_equal(
        windows[
            0
        ],
        x[
            0:30
        ],
    )

    np.testing.assert_array_equal(
        windows[
            1
        ],
        x[
            15:45
        ],
    )


def test_fall_rewindow_resets_at_fall_start(monkeypatch):
    m = load_runner()

    monkeypatch.setattr(
        m,
        "low_pass_filter",
        lambda x, cutoff, rate: np.asarray(
            x
        ).copy(),
    )

    x = np.arange(
        200 * 9,
        dtype=float,
    ).reshape(
        200,
        9,
    )

    labels = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
        ],
        dtype=np.int8,
    )

    windows = m.rewindow_sequence(
        x,
        labels,
        fall_start_position=100,
    )

    np.testing.assert_array_equal(
        windows[
            0
        ],
        x[
            0:30
        ],
    )

    np.testing.assert_array_equal(
        windows[
            2
        ],
        x[
            30:60
        ],
    )

    np.testing.assert_array_equal(
        windows[
            3
        ],
        x[
            100:130
        ],
    )

    np.testing.assert_array_equal(
        windows[
            4
        ],
        x[
            115:145
        ],
    )


def test_fall_requires_fall_start(monkeypatch):
    m = load_runner()

    monkeypatch.setattr(
        m,
        "low_pass_filter",
        lambda x, cutoff, rate: x,
    )

    with pytest.raises(
        ValueError
    ):
        m.rewindow_sequence(
            np.zeros(
                (
                    100,
                    9,
                )
            ),
            np.asarray(
                [
                    0,
                    1,
                ]
            ),
            fall_start_position=None,
        )


def test_interleaved_labels_rejected(monkeypatch):
    m = load_runner()

    monkeypatch.setattr(
        m,
        "low_pass_filter",
        lambda x, cutoff, rate: x,
    )

    with pytest.raises(
        ValueError
    ):
        m.rewindow_sequence(
            np.zeros(
                (
                    100,
                    9,
                )
            ),
            np.asarray(
                [
                    0,
                    1,
                    0,
                ]
            ),
            fall_start_position=40,
        )


def test_model_pair_policy_is_same_tensor():
    d = load_config()

    p = d[
        "model_pairing"
    ]

    assert p[
        "same_seed_fold"
    ] is True

    assert p[
        "same_fault_id"
    ] is True

    assert p[
        "same_replay_id"
    ] is True

    assert p[
        "same_corrupted_tensor_for_fp32_and_ptq"
    ] is True


def test_thresholds_are_not_used_for_qualification():
    d = load_config()

    assert d[
        "thresholds"
    ][
        "used_during_runner_qualification"
    ] is False

    assert d[
        "thresholds"
    ][
        "reselection_allowed"
    ] is False


def test_no_performance_based_acceptance():
    d = load_config()

    q = d[
        "qualification_execution"
    ]

    assert q[
        "performance_metric_gate"
    ] is False

    assert q[
        "probability_delta_gate"
    ] is False

    assert q[
        "threshold_gate"
    ] is False

    assert q[
        "robustness_claim"
    ] is False


def test_outer_requires_postqualification_manifest():
    d = load_config()

    assert d[
        "future_final_evaluation_policy"
    ][
        "outer_test_requires_separate_post_runner_qualification_execution_manifest"
    ] is True


def test_all_12_families_declared():
    d = load_config()

    families = set(
        d[
            "qualification_execution"
        ][
            "window_fault_families"
        ]
    ) | set(
        d[
            "qualification_execution"
        ][
            "sequence_fault_families"
        ]
    )

    assert len(
        families
    ) == 12

    assert families == {
        "bias",
        "drift",
        "scale_factor",
        "noise",
        "clipping_saturation",
        "stuck_channel",
        "axis_loss",
        "dropout",
        "frame_loss",
        "jitter",
        "delay",
        "orientation",
    }
