import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

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


def load_module():
    spec = importlib.util.spec_from_file_location(
        "phase4h_runner_v2",
        RUNNER,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def config():
    return json.loads(
        CONFIG.read_text()
    )


def test_single_scientific_repair():
    d = config()
    repair = d[
        "single_scientific_repair"
    ]

    assert repair[
        "v2_authoritative_semantic_name"
    ] == "fall_start_frame"

    assert repair[
        "frozen_rule"
    ] == (
        "fall_start_frame + 15 * falling_local_index"
    )

    assert repair[
        "no_other_scientific_parameter_changed"
    ] is True


def test_dataset_frame_conventions():
    d = config()

    conventions = d[
        "single_scientific_repair"
    ][
        "dataset_conventions"
    ]

    assert (
        "zero-based fall_start_position"
        in conventions[
            "UNIVR"
        ]
    )

    assert (
        "+ 1"
        in conventions[
            "KFALL"
        ]
    )


def test_kfall_subject_parser():
    m = load_module()

    assert m.numeric_subject(
        "SA06"
    ) == 6

    assert m.storage_subject(
        dataset_id="KFALL",
        source_subject_id="SA06",
    ) == 106


def test_univr_subject_parser():
    m = load_module()

    assert m.storage_subject(
        dataset_id="UNIVR",
        source_subject_id="SA09",
    ) == 9


def test_piecewise_falling_uses_frame_directly(monkeypatch):
    m = load_module()

    monkeypatch.setattr(
        m,
        "low_pass_filter",
        lambda x, cutoff, rate: np.asarray(
            x
        ).copy(),
    )

    x = np.arange(
        250 * 9,
        dtype=float,
    ).reshape(
        250,
        9,
    )

    labels = np.asarray(
        [
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
        fall_start_frame=101,
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

    np.testing.assert_array_equal(
        windows[
            2
        ],
        x[
            101:131
        ],
    )

    np.testing.assert_array_equal(
        windows[
            3
        ],
        x[
            116:146
        ],
    )


def test_activity_only_requires_no_event_route(monkeypatch):
    m = load_module()

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
        4,
        dtype=np.int8,
    )

    windows = m.rewindow_sequence(
        x,
        labels,
        fall_start_frame=None,
    )

    np.testing.assert_array_equal(
        windows[
            3
        ],
        x[
            45:75
        ],
    )


def test_falling_requires_frame():
    m = load_module()

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
                ],
                dtype=np.int8,
            ),
            fall_start_frame=None,
        )


def test_route_offset_conventions():
    m = load_module()

    assert m.route_offset_ok(
        subject=9,
        fall_start_frame=200,
        fall_start_position=200,
    )

    assert m.route_offset_ok(
        subject=106,
        fall_start_frame=200,
        fall_start_position=199,
    )

    assert not m.route_offset_ok(
        subject=106,
        fall_start_frame=199,
        fall_start_position=199,
    )


def test_v1_preserved_by_design():
    d = config()

    assert d[
        "runner_v1_preservation"
    ][
        "v1_rewritten"
    ] is False


def test_qualification_is_training_calibration_only():
    d = config()
    boundary = d[
        "scientific_boundary"
    ]

    assert boundary[
        "training_calibration_only"
    ] is True

    assert boundary[
        "validation_allowed"
    ] is False

    assert boundary[
        "outer_test_allowed"
    ] is False

    assert boundary[
        "onfield_allowed"
    ] is False


def test_no_performance_or_threshold_acceptance():
    d = config()
    q = d[
        "qualification"
    ]

    assert q[
        "performance_acceptance_gate"
    ] is False

    assert q[
        "probability_delta_gate"
    ] is False

    assert q[
        "thresholds_applied"
    ] is False

    assert q[
        "threshold_selection"
    ] is False


def test_expected_falling_parent_counts():
    d = config()
    q = d[
        "qualification"
    ]

    assert q[
        "expected_unique_falling_parent_count"
    ] == 2269

    assert q[
        "expected_dataset_parent_counts"
    ] == {
        "KFALL":
            1813,

        "UNIVR":
            456,
    }
