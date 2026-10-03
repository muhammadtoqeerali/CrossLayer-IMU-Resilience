import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_jitter_semantics_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_jitter.py"
)


def load_config():
    return json.loads(
        CONFIG.read_text()
    )


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_jitter",
        IMPL,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synthetic_trial(n=200):
    t = np.arange(
        n,
        dtype=float,
    )

    x = np.zeros(
        (n, 9),
        dtype=float,
    )

    x[:, 0] = t
    x[:, 1] = 2.0 * t
    x[:, 2] = -3.0 * t
    x[:, 3] = np.sin(t / 7.0)
    x[:, 4] = np.cos(t / 11.0)
    x[:, 5] = t ** 2 / 100.0

    x[:, 6] = 10.0
    x[:, 7] = 20.0
    x[:, 8] = 30.0

    return x


def test_frozen_levels():
    d = load_config()

    assert d[
        "random_displacement"
    ][
        "J_from_frozen_severity_level_ms"
    ] == {
        "L1": 1.0,
        "L2": 2.0,
        "L3": 4.0,
    }


def test_zero_jitter_is_exact_identity():
    m = load_impl()
    x = synthetic_trial()

    y, meta = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=0.0,
        seed=123,
    )

    np.testing.assert_array_equal(
        y,
        x,
    )

    np.testing.assert_array_equal(
        meta["query_time_ms"],
        meta["nominal_time_ms"],
    )


def test_same_seed_is_deterministic():
    m = load_impl()
    x = synthetic_trial()

    a, am = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=12345,
    )

    b, bm = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=12345,
    )

    np.testing.assert_array_equal(
        a,
        b,
    )

    np.testing.assert_array_equal(
        am["requested_displacement_ms"],
        bm["requested_displacement_ms"],
    )


def test_different_seed_changes_nonconstant_signal():
    m = load_impl()
    x = synthetic_trial()

    a, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=1,
    )

    b, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=2,
    )

    assert not np.array_equal(
        a[:, :6],
        b[:, :6],
    )


def test_requested_and_effective_displacements_are_bounded():
    m = load_impl()
    x = synthetic_trial()

    _, meta = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=777,
    )

    assert np.max(
        np.abs(
            meta["requested_displacement_ms"]
        )
    ) <= 4.0

    assert np.max(
        np.abs(
            meta["effective_displacement_ms"]
        )
    ) <= 4.0


def test_query_times_remain_strictly_ordered():
    m = load_impl()
    x = synthetic_trial()

    _, meta = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=42,
    )

    assert np.all(
        np.diff(
            meta["query_time_ms"]
        ) > 0.0
    )


def test_linear_channel_matches_exact_shifted_time():
    m = load_impl()
    x = synthetic_trial()

    y, meta = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=2025,
    )

    expected = (
        meta["query_time_ms"]
        / 10.0
    )

    np.testing.assert_allclose(
        y[:, 0],
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        y[:, 1],
        2.0 * expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_one_frame_time_warp_is_shared_across_channels():
    m = load_impl()
    x = synthetic_trial()

    y, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=99,
    )

    np.testing.assert_allclose(
        y[:, 1],
        2.0 * y[:, 0],
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        y[:, 2],
        -3.0 * y[:, 0],
        atol=1e-12,
        rtol=0.0,
    )


def test_euler_channels_are_exactly_preserved():
    m = load_impl()
    x = synthetic_trial()

    y, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=123,
    )

    np.testing.assert_array_equal(
        y[:, 6:9],
        x[:, 6:9],
    )


def test_shape_and_row_count_are_preserved():
    m = load_impl()
    x = synthetic_trial(
        n=317
    )

    y, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=2.0,
        seed=88,
    )

    assert y.shape == x.shape


def test_constant_effective_signal_is_invariant():
    m = load_impl()

    x = np.ones(
        (100, 9),
        dtype=float,
    )

    x[:, 6] = 7
    x[:, 7] = 8
    x[:, 8] = 9

    y, _ = m.apply_jitter_time_warp(
        x,
        max_displacement_ms=4.0,
        seed=100,
    )

    np.testing.assert_array_equal(
        y,
        x,
    )


def test_half_period_or_larger_is_rejected():
    m = load_impl()
    x = synthetic_trial()

    with pytest.raises(ValueError):
        m.apply_jitter_time_warp(
            x,
            max_displacement_ms=5.0,
            seed=1,
        )


def test_wrong_channel_count_is_rejected():
    m = load_impl()

    with pytest.raises(ValueError):
        m.apply_jitter_time_warp(
            np.zeros((100, 6)),
            max_displacement_ms=1.0,
            seed=1,
        )


def test_claim_boundary_is_p0_only():
    d = load_config()

    assert d["evidence_tier"] == "P0"
    assert d["physical_realism_claim"] is False

    assert d[
        "governance"
    ][
        "outer_test_used"
    ] is False

    assert d[
        "governance"
    ][
        "OnField_used"
    ] is False
