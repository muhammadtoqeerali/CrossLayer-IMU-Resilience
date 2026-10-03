import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

ENGINE = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_operators.py"
)

SAMPLING = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling_v3.py"
)

SEVERITY = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def severity():
    return json.loads(
        SEVERITY.read_text()
    )


def window():
    t = np.arange(
        30,
        dtype=float,
    )

    x = np.zeros(
        (30, 9),
        dtype=float,
    )

    for c in range(6):
        x[:, c] = (
            (c + 1) * 10.0
            + t
        )

    x[:, 6] = 100
    x[:, 7] = 200
    x[:, 8] = 300

    return x


def trial(n=300):
    t = np.arange(
        n,
        dtype=float,
    )

    x = np.zeros(
        (n, 9),
        dtype=float,
    )

    x[:, 0] = t
    x[:, 1] = 2 * t
    x[:, 2] = 3 * t
    x[:, 3] = np.sin(t / 13.0)
    x[:, 4] = np.cos(t / 17.0)
    x[:, 5] = t ** 2 / 100.0

    x[:, 6] = 10
    x[:, 7] = 20
    x[:, 8] = 30

    return x


def refs():
    return {
        "q99_abs":
            np.array([
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
            ]),

        "robust_sigma":
            np.array([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]),
    }


def instances():
    s = load_module(
        "sampling_for_tests",
        SAMPLING,
    )

    sev = severity()

    w = s.generate_window_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1|window=0",
        severity_protocol=sev,
    )

    q = s.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1",
        parent_length=300,
        severity_protocol=sev,
    )

    return w, q


def one(rows, family, level="L1"):
    for x in rows:
        if (
            x["family"] == family
            and x["severity"]["level"] == level
        ):
            return x

    raise AssertionError(
        f"missing {family}/{level}"
    )


def test_all_12_families_present():
    w, q = instances()

    families = {
        x["family"]
        for x in w + q
    }

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


def test_bias_exact():
    e = load_module(
        "engine_bias",
        ENGINE,
    )

    w, _ = instances()
    inst = one(
        w,
        "bias",
    )

    x = window()

    y, _ = e.apply_fault(
        x,
        inst,
        reference_scales=refs(),
    )

    ch = inst["target_channels"][0]

    expected = (
        inst["severity"]["sign"]
        * inst["severity"]["fraction_of_q99_abs"]
        * refs()["q99_abs"][ch]
    )

    np.testing.assert_allclose(
        y[:, ch] - x[:, ch],
        expected,
    )


def test_scale_factor_exact():
    e = load_module(
        "engine_scale",
        ENGINE,
    )

    w, _ = instances()
    inst = one(
        w,
        "scale_factor",
    )

    x = window()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    ch = inst["target_channels"][0]

    np.testing.assert_allclose(
        y[:, ch],
        x[:, ch]
        * inst["severity"]["gain"],
    )


def test_noise_deterministic_and_targeted():
    e = load_module(
        "engine_noise",
        ENGINE,
    )

    w, _ = instances()
    inst = one(
        w,
        "noise",
    )

    x = window()

    a, _ = e.apply_fault(
        x,
        inst,
        reference_scales=refs(),
    )

    b, _ = e.apply_fault(
        x,
        inst,
        reference_scales=refs(),
    )

    np.testing.assert_array_equal(
        a,
        b,
    )

    ch = inst["target_channels"][0]

    untouched = [
        c
        for c in range(9)
        if c != ch
    ]

    np.testing.assert_array_equal(
        a[:, untouched],
        x[:, untouched],
    )

    assert not np.array_equal(
        a[:, ch],
        x[:, ch],
    )


def test_clipping_exact_bound():
    e = load_module(
        "engine_clip",
        ENGINE,
    )

    w, _ = instances()
    inst = one(
        w,
        "clipping_saturation",
        "L3",
    )

    x = window() * 100

    y, _ = e.apply_fault(
        x,
        inst,
        reference_scales=refs(),
    )

    ch = inst["target_channels"][0]

    threshold = (
        inst[
            "severity"
        ][
            "threshold_fraction_of_q99_abs"
        ]
        * refs()["q99_abs"][ch]
    )

    assert np.max(
        np.abs(
            y[:, ch]
        )
    ) <= threshold


def test_axis_loss_zeroes_target():
    e = load_module(
        "engine_axis",
        ENGINE,
    )

    w, _ = instances()
    inst = one(
        w,
        "axis_loss",
        "L2",
    )

    x = window()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    np.testing.assert_array_equal(
        y[
            :,
            inst["target_channels"]
        ],
        0.0,
    )


def test_drift_linear_endpoint():
    e = load_module(
        "engine_drift",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "drift",
        "L2",
    )

    x = trial()

    y, _ = e.apply_fault(
        x,
        inst,
        reference_scales=refs(),
    )

    ch = inst["target_channels"][0]
    onset = inst["onset_sample"]
    duration = inst["duration_samples"]

    delta = (
        y[
            onset:onset + duration,
            ch
        ]
        - x[
            onset:onset + duration,
            ch
        ]
    )

    endpoint = (
        inst["severity"]["sign"]
        * inst[
            "severity"
        ][
            "endpoint_fraction_of_q99_abs"
        ]
        * refs()["q99_abs"][ch]
    )

    assert delta[0] == pytest.approx(0.0)
    assert delta[-1] == pytest.approx(endpoint)

    np.testing.assert_array_equal(
        y[:onset, ch],
        x[:onset, ch],
    )

    np.testing.assert_array_equal(
        y[
            onset + duration:,
            ch
        ],
        x[
            onset + duration:,
            ch
        ],
    )


def test_stuck_holds_prior_valid_sample():
    e = load_module(
        "engine_stuck",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "stuck_channel",
        "L3",
    )

    x = trial()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    ch = inst["target_channels"][0]
    onset = inst["onset_sample"]
    duration = inst["duration_samples"]

    np.testing.assert_array_equal(
        y[
            onset:onset + duration,
            ch
        ],
        x[
            onset - 1,
            ch
        ],
    )


def test_dropout_zero_fill():
    e = load_module(
        "engine_dropout",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "dropout",
        "L2",
    )

    x = trial()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    ch = inst["target_channels"][0]
    onset = inst["onset_sample"]
    duration = inst["duration_samples"]

    np.testing.assert_array_equal(
        y[
            onset:onset + duration,
            ch
        ],
        0.0,
    )


def test_frame_loss_causal_forward_fill():
    e = load_module(
        "engine_frame",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "frame_loss",
        "L3",
    )

    assert inst["onset_sample"] >= 1

    x = trial()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    onset = inst["onset_sample"]
    duration = inst["duration_samples"]

    expected = np.repeat(
        x[
            onset - 1:onset,
            0:6
        ],
        duration,
        axis=0,
    )

    np.testing.assert_array_equal(
        y[
            onset:onset + duration,
            0:6
        ],
        expected,
    )


def test_jitter_deterministic():
    e = load_module(
        "engine_jitter",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "jitter",
        "L3",
    )

    x = trial()

    a, audit_a = e.apply_fault(
        x,
        inst,
    )

    b, audit_b = e.apply_fault(
        x,
        inst,
    )

    np.testing.assert_array_equal(
        a,
        b,
    )

    assert audit_a[
        "jitter_query_strictly_ordered"
    ] is True

    assert audit_b[
        "jitter_query_strictly_ordered"
    ] is True


def test_delay_is_causal_shift():
    e = load_module(
        "engine_delay",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "delay",
        "L2",
    )

    x = trial()

    y, _ = e.apply_fault(
        x,
        inst,
    )

    d = inst["severity"]["delay_samples"]

    expected_prefix = np.repeat(
        x[0:1, 0:6],
        d,
        axis=0,
    )

    np.testing.assert_array_equal(
        y[:d, 0:6],
        expected_prefix,
    )

    np.testing.assert_array_equal(
        y[d:, 0:6],
        x[:-d, 0:6],
    )


def test_orientation_is_proper_and_norm_preserving():
    e = load_module(
        "engine_orientation",
        ENGINE,
    )

    _, q = instances()
    inst = one(
        q,
        "orientation",
        "L3",
    )

    x = trial()

    y, audit = e.apply_fault(
        x,
        inst,
    )

    assert audit["rotation_det"] == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert audit[
        "rotation_orthogonality_error"
    ] < 1e-12

    np.testing.assert_allclose(
        np.linalg.norm(
            y[:, 0:3],
            axis=1,
        ),
        np.linalg.norm(
            x[:, 0:3],
            axis=1,
        ),
        atol=1e-10,
        rtol=1e-10,
    )

    np.testing.assert_allclose(
        np.linalg.norm(
            y[:, 3:6],
            axis=1,
        ),
        np.linalg.norm(
            x[:, 3:6],
            axis=1,
        ),
        atol=1e-10,
        rtol=1e-10,
    )


def test_every_family_preserves_euler():
    e = load_module(
        "engine_euler",
        ENGINE,
    )

    w, q = instances()

    for rows, x in [
        (w, window()),
        (q, trial()),
    ]:
        seen = set()

        for inst in rows:
            family = inst["family"]

            if family in seen:
                continue

            seen.add(
                family
            )

            kwargs = {}

            if family in {
                "bias",
                "drift",
                "clipping_saturation",
                "noise",
            }:
                kwargs[
                    "reference_scales"
                ] = refs()

            y, _ = e.apply_fault(
                x,
                inst,
                **kwargs,
            )

            np.testing.assert_array_equal(
                y[:, 6:9],
                x[:, 6:9],
            )


def test_input_array_is_never_mutated():
    e = load_module(
        "engine_immutable",
        ENGINE,
    )

    w, q = instances()

    for inst, x in [
        (
            one(w, "noise"),
            window(),
        ),
        (
            one(q, "frame_loss"),
            trial(),
        ),
        (
            one(q, "jitter"),
            trial(),
        ),
    ]:
        before = x.copy()

        kwargs = {}

        if inst["family"] == "noise":
            kwargs[
                "reference_scales"
            ] = refs()

        e.apply_fault(
            x,
            inst,
            **kwargs,
        )

        np.testing.assert_array_equal(
            x,
            before,
        )


def test_reference_dependent_family_rejects_missing_scale():
    e = load_module(
        "engine_refs",
        ENGINE,
    )

    w, _ = instances()

    with pytest.raises(ValueError):
        e.apply_fault(
            window(),
            one(w, "bias"),
        )


def test_onfield_execution_rejected():
    e = load_module(
        "engine_onfield",
        ENGINE,
    )

    w, _ = instances()
    inst = dict(
        one(w, "axis_loss")
    )

    inst["partition"] = "onfield"

    with pytest.raises(ValueError):
        e.apply_fault(
            window(),
            inst,
        )


def test_wrong_injection_layer_rejected():
    e = load_module(
        "engine_layer",
        ENGINE,
    )

    w, _ = instances()
    inst = dict(
        one(w, "axis_loss")
    )

    inst["injection_layer"] = (
        "source_faithful_oriented_trial_before_filtering_windowing"
    )

    with pytest.raises(ValueError):
        e.apply_fault(
            window(),
            inst,
        )
