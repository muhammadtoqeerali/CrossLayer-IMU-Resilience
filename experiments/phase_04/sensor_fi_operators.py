"""Phase-4H complete sensor fault operator engine.

Evidence tier: P0 software fault injection.

This module mutates only in-memory copies of N x 9 measurement arrays.
It does not load models, perform inference, select thresholds, or write
dataset files.

Authoritative injection layers
------------------------------
Window-value families:
    bias
    scale_factor
    noise
    clipping_saturation
    axis_loss

Sequence families:
    drift
    stuck_channel
    dropout
    frame_loss
    jitter
    delay
    orientation

All learned-model-effective channels are 0..5.
Euler channels 6..8 are preserved unless a future protocol version
explicitly changes that rule.

Reference-scale-dependent families
----------------------------------
bias, drift, clipping_saturation:
    fold/channel q99 absolute reference

noise:
    fold/channel robust-sigma reference

Reference vectors are supplied by the caller from the already-frozen
fold-local training-calibration materialization. This engine never
estimates reference scales from the input being faulted.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Mapping

import numpy as np


WINDOW_LAYER = "stored_filtered_window_before_IMUNormalizer"

SEQUENCE_LAYER = (
    "source_faithful_oriented_trial_before_filtering_windowing"
)

WINDOW_FAMILIES = frozenset({
    "bias",
    "scale_factor",
    "noise",
    "clipping_saturation",
    "axis_loss",
})

SEQUENCE_FAMILIES = frozenset({
    "drift",
    "stuck_channel",
    "dropout",
    "frame_loss",
    "jitter",
    "delay",
    "orientation",
})

ALL_FAMILIES = WINDOW_FAMILIES | SEQUENCE_FAMILIES

EFFECTIVE_CHANNELS = tuple(range(6))
EULER_CHANNELS = (6, 7, 8)


def _load_jitter_helper():
    path = (
        Path(__file__).resolve().parent
        / "sensor_fi_jitter.py"
    )

    spec = importlib.util.spec_from_file_location(
        "_phase4h_sensor_fi_jitter",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_JITTER = _load_jitter_helper()


def _array9(data):
    x = np.asarray(
        data,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "fault input must be a 2D matrix"
        )

    if x.shape[1] != 9:
        raise ValueError(
            "fault input must contain exactly 9 measurement channels"
        )

    if x.shape[0] < 1:
        raise ValueError(
            "fault input must contain at least one row"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            "fault input must be finite"
        )

    return x


def _positive_reference_vector(
    value,
    *,
    name,
):
    x = np.asarray(
        value,
        dtype=np.float64,
    )

    if x.shape != (6,):
        raise ValueError(
            f"{name} must have shape (6,)"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            f"{name} must be finite"
        )

    if not np.all(
        x > 0.0
    ):
        raise ValueError(
            f"{name} must be strictly positive"
        )

    return x


def _require_q99(
    reference_scales,
):
    if reference_scales is None:
        raise ValueError(
            "q99 reference scales are required"
        )

    if "q99_abs" not in reference_scales:
        raise ValueError(
            "reference_scales missing q99_abs"
        )

    return _positive_reference_vector(
        reference_scales["q99_abs"],
        name="q99_abs",
    )


def _require_sigma(
    reference_scales,
):
    if reference_scales is None:
        raise ValueError(
            "robust-sigma reference scales are required"
        )

    if "robust_sigma" not in reference_scales:
        raise ValueError(
            "reference_scales missing robust_sigma"
        )

    return _positive_reference_vector(
        reference_scales["robust_sigma"],
        name="robust_sigma",
    )


def _validate_instance(
    data,
    instance,
):
    x = _array9(
        data
    )

    if not isinstance(
        instance,
        Mapping,
    ):
        raise ValueError(
            "instance must be a mapping"
        )

    family = str(
        instance.get(
            "family",
            "",
        )
    )

    if family not in ALL_FAMILIES:
        raise ValueError(
            f"unsupported fault family: {family}"
        )

    if instance.get("domain") != "sensor":
        raise ValueError(
            "instance domain must be sensor"
        )

    if instance.get("evidence_tier") != "P0":
        raise ValueError(
            "instance evidence tier must be P0"
        )

    if instance.get("partition") == "onfield":
        raise ValueError(
            "OnField fault execution is prohibited"
        )

    targets = tuple(
        int(c)
        for c in instance.get(
            "target_channels",
            [],
        )
    )

    if not targets:
        raise ValueError(
            "instance has no target channels"
        )

    if any(
        c not in EFFECTIVE_CHANNELS
        for c in targets
    ):
        raise ValueError(
            "only effective channels 0..5 may be targeted"
        )

    expected_layer = (
        WINDOW_LAYER
        if family in WINDOW_FAMILIES
        else SEQUENCE_LAYER
    )

    if instance.get(
        "injection_layer"
    ) != expected_layer:
        raise ValueError(
            "instance injection layer does not match family"
        )

    parent_kind = instance.get(
        "parent_kind"
    )

    if family in WINDOW_FAMILIES:
        if parent_kind != "stored_window":
            raise ValueError(
                "window family requires stored_window parent"
            )

        if x.shape[0] != 30:
            raise ValueError(
                "window-value operator requires exactly 30 rows"
            )

    else:
        if parent_kind != "source_trial":
            raise ValueError(
                "sequence family requires source_trial parent"
            )

    onset = int(
        instance.get(
            "onset_sample",
            0,
        )
    )

    duration = instance.get(
        "duration_samples"
    )

    if onset < 0:
        raise ValueError(
            "onset must be non-negative"
        )

    if duration is not None:
        duration = int(
            duration
        )

        if duration <= 0:
            raise ValueError(
                "duration must be positive when supplied"
            )

        if onset + duration > x.shape[0]:
            raise ValueError(
                "fault episode exceeds parent bounds"
            )

    if family == "frame_loss":
        if onset < 1:
            raise ValueError(
                "frame loss requires a causal predecessor frame"
            )

    if family in {
        "jitter",
        "delay",
        "orientation",
    }:
        if onset != 0:
            raise ValueError(
                f"{family} must start at sample zero"
            )

        if duration is not None:
            raise ValueError(
                f"{family} must persist until end"
            )

    return (
        x,
        family,
        targets,
        onset,
        duration,
    )


def _episode_slice(
    onset,
    duration,
):
    if duration is None:
        return slice(
            onset,
            None,
        )

    return slice(
        onset,
        onset + duration,
    )


def _rotation_matrix(
    axis,
    degrees,
):
    axis = str(
        axis
    ).lower()

    theta = math.radians(
        float(
            degrees
        )
    )

    c = math.cos(
        theta
    )

    s = math.sin(
        theta
    )

    if axis == "x":
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ])

    elif axis == "y":
        R = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])

    elif axis == "z":
        R = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    else:
        raise ValueError(
            f"unsupported rotation axis: {axis}"
        )

    return R


def apply_fault(
    data,
    instance,
    *,
    reference_scales=None,
):
    """Apply one frozen Phase-4H P0 sensor fault instance.

    Parameters
    ----------
    data
        N x 9 input array at the instance's frozen injection layer.

    instance
        Metadata generated by qualified sampling v3.

    reference_scales
        Optional mapping. Required only by data-scale-dependent families.

        {
            "q99_abs": length-6 strictly positive array,
            "robust_sigma": length-6 strictly positive array,
        }

        These values must come from the already-frozen fold-local
        training-calibration materialization. The operator never estimates
        them from the target sample.

    Returns
    -------
    output, audit
    """

    (
        x,
        family,
        targets,
        onset,
        duration,
    ) = _validate_instance(
        data,
        instance,
    )

    y = x.copy()

    severity = instance.get(
        "severity",
        {}
    )

    episode = _episode_slice(
        onset,
        duration,
    )

    # --------------------------------------------------------
    # WINDOW-VALUE FAMILIES
    # --------------------------------------------------------

    if family == "bias":
        q99 = _require_q99(
            reference_scales
        )

        fraction = float(
            severity[
                "fraction_of_q99_abs"
            ]
        )

        sign = int(
            severity["sign"]
        )

        if sign not in {
            -1,
            1,
        }:
            raise ValueError(
                "bias sign must be -1 or +1"
            )

        for channel in targets:
            delta = (
                sign
                * fraction
                * q99[channel]
            )

            y[
                episode,
                channel
            ] += delta

    elif family == "scale_factor":
        gain = float(
            severity[
                "gain"
            ]
        )

        if not np.isfinite(
            gain
        ):
            raise ValueError(
                "scale-factor gain must be finite"
            )

        for channel in targets:
            y[
                episode,
                channel
            ] *= gain

    elif family == "noise":
        sigma_ref = _require_sigma(
            reference_scales
        )

        fraction = float(
            severity[
                "sigma_fraction_of_robust_sigma"
            ]
        )

        if severity.get(
            "distribution"
        ) != "zero_mean_gaussian":
            raise ValueError(
                "noise distribution must be zero_mean_gaussian"
            )

        rng = np.random.default_rng(
            int(
                instance["seed"]
            )
        )

        count = (
            x.shape[0]
            if duration is None
            else duration
        )

        for channel in targets:
            sigma = (
                fraction
                * sigma_ref[channel]
            )

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=count,
            )

            y[
                episode,
                channel
            ] += noise

    elif family == "clipping_saturation":
        q99 = _require_q99(
            reference_scales
        )

        fraction = float(
            severity[
                "threshold_fraction_of_q99_abs"
            ]
        )

        for channel in targets:
            threshold = (
                fraction
                * q99[channel]
            )

            y[
                episode,
                channel
            ] = np.clip(
                y[
                    episode,
                    channel
                ],
                -threshold,
                threshold,
            )

    elif family == "axis_loss":
        replacement = float(
            severity.get(
                "replacement",
                0.0,
            )
        )

        y[
            episode,
            list(targets)
        ] = replacement

    # --------------------------------------------------------
    # SEQUENCE FAMILIES
    # --------------------------------------------------------

    elif family == "drift":
        q99 = _require_q99(
            reference_scales
        )

        fraction = float(
            severity[
                "endpoint_fraction_of_q99_abs"
            ]
        )

        sign = int(
            severity[
                "sign"
            ]
        )

        if sign not in {
            -1,
            1,
        }:
            raise ValueError(
                "drift sign must be -1 or +1"
            )

        if severity.get(
            "trajectory"
        ) != "linear_from_zero_to_endpoint":
            raise ValueError(
                "unsupported drift trajectory"
            )

        if duration is None:
            raise ValueError(
                "drift requires a finite duration"
            )

        for channel in targets:
            endpoint = (
                sign
                * fraction
                * q99[channel]
            )

            ramp = np.linspace(
                0.0,
                endpoint,
                duration,
                endpoint=True,
                dtype=np.float64,
            )

            y[
                episode,
                channel
            ] += ramp

    elif family == "stuck_channel":
        if duration is None:
            raise ValueError(
                "stuck_channel requires a duration"
            )

        if onset < 1:
            raise ValueError(
                "stuck_channel requires a prior valid sample"
            )

        for channel in targets:
            hold_value = x[
                onset - 1,
                channel
            ]

            y[
                episode,
                channel
            ] = hold_value

    elif family == "dropout":
        replacement = float(
            severity.get(
                "replacement",
                0.0,
            )
        )

        y[
            episode,
            list(targets)
        ] = replacement

    elif family == "frame_loss":
        if duration is None:
            raise ValueError(
                "frame_loss requires a duration"
            )

        previous_complete_frame = x[
            onset - 1,
            list(targets)
        ].copy()

        y[
            episode,
            list(targets)
        ] = previous_complete_frame

    elif family == "jitter":
        if tuple(
            sorted(
                targets
            )
        ) != EFFECTIVE_CHANNELS:
            raise ValueError(
                "jitter must target all six effective channels"
            )

        max_ms = float(
            severity[
                "maximum_absolute_timestamp_displacement_ms"
            ]
        )

        y, jitter_meta = (
            _JITTER.apply_jitter_time_warp(
                x,
                max_displacement_ms=max_ms,
                seed=int(
                    instance["seed"]
                ),
            )
        )

    elif family == "delay":
        if tuple(
            sorted(
                targets
            )
        ) != EFFECTIVE_CHANNELS:
            raise ValueError(
                "delay must target all six effective channels"
            )

        delay = int(
            severity[
                "delay_samples"
            ]
        )

        if delay <= 0:
            raise ValueError(
                "delay_samples must be positive"
            )

        if delay >= x.shape[0]:
            raise ValueError(
                "delay must be shorter than parent trial"
            )

        y[
            :delay,
            list(targets)
        ] = x[
            0,
            list(targets)
        ]

        y[
            delay:,
            list(targets)
        ] = x[
            :-delay,
            list(targets)
        ]

    elif family == "orientation":
        if tuple(
            sorted(
                targets
            )
        ) != EFFECTIVE_CHANNELS:
            raise ValueError(
                "orientation must target all six effective channels"
            )

        axis = severity[
            "rotation_axis"
        ]

        degrees = float(
            severity[
                "rotation_degrees"
            ]
        )

        R = _rotation_matrix(
            axis,
            degrees,
        )

        y[
            :,
            0:3
        ] = (
            x[
                :,
                0:3
            ]
            @ R.T
        )

        y[
            :,
            3:6
        ] = (
            x[
                :,
                3:6
            ]
            @ R.T
        )

    else:
        raise AssertionError(
            f"unhandled family: {family}"
        )

    if not np.isfinite(
        y
    ).all():
        raise RuntimeError(
            "fault operator produced non-finite output"
        )

    # Euler channels are frozen as non-target channels for every
    # Phase-4H family.
    if not np.array_equal(
        y[
            :,
            EULER_CHANNELS
        ],
        x[
            :,
            EULER_CHANNELS
        ],
    ):
        raise RuntimeError(
            "fault operator changed frozen Euler channels"
        )

    delta = (
        y
        - x
    )

    audit = {
        "family":
            family,

        "fault_id":
            instance.get(
                "fault_id"
            ),

        "replay_id":
            instance.get(
                "replay_id"
            ),

        "seed":
            int(
                instance.get(
                    "seed",
                    0,
                )
            ),

        "input_shape":
            list(
                x.shape
            ),

        "output_shape":
            list(
                y.shape
            ),

        "target_channels":
            list(
                targets
            ),

        "onset_sample":
            onset,

        "duration_samples":
            duration,

        "changed_value_count":
            int(
                np.count_nonzero(
                    delta
                )
            ),

        "max_abs_delta":
            float(
                np.max(
                    np.abs(
                        delta
                    )
                )
            ),

        "output_finite":
            True,

        "Euler_preserved":
            True,

        "input_mutated":
            False,
    }

    if family == "jitter":
        audit[
            "jitter_requested_max_abs_ms"
        ] = float(
            np.max(
                np.abs(
                    jitter_meta[
                        "requested_displacement_ms"
                    ]
                )
            )
        )

        audit[
            "jitter_query_strictly_ordered"
        ] = bool(
            np.all(
                np.diff(
                    jitter_meta[
                        "query_time_ms"
                    ]
                ) > 0.0
            )
        )

    if family == "orientation":
        R = _rotation_matrix(
            severity[
                "rotation_axis"
            ],
            float(
                severity[
                    "rotation_degrees"
                ]
            ),
        )

        audit[
            "rotation_det"
        ] = float(
            np.linalg.det(
                R
            )
        )

        audit[
            "rotation_orthogonality_error"
        ] = float(
            np.max(
                np.abs(
                    R.T @ R
                    - np.eye(3)
                )
            )
        )

    return (
        y,
        audit,
    )
