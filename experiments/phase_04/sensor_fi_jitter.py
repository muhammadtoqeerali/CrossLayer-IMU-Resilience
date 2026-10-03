"""Phase-4H executable P0 jitter semantics.

The historical preprocessing pipeline consumes ordered sensor rows at
nominal 100 Hz and does not use source timestamps to construct windows.

Therefore jitter is represented as a bounded software time-warp:

    t_i = i * 10 ms
    epsilon_i ~ Uniform(-J,+J)
    q_i = clip(t_i + epsilon_i, 0, t_last)
    y_i,c = linear_interp(q_i, t, x_c)

One epsilon is drawn per frame and shared across all six effective
accelerometer/gyroscope channels.

Euler channels are preserved exactly.

This is P0 software stress, not a physical timing-realism claim.
"""

from __future__ import annotations

import numpy as np


NOMINAL_PERIOD_MS = 10.0

EFFECTIVE_CHANNELS = (
    0,
    1,
    2,
    3,
    4,
    5,
)

EULER_CHANNELS = (
    6,
    7,
    8,
)


def apply_jitter_time_warp(
    trial,
    *,
    max_displacement_ms,
    seed,
):
    """Apply deterministic bounded jitter to an N x 9 trial matrix.

    Returns
    -------
    output : np.ndarray
        Float64 N x 9 output. Channels 0..5 are time-warped.
        Channels 6..8 are byte-value equivalent after float conversion.

    metadata : dict
        Deterministic displacement/query-time arrays for audit/testing.
    """

    x = np.asarray(
        trial,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "trial must be a 2D matrix"
        )

    if x.shape[1] != 9:
        raise ValueError(
            "trial must contain exactly 9 measurement channels"
        )

    if x.shape[0] < 2:
        raise ValueError(
            "trial must contain at least two frames"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            "trial must contain only finite values"
        )

    max_displacement_ms = float(
        max_displacement_ms
    )

    if not np.isfinite(
        max_displacement_ms
    ):
        raise ValueError(
            "max_displacement_ms must be finite"
        )

    if max_displacement_ms < 0.0:
        raise ValueError(
            "max_displacement_ms must be non-negative"
        )

    if not (
        max_displacement_ms
        < NOMINAL_PERIOD_MS / 2.0
    ):
        raise ValueError(
            "jitter bound must be strictly below half "
            "the nominal sample period"
        )

    n = x.shape[0]

    nominal_time_ms = (
        np.arange(
            n,
            dtype=np.float64,
        )
        * NOMINAL_PERIOD_MS
    )

    if max_displacement_ms == 0.0:
        requested_displacement_ms = np.zeros(
            n,
            dtype=np.float64,
        )
    else:
        rng = np.random.default_rng(
            int(
                seed
            )
        )

        requested_displacement_ms = rng.uniform(
            -max_displacement_ms,
            max_displacement_ms,
            size=n,
        )

    query_time_ms = np.clip(
        nominal_time_ms
        + requested_displacement_ms,
        nominal_time_ms[0],
        nominal_time_ms[-1],
    )

    if not np.all(
        np.diff(
            query_time_ms
        ) > 0.0
    ):
        raise RuntimeError(
            "jitter semantics violated strict temporal ordering"
        )

    effective_displacement_ms = (
        query_time_ms
        - nominal_time_ms
    )

    y = x.copy()

    for channel in EFFECTIVE_CHANNELS:
        y[
            :,
            channel
        ] = np.interp(
            query_time_ms,
            nominal_time_ms,
            x[
                :,
                channel
            ],
        )

    # Explicit preservation invariant.
    y[
        :,
        EULER_CHANNELS
    ] = x[
        :,
        EULER_CHANNELS
    ]

    metadata = {
        "nominal_period_ms":
            NOMINAL_PERIOD_MS,

        "max_displacement_ms":
            max_displacement_ms,

        "seed":
            int(
                seed
            ),

        "requested_displacement_ms":
            requested_displacement_ms,

        "effective_displacement_ms":
            effective_displacement_ms,

        "nominal_time_ms":
            nominal_time_ms,

        "query_time_ms":
            query_time_ms,

        "effective_channels":
            EFFECTIVE_CHANNELS,

        "Euler_channels":
            EULER_CHANNELS,
    }

    return y, metadata
