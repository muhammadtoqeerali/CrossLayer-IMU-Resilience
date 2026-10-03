"""Phase-4H severity protocol helpers.

This module materializes the frozen v1 software-stress coordinates.
It intentionally does not apply sensor faults and does not perform
model inference.
"""

from __future__ import annotations

import math
import numpy as np

LEVELS = ("L1", "L2", "L3")

BIAS_FRACTIONS = {
    "L1": 0.05,
    "L2": 0.10,
    "L3": 0.20,
}

DRIFT_ENDPOINT_FRACTIONS = {
    "L1": 0.05,
    "L2": 0.10,
    "L3": 0.20,
}

SCALE_DELTAS = {
    "L1": 0.05,
    "L2": 0.10,
    "L3": 0.20,
}

NOISE_SIGMA_FRACTIONS = {
    "L1": 0.50,
    "L2": 1.00,
    "L3": 2.00,
}

CLIPPING_Q99_FRACTIONS = {
    "L1": 1.00,
    "L2": 0.75,
    "L3": 0.50,
}

STUCK_DURATION_SAMPLES = {
    "L1": 5,
    "L2": 15,
    "L3": 30,
}

DROPOUT_DURATION_SAMPLES = {
    "L1": 5,
    "L2": 15,
    "L3": 30,
}

FRAME_LOSS_COUNTS = {
    "L1": 1,
    "L2": 3,
    "L3": 5,
}

JITTER_MAX_MS = {
    "L1": 1.0,
    "L2": 2.0,
    "L3": 4.0,
}

DELAY_SAMPLES = {
    "L1": 1,
    "L2": 5,
    "L3": 10,
}

ORIENTATION_DEGREES = {
    "L1": 5.0,
    "L2": 15.0,
    "L3": 30.0,
}

AXIS_LOSS_COUNTS = {
    "L1": 1,
    "L2": 3,
    "L3": 6,
}


def materialize_reference_fraction(reference, fractions):
    reference = np.asarray(reference, dtype=float)

    if (
        reference.ndim != 1
        or reference.size != 6
    ):
        raise ValueError(
            "reference must be six effective-channel values"
        )

    if (
        not np.isfinite(reference).all()
        or np.any(reference <= 0)
    ):
        raise ValueError(
            "reference values must be finite and positive"
        )

    return {
        level:
            (
                reference
                * float(fractions[level])
            ).tolist()

        for level in LEVELS
    }


def bias_magnitudes(q99_abs):
    return materialize_reference_fraction(
        q99_abs,
        BIAS_FRACTIONS,
    )


def drift_endpoint_magnitudes(q99_abs):
    return materialize_reference_fraction(
        q99_abs,
        DRIFT_ENDPOINT_FRACTIONS,
    )


def noise_sigmas(robust_sigma):
    return materialize_reference_fraction(
        robust_sigma,
        NOISE_SIGMA_FRACTIONS,
    )


def clipping_thresholds(q99_abs):
    return materialize_reference_fraction(
        q99_abs,
        CLIPPING_Q99_FRACTIONS,
    )


def scale_gains():
    return {
        level: [
            1.0 - SCALE_DELTAS[level],
            1.0 + SCALE_DELTAS[level],
        ]
        for level in LEVELS
    }


def rotation_matrix(axis, degrees):
    theta = math.radians(
        float(degrees)
    )

    c = math.cos(theta)
    s = math.sin(theta)

    if axis == "x":
        return np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ])

    if axis == "y":
        return np.asarray([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])

    if axis == "z":
        return np.asarray([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    raise ValueError(
        f"unsupported rotation axis: {axis}"
    )
