"""Phase-4H sensor-FI instance sampling and replay metadata.

Pre-robustness v1 protocol.

This module creates deterministic fault-instance metadata only.
It does not mutate sensor data and does not execute model inference.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

NAMESPACE = "crosslayer-phase4h-sensor-fi-instance-v1"

WINDOW_LAYER = "stored_filtered_window_before_IMUNormalizer"

SEQUENCE_LAYER = (
    "source_faithful_oriented_trial_before_filtering_windowing"
)

LEVELS = ("L1", "L2", "L3")

SINGLE_CHANNELS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
)

ALL_EFFECTIVE_CHANNELS = (
    0,
    1,
    2,
    3,
    4,
    5,
)

STOCHASTIC_REPLICATES = {
    "noise": 3,
    "dropout": 3,
    "frame_loss": 3,
    "jitter": 3,
}


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def derive_seed(
    *,
    partition,
    fold,
    parent_kind,
    parent_sequence_id,
    family,
    severity_level,
    variant_id,
    replicate_index,
):
    payload = "|".join([
        NAMESPACE,
        str(partition),
        str(fold),
        str(parent_kind),
        str(parent_sequence_id),
        str(family),
        str(severity_level),
        str(variant_id),
        str(replicate_index),
    ])

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def compute_replay_id(instance):
    payload = dict(instance)
    payload.pop("replay_id", None)

    return hashlib.sha256(
        canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()


def _fault_id(
    *,
    family,
    level,
    variant_id,
    replicate_index,
    parent_sequence_id,
):
    parent_hash = hashlib.sha256(
        str(
            parent_sequence_id
        ).encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"p4h-{family}-{level}-"
        f"{variant_id}-r{replicate_index}-"
        f"{parent_hash}"
    )


def _base_instance(
    *,
    family,
    level,
    variant_id,
    replicate_index,
    partition,
    fold,
    parent_kind,
    parent_sequence_id,
    injection_layer,
    target_channels,
    onset_sample,
    duration_samples,
    persistence,
    severity,
):
    seed = derive_seed(
        partition=partition,
        fold=fold,
        parent_kind=parent_kind,
        parent_sequence_id=parent_sequence_id,
        family=family,
        severity_level=level,
        variant_id=variant_id,
        replicate_index=replicate_index,
    )

    instance = {
        "fault_id":
            _fault_id(
                family=family,
                level=level,
                variant_id=variant_id,
                replicate_index=replicate_index,
                parent_sequence_id=parent_sequence_id,
            ),

        "family":
            family,

        "domain":
            "sensor",

        "evidence_tier":
            "P0",

        "injection_layer":
            injection_layer,

        "target_channels":
            list(
                target_channels
            ),

        "onset_sample":
            int(
                onset_sample
            ),

        "duration_samples":
            (
                None
                if duration_samples is None
                else int(
                    duration_samples
                )
            ),

        "persistence":
            persistence,

        "severity":
            severity,

        "severity_provenance": {
            "evidence_tier":
                "P0",

            "physical_realism_claim":
                False,

            "protocol":
                "phase4h_sensor_fi_severity_protocol_v1",
        },

        "seed":
            int(
                seed
            ),

        "parent_sequence_id":
            str(
                parent_sequence_id
            ),

        "partition":
            str(
                partition
            ),

        "fold":
            int(
                fold
            ),

        "parent_kind":
            str(
                parent_kind
            ),

        "variant_id":
            str(
                variant_id
            ),

        "replicate_index":
            int(
                replicate_index
            ),

        "replay_id":
            "",
    }

    instance["replay_id"] = compute_replay_id(
        instance
    )

    return instance


def _centered_onset(
    parent_length,
    duration,
):
    if duration <= 0:
        raise ValueError(
            "duration must be positive"
        )

    if parent_length < duration:
        raise ValueError(
            "parent shorter than requested duration"
        )

    return (
        parent_length
        - duration
    ) // 2


def _stratified_onset(
    *,
    parent_length,
    duration,
    replicate_index,
    seed,
):
    if replicate_index not in (0, 1, 2):
        raise ValueError(
            "replicate_index must be 0, 1, or 2"
        )

    if duration <= 0:
        raise ValueError(
            "duration must be positive"
        )

    eligible_count = (
        parent_length
        - duration
        + 1
    )

    if eligible_count <= 0:
        raise ValueError(
            "parent shorter than requested duration"
        )

    # Split eligible start indices into three deterministic
    # contiguous strata with numpy.array_split.
    strata = np.array_split(
        np.arange(
            eligible_count,
            dtype=np.int64,
        ),
        3,
    )

    stratum = strata[
        replicate_index
    ]

    if len(stratum) == 0:
        raise ValueError(
            "eligible onset space too short for three strata"
        )

    rng = np.random.default_rng(
        seed
    )

    return int(
        rng.choice(
            stratum
        )
    )


def generate_window_instances(
    *,
    fold,
    partition,
    parent_sequence_id,
    severity_protocol,
):
    if partition == "onfield":
        raise ValueError(
            "OnField fault generation is prohibited"
        )

    out = []

    # bias
    for level in LEVELS:
        fraction = severity_protocol[
            "families"
        ][
            "bias"
        ][
            "levels"
        ][
            level
        ]

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            for sign_name, sign in [
                ("neg", -1),
                ("pos", 1),
            ]:
                out.append(
                    _base_instance(
                        family="bias",
                        level=level,
                        variant_id=f"ch{ch}_{sign_name}",
                        replicate_index=0,
                        partition=partition,
                        fold=fold,
                        parent_kind="stored_window",
                        parent_sequence_id=parent_sequence_id,
                        injection_layer=WINDOW_LAYER,
                        target_channels=channels,
                        onset_sample=0,
                        duration_samples=30,
                        persistence="transient",
                        severity={
                            "level":
                                level,

                            "fraction_of_q99_abs":
                                float(
                                    fraction
                                ),

                            "sign":
                                int(
                                    sign
                                ),
                        },
                    )
                )

    # scale factor
    for level in LEVELS:
        delta = severity_protocol[
            "families"
        ][
            "scale_factor"
        ][
            "levels"
        ][
            level
        ]

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            for name, gain in [
                (
                    "lower",
                    1.0 - delta,
                ),
                (
                    "upper",
                    1.0 + delta,
                ),
            ]:
                out.append(
                    _base_instance(
                        family="scale_factor",
                        level=level,
                        variant_id=f"ch{ch}_{name}",
                        replicate_index=0,
                        partition=partition,
                        fold=fold,
                        parent_kind="stored_window",
                        parent_sequence_id=parent_sequence_id,
                        injection_layer=WINDOW_LAYER,
                        target_channels=channels,
                        onset_sample=0,
                        duration_samples=30,
                        persistence="transient",
                        severity={
                            "level":
                                level,

                            "gain_delta":
                                float(
                                    delta
                                ),

                            "gain":
                                float(
                                    gain
                                ),
                        },
                    )
                )

    # noise
    for level in LEVELS:
        fraction = severity_protocol[
            "families"
        ][
            "noise"
        ][
            "levels"
        ][
            level
        ]

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            for replicate in range(3):
                out.append(
                    _base_instance(
                        family="noise",
                        level=level,
                        variant_id=f"ch{ch}",
                        replicate_index=replicate,
                        partition=partition,
                        fold=fold,
                        parent_kind="stored_window",
                        parent_sequence_id=parent_sequence_id,
                        injection_layer=WINDOW_LAYER,
                        target_channels=channels,
                        onset_sample=0,
                        duration_samples=30,
                        persistence="transient",
                        severity={
                            "level":
                                level,

                            "sigma_fraction_of_robust_sigma":
                                float(
                                    fraction
                                ),

                            "distribution":
                                "zero_mean_gaussian",
                        },
                    )
                )

    # clipping
    for level in LEVELS:
        fraction = severity_protocol[
            "families"
        ][
            "clipping_saturation"
        ][
            "levels"
        ][
            level
        ]

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            out.append(
                _base_instance(
                    family="clipping_saturation",
                    level=level,
                    variant_id=f"ch{ch}",
                    replicate_index=0,
                    partition=partition,
                    fold=fold,
                    parent_kind="stored_window",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=WINDOW_LAYER,
                    target_channels=channels,
                    onset_sample=0,
                    duration_samples=30,
                    persistence="transient",
                    severity={
                        "level":
                            level,

                        "threshold_fraction_of_q99_abs":
                            float(
                                fraction
                            ),
                    },
                )
            )

    # axis loss
    axis = severity_protocol[
        "families"
    ][
        "axis_loss"
    ][
        "levels"
    ]

    for level in LEVELS:
        for idx, channels in enumerate(
            axis[
                level
            ][
                "target_variants"
            ]
        ):
            out.append(
                _base_instance(
                    family="axis_loss",
                    level=level,
                    variant_id=f"variant{idx}",
                    replicate_index=0,
                    partition=partition,
                    fold=fold,
                    parent_kind="stored_window",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=WINDOW_LAYER,
                    target_channels=channels,
                    onset_sample=0,
                    duration_samples=30,
                    persistence="transient",
                    severity={
                        "level":
                            level,

                        "replacement":
                            0.0,

                        "channel_count":
                            len(
                                channels
                            ),
                    },
                )
            )

    if len(out) != 153:
        raise AssertionError(
            f"Expected 153 window instances, got {len(out)}"
        )

    return out


def generate_sequence_instances(
    *,
    fold,
    partition,
    parent_sequence_id,
    parent_length,
    severity_protocol,
):
    if partition == "onfield":
        raise ValueError(
            "OnField fault generation is prohibited"
        )

    parent_length = int(
        parent_length
    )

    if parent_length <= 0:
        raise ValueError(
            "parent_length must be positive"
        )

    out = []

    # drift: centered 100-sample episode
    drift_duration = int(
        severity_protocol[
            "families"
        ][
            "drift"
        ][
            "nominal_fault_duration_samples"
        ]
    )

    drift_onset = _centered_onset(
        parent_length,
        drift_duration,
    )

    for level in LEVELS:
        fraction = severity_protocol[
            "families"
        ][
            "drift"
        ][
            "levels"
        ][
            level
        ]

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            for sign_name, sign in [
                ("neg", -1),
                ("pos", 1),
            ]:
                out.append(
                    _base_instance(
                        family="drift",
                        level=level,
                        variant_id=f"ch{ch}_{sign_name}",
                        replicate_index=0,
                        partition=partition,
                        fold=fold,
                        parent_kind="source_trial",
                        parent_sequence_id=parent_sequence_id,
                        injection_layer=SEQUENCE_LAYER,
                        target_channels=channels,
                        onset_sample=drift_onset,
                        duration_samples=drift_duration,
                        persistence="transient",
                        severity={
                            "level":
                                level,

                            "endpoint_fraction_of_q99_abs":
                                float(
                                    fraction
                                ),

                            "sign":
                                int(
                                    sign
                                ),

                            "trajectory":
                                "linear_from_zero_to_endpoint",
                        },
                    )
                )

    # stuck channel: centered severity-dependent episode
    for level in LEVELS:
        duration = int(
            severity_protocol[
                "families"
            ][
                "stuck_channel"
            ][
                "levels"
            ][
                level
            ]
        )

        onset = _centered_onset(
            parent_length,
            duration,
        )

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            out.append(
                _base_instance(
                    family="stuck_channel",
                    level=level,
                    variant_id=f"ch{ch}",
                    replicate_index=0,
                    partition=partition,
                    fold=fold,
                    parent_kind="source_trial",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=SEQUENCE_LAYER,
                    target_channels=channels,
                    onset_sample=onset,
                    duration_samples=duration,
                    persistence="transient",
                    severity={
                        "level":
                            level,

                        "duration_samples":
                            duration,

                        "hold_rule":
                            "last_value_available_at_fault_onset",
                    },
                )
            )

    # dropout: 3 seeded early/middle/late onset replicates
    for level in LEVELS:
        duration = int(
            severity_protocol[
                "families"
            ][
                "dropout"
            ][
                "levels"
            ][
                level
            ]
        )

        for channels in SINGLE_CHANNELS:
            ch = channels[0]

            for replicate in range(3):
                seed = derive_seed(
                    partition=partition,
                    fold=fold,
                    parent_kind="source_trial",
                    parent_sequence_id=parent_sequence_id,
                    family="dropout",
                    severity_level=level,
                    variant_id=f"ch{ch}",
                    replicate_index=replicate,
                )

                onset = _stratified_onset(
                    parent_length=parent_length,
                    duration=duration,
                    replicate_index=replicate,
                    seed=seed,
                )

                out.append(
                    _base_instance(
                        family="dropout",
                        level=level,
                        variant_id=f"ch{ch}",
                        replicate_index=replicate,
                        partition=partition,
                        fold=fold,
                        parent_kind="source_trial",
                        parent_sequence_id=parent_sequence_id,
                        injection_layer=SEQUENCE_LAYER,
                        target_channels=channels,
                        onset_sample=onset,
                        duration_samples=duration,
                        persistence="transient",
                        severity={
                            "level":
                                level,

                            "duration_samples":
                                duration,

                            "replacement":
                                0.0,

                            "onset_stratum":
                                replicate,
                        },
                    )
                )

    # frame loss: all channels, 3 seeded onset strata
    for level in LEVELS:
        duration = int(
            severity_protocol[
                "families"
            ][
                "frame_loss"
            ][
                "levels"
            ][
                level
            ]
        )

        for replicate in range(3):
            seed = derive_seed(
                partition=partition,
                fold=fold,
                parent_kind="source_trial",
                parent_sequence_id=parent_sequence_id,
                family="frame_loss",
                severity_level=level,
                variant_id="all6",
                replicate_index=replicate,
            )

            onset = _stratified_onset(
                parent_length=parent_length,
                duration=duration,
                replicate_index=replicate,
                seed=seed,
            )

            out.append(
                _base_instance(
                    family="frame_loss",
                    level=level,
                    variant_id="all6",
                    replicate_index=replicate,
                    partition=partition,
                    fold=fold,
                    parent_kind="source_trial",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=SEQUENCE_LAYER,
                    target_channels=ALL_EFFECTIVE_CHANNELS,
                    onset_sample=onset,
                    duration_samples=duration,
                    persistence="transient",
                    severity={
                        "level":
                            level,

                        "lost_frame_count":
                            duration,

                        "reconstruction_rule":
                            "causal_forward_fill_last_complete_frame",

                        "onset_stratum":
                            replicate,
                    },
                )
            )

    # jitter: full trial, 3 stochastic realizations
    for level in LEVELS:
        max_ms = float(
            severity_protocol[
                "families"
            ][
                "jitter"
            ][
                "levels"
            ][
                level
            ]
        )

        for replicate in range(3):
            out.append(
                _base_instance(
                    family="jitter",
                    level=level,
                    variant_id="all6",
                    replicate_index=replicate,
                    partition=partition,
                    fold=fold,
                    parent_kind="source_trial",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=SEQUENCE_LAYER,
                    target_channels=ALL_EFFECTIVE_CHANNELS,
                    onset_sample=0,
                    duration_samples=None,
                    persistence="until_end",
                    severity={
                        "level":
                            level,

                        "maximum_absolute_timestamp_displacement_ms":
                            max_ms,

                        "distribution":
                            "independent_uniform_bounded",
                    },
                )
            )

    # delay: full trial
    for level in LEVELS:
        delay = int(
            severity_protocol[
                "families"
            ][
                "delay"
            ][
                "levels"
            ][
                level
            ]
        )

        out.append(
            _base_instance(
                family="delay",
                level=level,
                variant_id="all6",
                replicate_index=0,
                partition=partition,
                fold=fold,
                parent_kind="source_trial",
                parent_sequence_id=parent_sequence_id,
                injection_layer=SEQUENCE_LAYER,
                target_channels=ALL_EFFECTIVE_CHANNELS,
                onset_sample=0,
                duration_samples=None,
                persistence="until_end",
                severity={
                    "level":
                        level,

                    "delay_samples":
                        delay,

                    "delay_ms":
                        delay * 10,

                    "prefix_rule":
                        "repeat_first_available_sample",
                },
            )
        )

    # orientation: full trial, x/y/z variants
    for level in LEVELS:
        degrees = float(
            severity_protocol[
                "families"
            ][
                "orientation"
            ][
                "levels"
            ][
                level
            ]
        )

        for axis in ("x", "y", "z"):
            out.append(
                _base_instance(
                    family="orientation",
                    level=level,
                    variant_id=f"axis_{axis}",
                    replicate_index=0,
                    partition=partition,
                    fold=fold,
                    parent_kind="source_trial",
                    parent_sequence_id=parent_sequence_id,
                    injection_layer=SEQUENCE_LAYER,
                    target_channels=ALL_EFFECTIVE_CHANNELS,
                    onset_sample=0,
                    duration_samples=None,
                    persistence="until_end",
                    severity={
                        "level":
                            level,

                        "rotation_axis":
                            axis,

                        "rotation_degrees":
                            degrees,

                        "euler_channels_used":
                            False,
                    },
                )
            )

    if len(out) != 138:
        raise AssertionError(
            f"Expected 138 sequence instances, got {len(out)}"
        )

    return out
