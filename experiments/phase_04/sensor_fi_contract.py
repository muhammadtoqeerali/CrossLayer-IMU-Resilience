"""Phase-4H sensor fault-injection metadata contract.

This module intentionally does NOT define severity defaults or execute
faults. It validates prospective P0 fault instances and creates stable
replay identifiers.

Severity schedules must be frozen separately before experiments.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

STORED_CHANNELS = (
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "EulerX",
    "EulerY",
    "EulerZ",
)

EFFECTIVE_CHANNEL_INDICES = frozenset(range(6))

WINDOW_LAYER = "stored_filtered_window_before_IMUNormalizer"
SEQUENCE_LAYER = (
    "source_faithful_oriented_trial_before_filtering_windowing"
)

FAULT_LAYERS = {
    "bias": WINDOW_LAYER,
    "drift": SEQUENCE_LAYER,
    "scale_factor": WINDOW_LAYER,
    "noise": WINDOW_LAYER,
    "clipping_saturation": WINDOW_LAYER,
    "stuck_channel": SEQUENCE_LAYER,
    "axis_loss": WINDOW_LAYER,
    "dropout": SEQUENCE_LAYER,
    "frame_loss": SEQUENCE_LAYER,
    "jitter": SEQUENCE_LAYER,
    "delay": SEQUENCE_LAYER,
    "orientation": SEQUENCE_LAYER,
}

PERSISTENCE = {
    "transient",
    "persistent",
    "until_end",
}


def canonical_instance(instance):
    """Return the replay-hash payload, excluding replay_id."""

    payload = deepcopy(instance)
    payload.pop("replay_id", None)

    return payload


def compute_replay_id(instance):
    """Stable SHA256 replay identifier."""

    payload = canonical_instance(instance)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def validate_fault_instance(instance):
    """Validate a prospective Phase-4H P0 fault instance."""

    required = {
        "fault_id",
        "family",
        "domain",
        "evidence_tier",
        "injection_layer",
        "target_channels",
        "onset_sample",
        "duration_samples",
        "persistence",
        "severity",
        "severity_provenance",
        "seed",
        "parent_sequence_id",
        "partition",
        "replay_id",
    }

    missing = sorted(
        required - set(instance)
    )

    if missing:
        raise ValueError(
            f"missing required fields: {missing}"
        )

    family = instance["family"]

    if family not in FAULT_LAYERS:
        raise ValueError(
            f"unknown fault family: {family}"
        )

    if instance["domain"] != "sensor":
        raise ValueError(
            "Phase-4H contract requires domain='sensor'"
        )

    if instance["evidence_tier"] != "P0":
        raise ValueError(
            "Phase-4H software FI requires evidence_tier='P0'"
        )

    expected_layer = FAULT_LAYERS[family]

    if instance["injection_layer"] != expected_layer:
        raise ValueError(
            f"{family} requires injection_layer={expected_layer}"
        )

    channels = instance["target_channels"]

    if (
        not isinstance(channels, list)
        or not channels
    ):
        raise ValueError(
            "target_channels must be a non-empty list"
        )

    if any(
        not isinstance(x, int)
        for x in channels
    ):
        raise ValueError(
            "target_channels must contain integer indices"
        )

    if not set(channels).issubset(
        EFFECTIVE_CHANNEL_INDICES
    ):
        raise ValueError(
            "primary Phase-4H faults may target only effective "
            "Acc/Gyr channels 0..5"
        )

    if len(channels) != len(set(channels)):
        raise ValueError(
            "target_channels must not contain duplicates"
        )

    onset = instance["onset_sample"]

    if (
        not isinstance(onset, int)
        or onset < 0
    ):
        raise ValueError(
            "onset_sample must be an integer >= 0"
        )

    duration = instance["duration_samples"]

    if (
        duration is not None
        and (
            not isinstance(duration, int)
            or duration <= 0
        )
    ):
        raise ValueError(
            "duration_samples must be null or an integer > 0"
        )

    if instance["persistence"] not in PERSISTENCE:
        raise ValueError(
            f"invalid persistence: {instance['persistence']}"
        )

    if (
        instance["persistence"] == "transient"
        and duration is None
    ):
        raise ValueError(
            "transient faults require duration_samples"
        )

    if not isinstance(
        instance["seed"],
        int,
    ):
        raise ValueError(
            "seed is required and must be an integer"
        )

    severity = instance["severity"]

    if not isinstance(
        severity,
        dict,
    ):
        raise ValueError(
            "severity must be an explicit dictionary"
        )

    if not severity:
        raise ValueError(
            "severity must not be empty"
        )

    provenance = instance[
        "severity_provenance"
    ]

    if not isinstance(
        provenance,
        dict,
    ):
        raise ValueError(
            "severity_provenance must be a dictionary"
        )

    if provenance.get(
        "physical_realism_claim"
    ) is not False:
        raise ValueError(
            "P0 severity provenance must explicitly set "
            "physical_realism_claim=False"
        )

    if provenance.get(
        "evidence_tier"
    ) != "P0":
        raise ValueError(
            "severity_provenance.evidence_tier must be P0"
        )

    if not str(
        instance["parent_sequence_id"]
    ).strip():
        raise ValueError(
            "parent_sequence_id must be non-empty"
        )

    if not str(
        instance["partition"]
    ).strip():
        raise ValueError(
            "partition must be non-empty"
        )

    expected_replay = compute_replay_id(
        instance
    )

    if instance["replay_id"] != expected_replay:
        raise ValueError(
            "replay_id does not match canonical instance SHA256"
        )

    return True
