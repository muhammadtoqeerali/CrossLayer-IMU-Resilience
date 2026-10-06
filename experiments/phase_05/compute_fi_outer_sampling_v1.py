"""Frozen Phase-5D prospective outer compute-FI sampling functions.

The functions in this module select software-FI coordinates only.

Model seed and model variant are intentionally excluded from the sampling
identity so the same logical fault coordinates can be reused across checkpoint
seeds and, for common FP32 boundaries, across FP32/PTQ variants.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


NAMESPACE = "crosslayer-phase5d-compute-fi-outer-v1"


def canonical_json(
    value,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def canonical_sampling_payload(
    *,
    partition: str,
    fold: int,
    subject: int,
    task: int,
    trial: int,
    parent_kind: str,
    window_index: int | None,
    representation_class: str,
    target_name: str,
    target_role: str,
    persistence: str,
    replicate_index: int,
) -> dict:
    if partition != "outer_test":
        raise ValueError(
            "Phase-5D frozen sampler is outer_test only"
        )

    if parent_kind not in {
        "window",
        "trial",
    }:
        raise ValueError(
            f"invalid parent_kind: {parent_kind}"
        )

    if persistence not in {
        "transient_one_inference",
        "persistent_from_onset_until_trial_end",
    }:
        raise ValueError(
            f"invalid persistence: {persistence}"
        )

    if (
        parent_kind == "window"
        and window_index is None
    ):
        raise ValueError(
            "window parent requires window_index"
        )

    if (
        parent_kind == "trial"
        and window_index is not None
    ):
        raise ValueError(
            "trial parent must not encode a single window_index"
        )

    return {
        "namespace":
            NAMESPACE,

        "partition":
            partition,

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "task":
            int(
                task
            ),

        "trial":
            int(
                trial
            ),

        "parent_kind":
            parent_kind,

        "window_index":
            (
                None
                if window_index is None
                else int(
                    window_index
                )
            ),

        "representation_class":
            representation_class,

        "target_name":
            target_name,

        "target_role":
            target_role,

        "persistence":
            persistence,

        "replicate_index":
            int(
                replicate_index
            ),
    }


def _digest(
    payload: dict,
    dimension: str,
) -> bytes:
    wrapped = {
        "sampling_payload":
            payload,

        "dimension":
            dimension,
    }

    return hashlib.sha256(
        canonical_json(
            wrapped
        ).encode(
            "utf-8"
        )
    ).digest()


def sampling_instance_id(
    payload: dict,
) -> str:
    return hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _u64(
    payload: dict,
    dimension: str,
) -> int:
    return int.from_bytes(
        _digest(
            payload,
            dimension,
        )[
            :8
        ],
        byteorder="big",
        signed=False,
    )


def derive_element_index(
    payload: dict,
    *,
    target_numel: int,
) -> int:
    target_numel = int(
        target_numel
    )

    if target_numel <= 0:
        raise ValueError(
            "target_numel must be > 0"
        )

    return (
        _u64(
            payload,
            "element_index",
        )
        % target_numel
    )


def derive_bit_position(
    payload: dict,
    *,
    eligible_bits: Iterable[int],
) -> int:
    bits = tuple(
        int(
            bit
        )
        for bit in eligible_bits
    )

    if not bits:
        raise ValueError(
            "eligible_bits must not be empty"
        )

    if len(
        bits
    ) != len(
        set(
            bits
        )
    ):
        raise ValueError(
            "eligible_bits must be unique"
        )

    return bits[
        _u64(
            payload,
            "bit_position",
        )
        % len(
            bits
        )
    ]


def derive_persistent_onset_index(
    payload: dict,
    *,
    trial_window_count: int,
) -> int:
    trial_window_count = int(
        trial_window_count
    )

    if trial_window_count <= 0:
        raise ValueError(
            "trial_window_count must be > 0"
        )

    if (
        payload[
            "persistence"
        ]
        != "persistent_from_onset_until_trial_end"
    ):
        raise ValueError(
            "persistent onset requested for non-persistent payload"
        )

    if payload[
        "parent_kind"
    ] != "trial":
        raise ValueError(
            "persistent onset requires trial parent"
        )

    return (
        _u64(
            payload,
            "persistent_onset_index",
        )
        % trial_window_count
    )


def transient_inference_index(
    payload: dict,
) -> int:
    if (
        payload[
            "persistence"
        ]
        != "transient_one_inference"
    ):
        raise ValueError(
            "transient index requested for non-transient payload"
        )

    if payload[
        "parent_kind"
    ] != "window":
        raise ValueError(
            "transient fault requires window parent"
        )

    return int(
        payload[
            "window_index"
        ]
    )


OUTER_INSTANCE_NAMESPACE = (
    "crosslayer-phase5d-compute-fi-outer-instance-v1"
)


def canonical_outer_instance_payload(
    *,
    sampling_instance_id_value: str,
    phase5a_fault_id: str,
    model_variant: str,
    checkpoint_seed: int,
) -> dict:
    """Build the parent-bound model-execution identity payload.

    `sampling_instance_id_value` already binds the outer parent identity,
    representation, target, persistence and replicate.

    `phase5a_fault_id` remains the frozen mutation-specification identity and
    is intentionally preserved unchanged.
    """

    if not sampling_instance_id_value:
        raise ValueError(
            "sampling_instance_id_value must not be empty"
        )

    if not phase5a_fault_id:
        raise ValueError(
            "phase5a_fault_id must not be empty"
        )

    if model_variant not in {
        "fp32",
        "ptq_v7",
    }:
        raise ValueError(
            f"unsupported model_variant: {model_variant}"
        )

    checkpoint_seed = int(
        checkpoint_seed
    )

    if checkpoint_seed not in {
        42,
        123,
        2025,
    }:
        raise ValueError(
            f"unsupported checkpoint_seed: {checkpoint_seed}"
        )

    return {
        "namespace":
            OUTER_INSTANCE_NAMESPACE,

        "sampling_instance_id":
            sampling_instance_id_value,

        "phase5a_fault_id":
            phase5a_fault_id,

        "model_variant":
            model_variant,

        "checkpoint_seed":
            checkpoint_seed,
    }


def outer_instance_id(
    *,
    sampling_instance_id_value: str,
    phase5a_fault_id: str,
    model_variant: str,
    checkpoint_seed: int,
) -> str:
    """Return the globally parent-bound Phase-5D execution-instance ID."""

    payload = canonical_outer_instance_payload(
        sampling_instance_id_value=sampling_instance_id_value,
        phase5a_fault_id=phase5a_fault_id,
        model_variant=model_variant,
        checkpoint_seed=checkpoint_seed,
    )

    return hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()
