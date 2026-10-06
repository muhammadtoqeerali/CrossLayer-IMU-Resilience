"""Phase-5H shard-style compute-FI sequence runtime v1.

This module contains reusable transient/persistent sequence mechanics only.
It does not select partitions, targets, elements, bits, or thresholds.

Qualification is performed on training-calibration fixtures only.
"""

from __future__ import annotations

from typing import Sequence

import torch

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_execution_harness import (
    PTQWeightFaultSession,
    PairedExecution,
    outputs_bitwise_equal,
    run_fp32_paired,
    run_ptq_activation_buffer_paired,
)


def _validate_sequence(
    inputs: Sequence[torch.Tensor],
    inference_indices: Sequence[int],
) -> None:
    if not inputs:
        raise ValueError(
            "sequence must contain at least one input"
        )

    if len(inputs) != len(inference_indices):
        raise ValueError(
            "input/index sequence length mismatch"
        )

    if len(set(int(x) for x in inference_indices)) != len(inference_indices):
        raise ValueError(
            "inference indices must be unique"
        )

    if list(map(int, inference_indices)) != sorted(
        int(x)
        for x in inference_indices
    ):
        raise ValueError(
            "inference indices must be monotonically increasing"
        )


def run_activation_buffer_sequence(
    *,
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    inference_indices: Sequence[int],
    identities: Sequence[FaultIdentity],
) -> list[PairedExecution]:
    """Execute one transient shard or one persistent shard sequence.

    Transient:
      one FaultIdentity per inference.

    Persistent:
      exactly one FaultIdentity reused from onset through trial end.
    """

    _validate_sequence(
        inputs,
        inference_indices,
    )

    if not identities:
        raise ValueError(
            "identities must not be empty"
        )

    for identity in identities:
        validate_fault_identity(
            identity
        )

    persistence = identities[0].persistence

    if any(
        identity.persistence != persistence
        for identity in identities
    ):
        raise ValueError(
            "mixed persistence modes are prohibited within one shard"
        )

    model_variant = identities[0].model_variant

    if any(
        identity.model_variant != model_variant
        for identity in identities
    ):
        raise ValueError(
            "mixed model variants are prohibited within one shard"
        )

    if persistence == "transient_one_inference":
        if len(identities) != len(inputs):
            raise ValueError(
                "transient shard requires one identity per inference"
            )

    elif persistence == "persistent_from_onset_until_trial_end":
        if len(identities) != 1:
            raise ValueError(
                "persistent shard requires exactly one identity"
            )

    else:
        raise ValueError(
            f"unsupported persistence mode: {persistence}"
        )

    if model_variant == "fp32":
        runner = run_fp32_paired

    elif model_variant == "ptq_v7":
        runner = run_ptq_activation_buffer_paired

    else:
        raise ValueError(
            f"unsupported model variant: {model_variant}"
        )

    pairs = []

    for sequence_index, (
        x,
        current_inference_index,
    ) in enumerate(
        zip(
            inputs,
            inference_indices,
        )
    ):
        identity = (
            identities[
                sequence_index
            ]
            if persistence
            == "transient_one_inference"
            else identities[
                0
            ]
        )

        pair = runner(
            model,
            x,
            identity,
            current_inference_index=int(
                current_inference_index
            ),
        )

        pairs.append(
            pair
        )

    return pairs


def run_ptq_weight_sequence(
    *,
    fault_model: torch.nn.Module,
    clean_model: torch.nn.Module,
    clean_state_dict,
    inputs: Sequence[torch.Tensor],
    inference_indices: Sequence[int],
    identities: Sequence[FaultIdentity],
    reset_probe_input: torch.Tensor,
) -> tuple[
    list[PairedExecution],
    bool,
]:
    """Execute transient/persistent PTQ qint8 weight shard semantics.

    The weight-fault session is reset unconditionally before returning.
    The returned boolean proves post-reset fault-model output equals the clean
    model on `reset_probe_input`.
    """

    _validate_sequence(
        inputs,
        inference_indices,
    )

    if not identities:
        raise ValueError(
            "identities must not be empty"
        )

    for identity in identities:
        validate_fault_identity(
            identity
        )

        if identity.model_variant != "ptq_v7":
            raise ValueError(
                "weight session is PTQ-only"
            )

        if identity.representation_class != "int8_persistent_weight":
            raise ValueError(
                "weight session requires int8_persistent_weight"
            )

    persistence = identities[0].persistence

    if any(
        identity.persistence != persistence
        for identity in identities
    ):
        raise ValueError(
            "mixed persistence modes are prohibited"
        )

    if persistence == "transient_one_inference":
        if len(identities) != len(inputs):
            raise ValueError(
                "transient weight shard requires one identity per inference"
            )

    elif persistence == "persistent_from_onset_until_trial_end":
        if len(identities) != 1:
            raise ValueError(
                "persistent weight shard requires exactly one identity"
            )

    else:
        raise ValueError(
            f"unsupported persistence mode: {persistence}"
        )

    session = PTQWeightFaultSession(
        fault_model,
        clean_state_dict,
    )

    pairs = []

    try:
        for sequence_index, (
            x,
            current_inference_index,
        ) in enumerate(
            zip(
                inputs,
                inference_indices,
            )
        ):
            identity = (
                identities[
                    sequence_index
                ]
                if persistence
                == "transient_one_inference"
                else identities[
                    0
                ]
            )

            pair = session.run_paired(
                clean_model,
                x,
                identity,
                current_inference_index=int(
                    current_inference_index
                ),
            )

            pairs.append(
                pair
            )

    finally:
        session.reset_clean()

    with torch.no_grad():
        reset_output = fault_model(
            reset_probe_input.clone()
        )

        clean_output = clean_model(
            reset_probe_input.clone()
        )

    reset_equal = outputs_bitwise_equal(
        reset_output,
        clean_output,
    )

    return (
        pairs,
        reset_equal,
    )


def active_mask(
    pairs: Sequence[PairedExecution],
) -> list[bool]:
    return [
        bool(
            pair.mutation.active
        )
        for pair in pairs
    ]
