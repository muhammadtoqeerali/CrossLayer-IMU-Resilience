"""Phase-5K fault-only compute-FI execution primitives.

These functions perform exactly one requested fault-path execution and do not
perform a separate clean-reference forward.

They deliberately reuse the already-qualified Phase-5A/B mutation machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_execution_harness import (
    FP32_BOUNDARY_TO_MODULE,
    PTQ_BOUNDARY_TO_NODE,
    MutationRecord,
    PTQFaultInterpreter,
    clone_state_dict,
    inactive_record,
    mutation_record,
    tensor_bytes_sha256,
    validate_exact_single_bit_mutation,
)
from compute_fi_operators import (
    apply_scheduled_bit_fault,
    fault_active,
    flip_qint8_tensor_bit,
)


@dataclass(frozen=True)
class FaultOnlyExecution:
    faulted_output: torch.Tensor
    mutation: MutationRecord
    fault_id: str
    input_sha256: str


def run_fp32_fault_only(
    model: torch.nn.Module,
    x: torch.Tensor,
    identity: FaultIdentity,
    *,
    current_inference_index: int,
) -> FaultOnlyExecution:
    validate_fault_identity(
        identity
    )

    if identity.model_variant != "fp32":
        raise ValueError(
            "FP32 fault-only runner requires model_variant='fp32'"
        )

    if identity.representation_class not in {
        "fp32_activation",
        "fp32_buffer",
    }:
        raise ValueError(
            "FP32 fault-only runner supports FP32 activation/buffer only"
        )

    if identity.target_name not in FP32_BOUNDARY_TO_MODULE:
        raise ValueError(
            f"unknown FP32 target: {identity.target_name}"
        )

    active = fault_active(
        inference_index=current_inference_index,
        onset_index=identity.inference_index,
        persistence=identity.persistence,
    )

    if not active:
        with torch.no_grad():
            faulted_output = model(
                x.clone()
            )

        return FaultOnlyExecution(
            faulted_output=faulted_output,
            mutation=inactive_record(
                identity
            ),
            fault_id=identity.fault_id(),
            input_sha256=tensor_bytes_sha256(
                x
            ),
        )

    target_module = model.get_submodule(
        FP32_BOUNDARY_TO_MODULE[
            identity.target_name
        ]
    )

    holder: dict[str, MutationRecord] = {}

    def hook(
        _module,
        _inputs,
        output,
    ):
        before = output.detach().clone()

        after, metadata = apply_scheduled_bit_fault(
            output,
            inference_index=current_inference_index,
            onset_index=identity.inference_index,
            persistence=identity.persistence,
            representation_class=identity.representation_class,
            element_index=identity.element_index,
            bit_position=identity.bit_position,
        )

        assert metadata.active

        record = mutation_record(
            before,
            after,
            representation_class=identity.representation_class,
            target_name=identity.target_name,
            element_index=identity.element_index,
            bit_position=identity.bit_position,
        )

        validate_exact_single_bit_mutation(
            record
        )

        holder[
            "record"
        ] = record

        return after

    handle = target_module.register_forward_hook(
        hook
    )

    try:
        with torch.no_grad():
            faulted_output = model(
                x.clone()
            )

    finally:
        handle.remove()

    if "record" not in holder:
        raise AssertionError(
            "FP32 fault-only hook did not execute"
        )

    return FaultOnlyExecution(
        faulted_output=faulted_output,
        mutation=holder[
            "record"
        ],
        fault_id=identity.fault_id(),
        input_sha256=tensor_bytes_sha256(
            x
        ),
    )


def run_ptq_activation_buffer_fault_only(
    model: torch.fx.GraphModule,
    x: torch.Tensor,
    identity: FaultIdentity,
    *,
    current_inference_index: int,
) -> FaultOnlyExecution:
    validate_fault_identity(
        identity
    )

    if identity.model_variant != "ptq_v7":
        raise ValueError(
            "PTQ fault-only runner requires model_variant='ptq_v7'"
        )

    if identity.representation_class not in {
        "quantized_activation",
        "quantized_buffer",
        "fp32_activation",
        "fp32_buffer",
    }:
        raise ValueError(
            "PTQ fault-only activation/buffer representation unsupported"
        )

    if identity.target_name not in PTQ_BOUNDARY_TO_NODE:
        raise ValueError(
            f"unknown PTQ target: {identity.target_name}"
        )

    interpreter = PTQFaultInterpreter(
        model,
        identity,
        current_inference_index=current_inference_index,
    )

    with torch.no_grad():
        faulted_output = interpreter.run(
            x.clone()
        )

    return FaultOnlyExecution(
        faulted_output=faulted_output,
        mutation=interpreter.record,
        fault_id=identity.fault_id(),
        input_sha256=tensor_bytes_sha256(
            x
        ),
    )


class PTQWeightFaultOnlySession:
    """Private PTQ state for one-forward qint8 weight FI execution."""

    def __init__(
        self,
        fault_model: torch.fx.GraphModule,
        clean_state: dict[str, Any],
    ):
        self.model = fault_model

        self.clean_state = clone_state_dict(
            clean_state
        )

        if "conv_2.0.weight" not in self.clean_state:
            raise KeyError(
                "frozen PTQ state missing conv_2.0.weight"
            )

    def run_fault_only(
        self,
        x: torch.Tensor,
        identity: FaultIdentity,
        *,
        current_inference_index: int,
    ) -> FaultOnlyExecution:
        validate_fault_identity(
            identity
        )

        if identity.model_variant != "ptq_v7":
            raise ValueError(
                "PTQ weight fault-only session requires ptq_v7"
            )

        if identity.representation_class != "int8_persistent_weight":
            raise ValueError(
                "PTQ weight fault-only session requires int8_persistent_weight"
            )

        if identity.target_name != "conv_2.0.weight":
            raise ValueError(
                "Phase-5A v1 qint8 weight target must be conv_2.0.weight"
            )

        active = fault_active(
            inference_index=current_inference_index,
            onset_index=identity.inference_index,
            persistence=identity.persistence,
        )

        state = clone_state_dict(
            self.clean_state
        )

        if active:
            before = state[
                "conv_2.0.weight"
            ].clone()

            after = flip_qint8_tensor_bit(
                before,
                element_index=identity.element_index,
                bit_position=identity.bit_position,
            )

            state[
                "conv_2.0.weight"
            ] = after

            record = mutation_record(
                before,
                after,
                representation_class=identity.representation_class,
                target_name=identity.target_name,
                element_index=identity.element_index,
                bit_position=identity.bit_position,
            )

            validate_exact_single_bit_mutation(
                record
            )

        else:
            record = inactive_record(
                identity
            )

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.model.eval()

        with torch.no_grad():
            faulted_output = self.model(
                x.clone()
            )

        return FaultOnlyExecution(
            faulted_output=faulted_output,
            mutation=record,
            fault_id=identity.fault_id(),
            input_sha256=tensor_bytes_sha256(
                x
            ),
        )

    def reset_clean(
        self,
    ) -> None:
        self.model.load_state_dict(
            clone_state_dict(
                self.clean_state
            ),
            strict=True,
        )

        self.model.eval()
