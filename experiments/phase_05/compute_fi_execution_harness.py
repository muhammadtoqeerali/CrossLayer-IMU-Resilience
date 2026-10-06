"""Phase-5B paired clean/faulted synthetic execution harness.

Evidence tier: P0 software FI.

Only synthetic inputs are required by this module's qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.fx import Interpreter
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    prepare_fx,
)

from models.CNN import CNN
from ptq_v7_fc1_fp32 import (
    build_v7_candidate,
    build_v7_prepare_custom_config,
    build_v7_qconfig_mapping,
)

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_operators import (
    apply_scheduled_bit_fault,
    fault_active,
    flip_qint8_tensor_bit,
    fp32_payload_equal,
    quantized_payload_equal,
)


FP32_BOUNDARY_TO_MODULE = {
    "front_end_output_fp32":
        "conv_1.4",

    # The PTQ conv2 includes folded BatchNorm.  The FP32 analogue is the
    # output after native conv2 BatchNorm.
    "conv2_dequantized_output_fp32":
        "conv_2.1",

    "post_conv2_prelu_fp32":
        "conv_2.2",

    "post_conv2_pool_fp32":
        "conv_2.3",

    "post_conv2_dropout_fp32":
        "conv_2.4",

    "flatten_buffer_fp32":
        "fc.0",

    "fc1_output_fp32":
        "fc.1",

    "fc2_prelu_output_fp32":
        "fc.2",

    # FP32 counterpart of the PTQ dequantized buffer before classifier.
    "post_fc2_dequantized_buffer_fp32":
        "fc.3",

    "classifier_output_fp32":
        "fc.4",
}


PTQ_BOUNDARY_TO_NODE = {
    "front_end_output_fp32":
        "front_end",

    "conv2_quantized_input":
        "quantize_per_tensor",

    "conv2_quantized_output":
        "conv_2_0",

    "conv2_dequantized_output_fp32":
        "dequantize_1",

    "post_conv2_prelu_fp32":
        "conv_2_2",

    "post_conv2_pool_fp32":
        "conv_2_3",

    "post_conv2_dropout_fp32":
        "conv_2_4",

    "flatten_buffer_fp32":
        "fc_0",

    "fc1_output_fp32":
        "fc_1",

    "fc2_prelu_output_fp32":
        "fc_2",

    "post_fc2_quantized_buffer":
        "quantize_per_tensor_2",

    "post_fc2_dequantized_buffer_fp32":
        "dequantize_3",

    "classifier_output_fp32":
        "fc_4",
}


@dataclass(frozen=True)
class MutationRecord:
    active: bool
    representation_class: str
    target_name: str
    element_index: int
    bit_position: int
    before_payload: int | None
    after_payload: int | None
    changed_indices: tuple[int, ...]
    tensor_dtype: str | None
    integer_payload_dtype: str | None
    quantization_metadata_preserved: bool | None


@dataclass(frozen=True)
class PairedExecution:
    clean_output: torch.Tensor
    faulted_output: torch.Tensor
    mutation: MutationRecord
    fault_id: str
    input_sha256: str


def sha256_file(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def tensor_bytes_sha256(
    tensor: torch.Tensor,
) -> str:
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    if tensor.is_quantized:
        payload = tensor.int_repr().contiguous().numpy().tobytes()

    else:
        payload = tensor.detach().contiguous().numpy().tobytes()

    return hashlib.sha256(
        payload
    ).hexdigest()


def clone_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    out = {}

    for key, value in state.items():
        if torch.is_tensor(
            value
        ):
            out[key] = value.clone()

        else:
            out[key] = copy.deepcopy(
                value
            )

    return out


def load_frozen_fp32_model(
    freeze_manifest: str | Path,
    *,
    seed: int,
    fold: int,
) -> tuple[CNN, dict[str, Any]]:
    manifest = json.loads(
        Path(
            freeze_manifest
        ).read_text()
    )

    row = next(
        x
        for x in manifest[
            "checkpoints"
        ]
        if x["seed"] == seed
        and x["fold"] == fold
    )

    checkpoint = Path(
        row[
            "checkpoint"
        ]
    )

    assert checkpoint.is_file()

    assert sha256_file(
        checkpoint
    ) == row[
        "sha256"
    ]

    obj = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    hp = obj[
        "hyper_parameters"
    ]

    model = CNN(
        n_features=int(
            hp[
                "n_features"
            ]
        ),
        n_classes=int(
            hp[
                "n_classes"
            ]
        ),
        config=hp[
            "cfg"
        ],
    )

    state = {
        key[len("model."):]:
            value
        for key, value in obj[
            "state_dict"
        ].items()
        if key.startswith(
            "model."
        )
    }

    result = model.load_state_dict(
        state,
        strict=True,
    )

    assert not result.missing_keys
    assert not result.unexpected_keys

    model.eval()

    metadata = {
        "checkpoint":
            str(
                checkpoint
            ),

        "checkpoint_sha256":
            row[
                "sha256"
            ],

        "seed":
            seed,

        "fold":
            fold,
    }

    return model, metadata


def build_frozen_ptq_eager(
    fp32_model: CNN,
    ptq_state_path: str | Path,
) -> torch.fx.GraphModule:
    candidate = build_v7_candidate(
        fp32_model
    )

    candidate.eval()

    prepared = prepare_fx(
        candidate,
        build_v7_qconfig_mapping(),
        (
            torch.zeros(
                1,
                30,
                9,
                dtype=torch.float32,
            ),
        ),
        prepare_custom_config=build_v7_prepare_custom_config(),
    )

    prepared.eval()

    converted = convert_fx(
        prepared
    )

    frozen_state = torch.load(
        ptq_state_path,
        map_location="cpu",
        weights_only=False,
    )

    result = converted.load_state_dict(
        frozen_state,
        strict=True,
    )

    assert not result.missing_keys
    assert not result.unexpected_keys

    converted.eval()

    return converted


def mutation_record(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    representation_class: str,
    target_name: str,
    element_index: int,
    bit_position: int,
) -> MutationRecord:
    if before.shape != after.shape:
        raise AssertionError(
            "fault operator changed tensor shape"
        )

    if before.is_quantized:
        assert after.is_quantized
        assert before.dtype == after.dtype
        assert before.qscheme() == after.qscheme()

        before_payload = (
            before
            .int_repr()
            .contiguous()
            .reshape(
                -1
            )
        )

        after_payload = (
            after
            .int_repr()
            .contiguous()
            .reshape(
                -1
            )
        )

        changed = (
            before_payload
            != after_payload
        ).nonzero(
            as_tuple=False
        ).reshape(
            -1
        ).tolist()

        metadata_preserved = True

        if before.qscheme() in (
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ):
            metadata_preserved = (
                before.q_per_channel_axis()
                == after.q_per_channel_axis()
                and torch.equal(
                    before.q_per_channel_scales(),
                    after.q_per_channel_scales(),
                )
                and torch.equal(
                    before.q_per_channel_zero_points(),
                    after.q_per_channel_zero_points(),
                )
            )

        else:
            metadata_preserved = (
                before.q_scale()
                == after.q_scale()
                and before.q_zero_point()
                == after.q_zero_point()
            )

        return MutationRecord(
            active=True,
            representation_class=representation_class,
            target_name=target_name,
            element_index=element_index,
            bit_position=bit_position,
            before_payload=int(
                before_payload[
                    element_index
                ].item()
            ),
            after_payload=int(
                after_payload[
                    element_index
                ].item()
            ),
            changed_indices=tuple(
                int(x)
                for x in changed
            ),
            tensor_dtype=str(
                before.dtype
            ),
            integer_payload_dtype=str(
                before_payload.dtype
            ),
            quantization_metadata_preserved=metadata_preserved,
        )

    assert before.dtype == torch.float32
    assert after.dtype == torch.float32

    before_payload = (
        before
        .detach()
        .contiguous()
        .numpy()
        .view(
            np.uint32
        )
        .reshape(
            -1
        )
    )

    after_payload = (
        after
        .detach()
        .contiguous()
        .numpy()
        .view(
            np.uint32
        )
        .reshape(
            -1
        )
    )

    changed = np.flatnonzero(
        before_payload
        != after_payload
    ).tolist()

    return MutationRecord(
        active=True,
        representation_class=representation_class,
        target_name=target_name,
        element_index=element_index,
        bit_position=bit_position,
        before_payload=int(
            before_payload[
                element_index
            ]
        ),
        after_payload=int(
            after_payload[
                element_index
            ]
        ),
        changed_indices=tuple(
            int(x)
            for x in changed
        ),
        tensor_dtype=str(
            before.dtype
        ),
        integer_payload_dtype="numpy.uint32",
        quantization_metadata_preserved=None,
    )


def inactive_record(
    identity: FaultIdentity,
) -> MutationRecord:
    return MutationRecord(
        active=False,
        representation_class=identity.representation_class,
        target_name=identity.target_name,
        element_index=identity.element_index,
        bit_position=identity.bit_position,
        before_payload=None,
        after_payload=None,
        changed_indices=(),
        tensor_dtype=None,
        integer_payload_dtype=None,
        quantization_metadata_preserved=None,
    )


def validate_exact_single_bit_mutation(
    record: MutationRecord,
) -> None:
    if not record.active:
        return

    if record.changed_indices != (
        record.element_index,
    ):
        raise AssertionError(
            "mutation changed an unexpected element set: "
            f"{record.changed_indices}"
        )

    assert record.before_payload is not None
    assert record.after_payload is not None

    expected = (
        record.before_payload
        ^ (
            1
            << record.bit_position
        )
    )

    # Quantized uint8 values are directly 0..255. Signed qint8 values are
    # represented as two-complement, so compare their low 8 bits.
    if (
        record.tensor_dtype
        in (
            "torch.qint8",
            "torch.quint8",
        )
    ):
        assert (
            record.after_payload
            & 0xFF
        ) == (
            expected
            & 0xFF
        )

    else:
        assert record.after_payload == expected


def run_fp32_paired(
    model: CNN,
    x: torch.Tensor,
    identity: FaultIdentity,
    *,
    current_inference_index: int,
) -> PairedExecution:
    validate_fault_identity(
        identity
    )

    if identity.model_variant != "fp32":
        raise ValueError(
            "FP32 runner requires model_variant='fp32'"
        )

    if identity.representation_class not in {
        "fp32_activation",
        "fp32_buffer",
    }:
        raise ValueError(
            "FP32 runner supports FP32 activation/buffer faults only"
        )

    if identity.target_name not in FP32_BOUNDARY_TO_MODULE:
        raise ValueError(
            f"unknown FP32 target: {identity.target_name}"
        )

    with torch.no_grad():
        clean_output = model(
            x.clone()
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

        return PairedExecution(
            clean_output=clean_output,
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
            "FP32 fault hook did not execute"
        )

    return PairedExecution(
        clean_output=clean_output,
        faulted_output=faulted_output,
        mutation=holder[
            "record"
        ],
        fault_id=identity.fault_id(),
        input_sha256=tensor_bytes_sha256(
            x
        ),
    )


class PTQFaultInterpreter(
    Interpreter,
):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        identity: FaultIdentity,
        *,
        current_inference_index: int,
    ):
        super().__init__(
            module
        )

        self.identity = identity
        self.current_inference_index = current_inference_index
        self.target_node = PTQ_BOUNDARY_TO_NODE[
            identity.target_name
        ]
        self.record = inactive_record(
            identity
        )

    def run_node(
        self,
        node,
    ):
        result = super().run_node(
            node
        )

        if node.name != self.target_node:
            return result

        active = fault_active(
            inference_index=self.current_inference_index,
            onset_index=self.identity.inference_index,
            persistence=self.identity.persistence,
        )

        if not active:
            return result

        if not torch.is_tensor(
            result
        ):
            raise TypeError(
                "target FX node did not produce a tensor"
            )

        before = result.detach().clone()

        after, metadata = apply_scheduled_bit_fault(
            result,
            inference_index=self.current_inference_index,
            onset_index=self.identity.inference_index,
            persistence=self.identity.persistence,
            representation_class=self.identity.representation_class,
            element_index=self.identity.element_index,
            bit_position=self.identity.bit_position,
        )

        assert metadata.active

        self.record = mutation_record(
            before,
            after,
            representation_class=self.identity.representation_class,
            target_name=self.identity.target_name,
            element_index=self.identity.element_index,
            bit_position=self.identity.bit_position,
        )

        validate_exact_single_bit_mutation(
            self.record
        )

        return after


def run_ptq_activation_buffer_paired(
    model: torch.fx.GraphModule,
    x: torch.Tensor,
    identity: FaultIdentity,
    *,
    current_inference_index: int,
) -> PairedExecution:
    validate_fault_identity(
        identity
    )

    if identity.model_variant != "ptq_v7":
        raise ValueError(
            "PTQ runner requires model_variant='ptq_v7'"
        )

    if identity.representation_class not in {
        "quantized_activation",
        "quantized_buffer",
        "fp32_activation",
        "fp32_buffer",
    }:
        raise ValueError(
            "PTQ activation/buffer runner received unsupported representation"
        )

    if identity.target_name not in PTQ_BOUNDARY_TO_NODE:
        raise ValueError(
            f"unknown PTQ target: {identity.target_name}"
        )

    with torch.no_grad():
        clean_output = model(
            x.clone()
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

    return PairedExecution(
        clean_output=clean_output,
        faulted_output=faulted_output,
        mutation=interpreter.record,
        fault_id=identity.fault_id(),
        input_sha256=tensor_bytes_sha256(
            x
        ),
    )


class PTQWeightFaultSession:
    """Private PTQ model state for transient/persistent qint8 weight FI."""

    def __init__(
        self,
        clean_model: torch.fx.GraphModule,
        clean_state: dict[str, Any],
    ):
        self.model = clean_model
        self.clean_state = clone_state_dict(
            clean_state
        )

        if "conv_2.0.weight" not in self.clean_state:
            raise KeyError(
                "frozen PTQ state missing conv_2.0.weight"
            )

    def run_paired(
        self,
        clean_reference_model: torch.fx.GraphModule,
        x: torch.Tensor,
        identity: FaultIdentity,
        *,
        current_inference_index: int,
    ) -> PairedExecution:
        validate_fault_identity(
            identity
        )

        if identity.model_variant != "ptq_v7":
            raise ValueError(
                "PTQ weight session requires model_variant='ptq_v7'"
            )

        if identity.representation_class != "int8_persistent_weight":
            raise ValueError(
                "PTQ weight session requires int8_persistent_weight"
            )

        if identity.target_name != "conv_2.0.weight":
            raise ValueError(
                "Phase-5A v1 weight target must be conv_2.0.weight"
            )

        with torch.no_grad():
            clean_output = clean_reference_model(
                x.clone()
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

        return PairedExecution(
            clean_output=clean_output,
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


def outputs_bitwise_equal(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and tensor_bytes_sha256(
            left
        )
        == tensor_bytes_sha256(
            right
        )
    )
