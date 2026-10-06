"""Bit-exact P0 compute-fault operators for Phase 5A.

These functions mutate copies of tensors only.

They do not:
- run model inference,
- select data,
- infer physical fault rates,
- model MCU/register/cache placement.

The implementation is intentionally CPU-only for v1 because the scientific
contract requires exact payload-level replay and qualification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compute_fi_contract import (
    FP32_BIT_POSITIONS,
    INT8_BIT_POSITIONS,
    PERSISTENCE_MODES,
)


QUANTIZED_REPRESENTATIONS = {
    "int8_persistent_weight",
    "quantized_activation",
    "quantized_buffer",
}

FP32_REPRESENTATIONS = {
    "fp32_activation",
    "fp32_buffer",
}


@dataclass(frozen=True)
class AppliedFault:
    active: bool
    inference_index: int
    onset_index: int
    persistence: str
    representation_class: str
    element_index: int
    bit_position: int


def _validate_element_index(
    *,
    element_index: int,
    numel: int,
) -> None:
    if element_index < 0 or element_index >= numel:
        raise IndexError(
            f"element_index={element_index} outside [0,{numel})"
        )


def _require_cpu(
    tensor: torch.Tensor,
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(
            "Phase-5A v1 bit-exact operators require CPU tensors"
        )


def int8_payload_to_uint8(
    value: int,
) -> int:
    if value < -128 or value > 127:
        raise ValueError(
            "signed INT8 payload must be in [-128,127]"
        )

    return value & 0xFF


def uint8_payload_to_int8(
    value: int,
) -> int:
    if value < 0 or value > 255:
        raise ValueError(
            "unsigned INT8 payload must be in [0,255]"
        )

    return value - 256 if value >= 128 else value


def flip_int8_payload(
    value: int,
    bit_position: int,
) -> int:
    if bit_position not in INT8_BIT_POSITIONS:
        raise ValueError(
            f"invalid INT8 bit position: {bit_position}"
        )

    payload = int8_payload_to_uint8(
        value
    )

    payload ^= 1 << bit_position

    return uint8_payload_to_int8(
        payload
    )


def fp32_payload_bits(
    value: np.float32,
) -> int:
    arr = np.asarray(
        [value],
        dtype=np.float32,
    )

    return int(
        arr.view(
            np.uint32
        )[0]
    )


def fp32_from_payload_bits(
    bits: int,
) -> np.float32:
    if bits < 0 or bits > 0xFFFFFFFF:
        raise ValueError(
            "FP32 payload must be a uint32 value"
        )

    arr = np.asarray(
        [bits],
        dtype=np.uint32,
    )

    return arr.view(
        np.float32
    )[0]


def flip_fp32_scalar(
    value: np.float32,
    bit_position: int,
) -> np.float32:
    if bit_position not in FP32_BIT_POSITIONS:
        raise ValueError(
            f"invalid FP32 bit position: {bit_position}"
        )

    bits = fp32_payload_bits(
        np.float32(
            value
        )
    )

    bits ^= 1 << bit_position

    return fp32_from_payload_bits(
        bits
    )


def flip_fp32_tensor_bit(
    tensor: torch.Tensor,
    *,
    element_index: int,
    bit_position: int,
) -> torch.Tensor:
    _require_cpu(
        tensor
    )

    if tensor.dtype != torch.float32:
        raise TypeError(
            "FP32 operator requires torch.float32"
        )

    if bit_position not in FP32_BIT_POSITIONS:
        raise ValueError(
            f"invalid FP32 bit position: {bit_position}"
        )

    _validate_element_index(
        element_index=element_index,
        numel=tensor.numel(),
    )

    arr = (
        tensor
        .detach()
        .contiguous()
        .numpy()
        .copy()
    )

    payload = arr.view(
        np.uint32
    ).reshape(
        -1
    )

    payload[
        element_index
    ] ^= np.uint32(
        1 << bit_position
    )

    out = torch.from_numpy(
        arr
    ).reshape(
        tensor.shape
    )

    assert out.dtype == torch.float32

    return out


def _rebuild_quantized_tensor(
    original: torch.Tensor,
    integer_payload: torch.Tensor,
) -> torch.Tensor:
    qscheme = original.qscheme()

    if qscheme in (
        torch.per_channel_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine_float_qparams,
    ):
        return torch._make_per_channel_quantized_tensor(
            integer_payload,
            original.q_per_channel_scales().clone(),
            original.q_per_channel_zero_points().clone(),
            original.q_per_channel_axis(),
        )

    if qscheme in (
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ):
        return torch._make_per_tensor_quantized_tensor(
            integer_payload,
            original.q_scale(),
            original.q_zero_point(),
        )

    raise ValueError(
        f"unsupported quantization scheme: {qscheme}"
    )


def flip_quantized_tensor_bit(
    tensor: torch.Tensor,
    *,
    element_index: int,
    bit_position: int,
) -> torch.Tensor:
    """Flip one payload bit in qint8 or quint8 quantized tensors.

    The original quantization metadata and source tensor are preserved.
    """

    _require_cpu(
        tensor
    )

    if not tensor.is_quantized:
        raise TypeError(
            "quantized operator requires a quantized tensor"
        )

    if tensor.dtype not in (
        torch.qint8,
        torch.quint8,
    ):
        raise TypeError(
            "Phase-5A v1 supports torch.qint8 and torch.quint8 payloads only"
        )

    if bit_position not in INT8_BIT_POSITIONS:
        raise ValueError(
            f"invalid 8-bit payload bit position: {bit_position}"
        )

    _validate_element_index(
        element_index=element_index,
        numel=tensor.numel(),
    )

    integer_repr = (
        tensor
        .int_repr()
        .contiguous()
    )

    expected_integer_dtype = (
        torch.int8
        if tensor.dtype == torch.qint8
        else torch.uint8
    )

    if integer_repr.dtype != expected_integer_dtype:
        raise TypeError(
            "unexpected integer payload dtype: "
            f"{integer_repr.dtype} for {tensor.dtype}"
        )

    integer_np = (
        integer_repr
        .numpy()
        .copy()
    )

    # Viewing both signed int8 and unsigned uint8 payloads as uint8 makes
    # the XOR operation representation-exact without numerical conversion.
    payload_u8 = integer_np.view(
        np.uint8
    ).reshape(
        -1
    )

    payload_u8[
        element_index
    ] ^= np.uint8(
        1 << bit_position
    )

    integer_payload = torch.from_numpy(
        integer_np
    ).reshape(
        integer_repr.shape
    )

    assert integer_payload.dtype == expected_integer_dtype

    out = _rebuild_quantized_tensor(
        tensor,
        integer_payload,
    )

    assert out.dtype == tensor.dtype
    assert out.int_repr().dtype == expected_integer_dtype

    return out


def flip_qint8_tensor_bit(
    tensor: torch.Tensor,
    *,
    element_index: int,
    bit_position: int,
) -> torch.Tensor:
    """Weight-specific qint8 wrapper retained for frozen v1 semantics."""

    if tensor.dtype != torch.qint8:
        raise TypeError(
            "qint8 weight operator requires torch.qint8"
        )

    return flip_quantized_tensor_bit(
        tensor,
        element_index=element_index,
        bit_position=bit_position,
    )


def flip_quint8_tensor_bit(
    tensor: torch.Tensor,
    *,
    element_index: int,
    bit_position: int,
) -> torch.Tensor:
    """Activation/buffer-specific quint8 wrapper."""

    if tensor.dtype != torch.quint8:
        raise TypeError(
            "quint8 activation/buffer operator requires torch.quint8"
        )

    return flip_quantized_tensor_bit(
        tensor,
        element_index=element_index,
        bit_position=bit_position,
    )


def fault_active(
    *,
    inference_index: int,
    onset_index: int,
    persistence: str,
) -> bool:
    if inference_index < 0:
        raise ValueError(
            "inference_index must be non-negative"
        )

    if onset_index < 0:
        raise ValueError(
            "onset_index must be non-negative"
        )

    if persistence not in PERSISTENCE_MODES:
        raise ValueError(
            f"unsupported persistence: {persistence}"
        )

    if persistence == "transient_one_inference":
        return inference_index == onset_index

    if persistence == "persistent_from_onset_until_trial_end":
        return inference_index >= onset_index

    raise AssertionError(
        "unreachable persistence branch"
    )


def apply_scheduled_bit_fault(
    tensor: torch.Tensor,
    *,
    inference_index: int,
    onset_index: int,
    persistence: str,
    representation_class: str,
    element_index: int,
    bit_position: int,
) -> tuple[torch.Tensor, AppliedFault]:
    active = fault_active(
        inference_index=inference_index,
        onset_index=onset_index,
        persistence=persistence,
    )

    if not active:
        return (
            tensor.clone(),
            AppliedFault(
                active=False,
                inference_index=inference_index,
                onset_index=onset_index,
                persistence=persistence,
                representation_class=representation_class,
                element_index=element_index,
                bit_position=bit_position,
            ),
        )

    if representation_class == "int8_persistent_weight":
        out = flip_qint8_tensor_bit(
            tensor,
            element_index=element_index,
            bit_position=bit_position,
        )

    elif representation_class in {
        "quantized_activation",
        "quantized_buffer",
    }:
        out = flip_quantized_tensor_bit(
            tensor,
            element_index=element_index,
            bit_position=bit_position,
        )

    elif representation_class in FP32_REPRESENTATIONS:
        out = flip_fp32_tensor_bit(
            tensor,
            element_index=element_index,
            bit_position=bit_position,
        )

    else:
        raise ValueError(
            "unsupported representation_class: "
            f"{representation_class}"
        )

    return (
        out,
        AppliedFault(
            active=True,
            inference_index=inference_index,
            onset_index=onset_index,
            persistence=persistence,
            representation_class=representation_class,
            element_index=element_index,
            bit_position=bit_position,
        ),
    )


def fp32_payload_equal(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    _require_cpu(
        left
    )

    _require_cpu(
        right
    )

    if (
        left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.shape != right.shape
    ):
        return False

    return (
        left.detach().contiguous().numpy().tobytes()
        == right.detach().contiguous().numpy().tobytes()
    )


def quantized_payload_equal(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    _require_cpu(
        left
    )

    _require_cpu(
        right
    )

    if (
        not left.is_quantized
        or not right.is_quantized
        or left.dtype != right.dtype
        or left.shape != right.shape
        or left.qscheme() != right.qscheme()
    ):
        return False

    if not torch.equal(
        left.int_repr(),
        right.int_repr(),
    ):
        return False

    if left.qscheme() in (
        torch.per_channel_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine_float_qparams,
    ):
        return (
            left.q_per_channel_axis()
            == right.q_per_channel_axis()
            and torch.equal(
                left.q_per_channel_scales(),
                right.q_per_channel_scales(),
            )
            and torch.equal(
                left.q_per_channel_zero_points(),
                right.q_per_channel_zero_points(),
            )
        )

    return (
        left.q_scale() == right.q_scale()
        and left.q_zero_point() == right.q_zero_point()
    )
