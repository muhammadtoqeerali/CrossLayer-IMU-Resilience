from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_operators import (  # noqa: E402
    apply_scheduled_bit_fault,
    flip_fp32_scalar,
    flip_fp32_tensor_bit,
    flip_int8_payload,
    flip_qint8_tensor_bit,
    flip_quantized_tensor_bit,
    flip_quint8_tensor_bit,
    fp32_from_payload_bits,
    fp32_payload_bits,
    fp32_payload_equal,
    int8_payload_to_uint8,
    quantized_payload_equal,
    uint8_payload_to_int8,
)


def qtensor():
    payload = torch.tensor(
        [
            [-128, -1, 0, 1],
            [7, 63, 126, 127],
        ],
        dtype=torch.int8,
    )

    scales = torch.tensor(
        [
            0.125,
            0.25,
        ],
        dtype=torch.float64,
    )

    zero_points = torch.tensor(
        [
            0,
            0,
        ],
        dtype=torch.int64,
    )

    return torch._make_per_channel_quantized_tensor(
        payload,
        scales,
        zero_points,
        0,
    )



def quint8_tensor():
    payload = torch.tensor(
        [
            [0, 1, 63, 127],
            [128, 129, 200, 255],
        ],
        dtype=torch.uint8,
    )

    return torch._make_per_tensor_quantized_tensor(
        payload,
        0.125,
        128,
    )


def test_quint8_mutates_exactly_one_payload():
    q = quint8_tensor()

    before = q.int_repr().reshape(-1).clone()

    mutated = flip_quint8_tensor_bit(
        q,
        element_index=6,
        bit_position=7,
    )

    after = mutated.int_repr().reshape(-1)

    changed = (
        before != after
    ).nonzero(
        as_tuple=False
    ).reshape(
        -1
    )

    assert changed.tolist() == [6]

    assert int(
        after[6].item()
    ) == (
        int(
            before[6].item()
        )
        ^ (
            1 << 7
        )
    )


def test_quint8_preserves_quantization_metadata():
    q = quint8_tensor()

    mutated = flip_quantized_tensor_bit(
        q,
        element_index=2,
        bit_position=4,
    )

    assert mutated.dtype == torch.quint8
    assert mutated.int_repr().dtype == torch.uint8
    assert mutated.qscheme() == q.qscheme()
    assert mutated.q_scale() == q.q_scale()
    assert mutated.q_zero_point() == q.q_zero_point()


def test_quint8_double_flip_restores_exactly():
    q = quint8_tensor()

    once = flip_quint8_tensor_bit(
        q,
        element_index=5,
        bit_position=6,
    )

    twice = flip_quint8_tensor_bit(
        once,
        element_index=5,
        bit_position=6,
    )

    assert quantized_payload_equal(
        q,
        twice,
    )


def test_quantized_activation_schedule_accepts_quint8():
    q = quint8_tensor()

    active = []

    for i in range(
        5
    ):
        out, meta = apply_scheduled_bit_fault(
            q,
            inference_index=i,
            onset_index=2,
            persistence="transient_one_inference",
            representation_class="quantized_activation",
            element_index=0,
            bit_position=7,
        )

        active.append(
            meta.active
        )

        if i == 2:
            assert not quantized_payload_equal(
                q,
                out,
            )
        else:
            assert quantized_payload_equal(
                q,
                out,
            )

    assert active == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_weight_wrapper_rejects_quint8():
    with pytest.raises(TypeError):
        flip_qint8_tensor_bit(
            quint8_tensor(),
            element_index=0,
            bit_position=0,
        )


def test_signed_unsigned_int8_roundtrip_exhaustive():
    for x in range(
        -128,
        128,
    ):
        assert uint8_payload_to_int8(
            int8_payload_to_uint8(
                x
            )
        ) == x


def test_int8_single_bit_flip_exhaustive():
    for value in range(
        -128,
        128,
    ):
        for bit in range(
            8
        ):
            observed = flip_int8_payload(
                value,
                bit,
            )

            expected_u8 = (
                (value & 0xFF)
                ^ (1 << bit)
            )

            expected = (
                expected_u8 - 256
                if expected_u8 >= 128
                else expected_u8
            )

            assert observed == expected

            assert flip_int8_payload(
                observed,
                bit,
            ) == value


def test_qint8_mutates_exactly_one_payload():
    q = qtensor()

    before = q.int_repr().reshape(-1).clone()

    mutated = flip_qint8_tensor_bit(
        q,
        element_index=3,
        bit_position=7,
    )

    after = mutated.int_repr().reshape(-1)

    changed = (
        before != after
    ).nonzero(
        as_tuple=False
    ).reshape(
        -1
    )

    assert changed.tolist() == [3]


def test_qint8_preserves_quantization_metadata():
    q = qtensor()

    mutated = flip_qint8_tensor_bit(
        q,
        element_index=2,
        bit_position=4,
    )

    assert (
        mutated.qscheme()
        == q.qscheme()
    )

    assert (
        mutated.q_per_channel_axis()
        == q.q_per_channel_axis()
    )

    assert torch.equal(
        mutated.q_per_channel_scales(),
        q.q_per_channel_scales(),
    )

    assert torch.equal(
        mutated.q_per_channel_zero_points(),
        q.q_per_channel_zero_points(),
    )


def test_qint8_double_flip_restores_exactly():
    q = qtensor()

    once = flip_qint8_tensor_bit(
        q,
        element_index=5,
        bit_position=6,
    )

    twice = flip_qint8_tensor_bit(
        once,
        element_index=5,
        bit_position=6,
    )

    assert quantized_payload_equal(
        q,
        twice,
    )


def test_qint8_source_is_unchanged():
    q = qtensor()

    before = q.int_repr().clone()

    _ = flip_qint8_tensor_bit(
        q,
        element_index=0,
        bit_position=7,
    )

    assert torch.equal(
        before,
        q.int_repr(),
    )


def test_fp32_scalar_flip_is_exact():
    value = np.float32(
        1.0
    )

    before = fp32_payload_bits(
        value
    )

    after = flip_fp32_scalar(
        value,
        31,
    )

    assert (
        fp32_payload_bits(
            after
        )
        == (
            before
            ^ (1 << 31)
        )
    )


def test_fp32_tensor_mutates_exactly_one_payload():
    x = torch.tensor(
        [
            0.0,
            1.0,
            -1.0,
            10.0,
        ],
        dtype=torch.float32,
    )

    before = (
        x.numpy()
        .view(
            np.uint32
        )
        .copy()
    )

    mutated = flip_fp32_tensor_bit(
        x,
        element_index=2,
        bit_position=23,
    )

    after = (
        mutated.numpy()
        .view(
            np.uint32
        )
    )

    changed = np.flatnonzero(
        before
        != after
    )

    assert changed.tolist() == [2]

    assert int(
        after[2]
    ) == (
        int(
            before[2]
        )
        ^ (
            1 << 23
        )
    )


def test_fp32_double_flip_restores_payload_exactly():
    x = torch.tensor(
        [
            -3.25,
            0.0,
            100.5,
        ],
        dtype=torch.float32,
    )

    once = flip_fp32_tensor_bit(
        x,
        element_index=0,
        bit_position=31,
    )

    twice = flip_fp32_tensor_bit(
        once,
        element_index=0,
        bit_position=31,
    )

    assert fp32_payload_equal(
        x,
        twice,
    )


def test_fp32_nonfinite_is_not_sanitized():
    finite = fp32_from_payload_bits(
        0x7F000000
    )

    assert np.isfinite(
        finite
    )

    inf_value = flip_fp32_scalar(
        finite,
        23,
    )

    assert np.isinf(
        inf_value
    )

    max_finite = fp32_from_payload_bits(
        0x7F7FFFFF
    )

    nan_value = flip_fp32_scalar(
        max_finite,
        23,
    )

    assert np.isnan(
        nan_value
    )


def test_transient_schedule_only_on_selected_inference():
    q = qtensor()

    active = []

    for i in range(
        5
    ):
        out, meta = apply_scheduled_bit_fault(
            q,
            inference_index=i,
            onset_index=2,
            persistence="transient_one_inference",
            representation_class="quantized_activation",
            element_index=0,
            bit_position=0,
        )

        active.append(
            meta.active
        )

        if i == 2:
            assert not quantized_payload_equal(
                q,
                out,
            )
        else:
            assert quantized_payload_equal(
                q,
                out,
            )

    assert active == [
        False,
        False,
        True,
        False,
        False,
    ]


def test_persistent_schedule_starts_at_onset():
    x = torch.tensor(
        [
            1.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    active = []

    for i in range(
        5
    ):
        out, meta = apply_scheduled_bit_fault(
            x,
            inference_index=i,
            onset_index=2,
            persistence="persistent_from_onset_until_trial_end",
            representation_class="fp32_buffer",
            element_index=0,
            bit_position=31,
        )

        active.append(
            meta.active
        )

        if i >= 2:
            assert not fp32_payload_equal(
                x,
                out,
            )
        else:
            assert fp32_payload_equal(
                x,
                out,
            )

    assert active == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_new_trial_has_no_persistent_state_leakage():
    q1 = qtensor()

    _out, meta = apply_scheduled_bit_fault(
        q1,
        inference_index=4,
        onset_index=2,
        persistence="persistent_from_onset_until_trial_end",
        representation_class="quantized_buffer",
        element_index=0,
        bit_position=3,
    )

    assert meta.active

    q2 = qtensor()

    assert quantized_payload_equal(
        q1,
        q2,
    )


@pytest.mark.parametrize(
    "bit",
    [-1, 8],
)
def test_invalid_qint8_bits_rejected(bit):
    with pytest.raises(ValueError):
        flip_qint8_tensor_bit(
            qtensor(),
            element_index=0,
            bit_position=bit,
        )


@pytest.mark.parametrize(
    "bit",
    [-1, 32],
)
def test_invalid_fp32_bits_rejected(bit):
    x = torch.zeros(
        2,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        flip_fp32_tensor_bit(
            x,
            element_index=0,
            bit_position=bit,
        )


def test_wrong_fp32_dtype_rejected():
    x = torch.zeros(
        2,
        dtype=torch.float64,
    )

    with pytest.raises(TypeError):
        flip_fp32_tensor_bit(
            x,
            element_index=0,
            bit_position=0,
        )


def test_out_of_bounds_element_rejected():
    x = torch.zeros(
        2,
        dtype=torch.float32,
    )

    with pytest.raises(IndexError):
        flip_fp32_tensor_bit(
            x,
            element_index=2,
            bit_position=0,
        )
