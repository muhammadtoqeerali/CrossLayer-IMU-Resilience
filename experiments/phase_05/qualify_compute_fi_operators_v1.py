"""Qualify Phase-5A compute-FI operators without task inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from compute_fi_contract import deterministic_index
from compute_fi_operators import (
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
    quantized_payload_equal,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def qualify_int8_scalars() -> int:
    cases = 0

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

            restored = flip_int8_payload(
                observed,
                bit,
            )

            assert restored == value

            cases += 1

    return cases


def make_synthetic_qint8() -> torch.Tensor:
    payload = torch.tensor(
        [
            [
                [-128, -127, -64, -1],
                [0, 1, 63, 127],
            ],
            [
                [-120, -80, -32, -2],
                [2, 32, 80, 120],
            ],
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


def qualify_qint8_tensor() -> int:
    q = make_synthetic_qint8()

    original_payload = q.int_repr().clone()

    cases = 0

    for index in range(
        q.numel()
    ):
        for bit in range(
            8
        ):
            mutated = flip_qint8_tensor_bit(
                q,
                element_index=index,
                bit_position=bit,
            )

            assert torch.equal(
                q.int_repr(),
                original_payload,
            )

            before = (
                original_payload
                .reshape(-1)
                .clone()
            )

            after = (
                mutated
                .int_repr()
                .reshape(-1)
            )

            expected_u8 = (
                int(
                    before[index].item()
                )
                & 0xFF
            ) ^ (
                1 << bit
            )

            expected = (
                expected_u8 - 256
                if expected_u8 >= 128
                else expected_u8
            )

            assert int(
                after[index].item()
            ) == expected

            mask = torch.ones(
                q.numel(),
                dtype=torch.bool,
            )

            mask[index] = False

            assert torch.equal(
                after[mask],
                before[mask],
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

            restored = flip_qint8_tensor_bit(
                mutated,
                element_index=index,
                bit_position=bit,
            )

            assert quantized_payload_equal(
                restored,
                q,
            )

            cases += 1

    return cases



def make_synthetic_quint8() -> torch.Tensor:
    payload = torch.tensor(
        [
            [
                [0, 1, 63, 127],
                [128, 129, 200, 255],
            ],
            [
                [2, 32, 80, 120],
                [130, 160, 220, 254],
            ],
        ],
        dtype=torch.uint8,
    )

    return torch._make_per_tensor_quantized_tensor(
        payload,
        0.125,
        128,
    )


def qualify_quint8_tensor() -> int:
    q = make_synthetic_quint8()

    assert q.dtype == torch.quint8
    assert q.int_repr().dtype == torch.uint8

    original_payload = q.int_repr().clone()

    cases = 0

    for index in range(
        q.numel()
    ):
        for bit in range(
            8
        ):
            mutated = flip_quint8_tensor_bit(
                q,
                element_index=index,
                bit_position=bit,
            )

            assert q.dtype == torch.quint8
            assert mutated.dtype == torch.quint8
            assert mutated.int_repr().dtype == torch.uint8

            assert torch.equal(
                q.int_repr(),
                original_payload,
            )

            before = (
                original_payload
                .reshape(-1)
                .clone()
            )

            after = (
                mutated
                .int_repr()
                .reshape(-1)
            )

            expected = (
                int(
                    before[index].item()
                )
                ^ (
                    1 << bit
                )
            )

            assert int(
                after[index].item()
            ) == expected

            mask = torch.ones(
                q.numel(),
                dtype=torch.bool,
            )

            mask[index] = False

            assert torch.equal(
                after[mask],
                before[mask],
            )

            assert (
                mutated.qscheme()
                == q.qscheme()
            )

            assert (
                mutated.q_scale()
                == q.q_scale()
            )

            assert (
                mutated.q_zero_point()
                == q.q_zero_point()
            )

            restored = flip_quantized_tensor_bit(
                mutated,
                element_index=index,
                bit_position=bit,
            )

            assert quantized_payload_equal(
                restored,
                q,
            )

            cases += 1

    return cases


def synthetic_fp32_payloads() -> torch.Tensor:
    payloads = np.asarray(
        [
            0x00000000,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x40000000,
            0xC0000000,
            0x00800000,
            0x007FFFFF,
            0x7F000000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x3EAAAAAB,
            0x41200000,
            0xC1200000,
            0x00000001,
            0x80000001,
        ],
        dtype=np.uint32,
    )

    values = payloads.view(
        np.float32
    ).copy()

    return torch.from_numpy(
        values
    )


def qualify_fp32_tensor() -> int:
    x = synthetic_fp32_payloads()

    original_bytes = (
        x.numpy().tobytes()
    )

    cases = 0

    for index in range(
        x.numel()
    ):
        for bit in range(
            32
        ):
            mutated = flip_fp32_tensor_bit(
                x,
                element_index=index,
                bit_position=bit,
            )

            assert (
                x.numpy().tobytes()
                == original_bytes
            )

            before_bits = fp32_payload_bits(
                x[index].numpy()
            )

            after_bits = fp32_payload_bits(
                mutated[index].numpy()
            )

            assert (
                after_bits
                == (
                    before_bits
                    ^ (1 << bit)
                )
            )

            for other in range(
                x.numel()
            ):
                if other == index:
                    continue

                assert (
                    fp32_payload_bits(
                        x[other].numpy()
                    )
                    == fp32_payload_bits(
                        mutated[other].numpy()
                    )
                )

            restored = flip_fp32_tensor_bit(
                mutated,
                element_index=index,
                bit_position=bit,
            )

            assert fp32_payload_equal(
                restored,
                x,
            )

            cases += 1

    return cases


def qualify_nonfinite_preservation() -> dict:
    # Exponent=254, mantissa=0.
    # Flipping exponent LSB (bit 23) produces +Inf exactly.
    finite_bits = 0x7F000000

    finite = fp32_from_payload_bits(
        finite_bits
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

    assert fp32_payload_bits(
        inf_value
    ) == 0x7F800000

    # Max finite -> exponent 255 with non-zero mantissa = NaN.
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

    assert (
        fp32_payload_bits(
            nan_value
        )
        == 0x7FFFFFFF
    )

    return {
        "finite_to_positive_infinity_preserved":
            True,

        "finite_to_nan_preserved":
            True,

        "sanitization_performed":
            False,
    }


def qualify_schedule_semantics() -> dict:
    q = make_synthetic_quint8()

    transient_active = []
    persistent_active = []

    for inference_index in range(
        6
    ):
        transient, tmeta = apply_scheduled_bit_fault(
            q,
            inference_index=inference_index,
            onset_index=2,
            persistence="transient_one_inference",
            representation_class="quantized_activation",
            element_index=3,
            bit_position=7,
        )

        persistent, pmeta = apply_scheduled_bit_fault(
            q,
            inference_index=inference_index,
            onset_index=2,
            persistence="persistent_from_onset_until_trial_end",
            representation_class="quantized_activation",
            element_index=3,
            bit_position=7,
        )

        transient_active.append(
            tmeta.active
        )

        persistent_active.append(
            pmeta.active
        )

        if tmeta.active:
            assert not quantized_payload_equal(
                transient,
                q,
            )
        else:
            assert quantized_payload_equal(
                transient,
                q,
            )

        if pmeta.active:
            assert not quantized_payload_equal(
                persistent,
                q,
            )
        else:
            assert quantized_payload_equal(
                persistent,
                q,
            )

        # Source object is never mutated.
        assert quantized_payload_equal(
            q,
            make_synthetic_quint8(),
        )

    assert transient_active == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]

    assert persistent_active == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    # New trial starts from a clean parent tensor: no state leakage.
    new_trial = make_synthetic_quint8()

    assert quantized_payload_equal(
        new_trial,
        q,
    )

    return {
        "transient_activity_mask":
            transient_active,

        "persistent_activity_mask":
            persistent_active,

        "cross_trial_state_leakage":
            False,
    }


def qualify_real_ptq_state(
    state_path: Path,
) -> dict:
    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    q = state[
        "conv_2.0.weight"
    ]

    assert q.dtype == torch.qint8
    assert q.is_quantized
    assert q.numel() == 16384

    original_payload = (
        q.int_repr().clone()
    )

    cases = [
        (0, 0),
        (1, 7),
        (8191, 3),
        (16383, 5),
    ]

    for index, bit in cases:
        mutated = flip_qint8_tensor_bit(
            q,
            element_index=index,
            bit_position=bit,
        )

        before = (
            original_payload
            .reshape(-1)
        )

        after = (
            mutated
            .int_repr()
            .reshape(-1)
        )

        expected_u8 = (
            int(
                before[index].item()
            )
            & 0xFF
        ) ^ (
            1 << bit
        )

        expected = (
            expected_u8 - 256
            if expected_u8 >= 128
            else expected_u8
        )

        assert int(
            after[index].item()
        ) == expected

        changed = (
            before
            != after
        ).nonzero(
            as_tuple=False
        ).reshape(
            -1
        )

        assert changed.tolist() == [
            index
        ]

        assert torch.equal(
            mutated.q_per_channel_scales(),
            q.q_per_channel_scales(),
        )

        assert torch.equal(
            mutated.q_per_channel_zero_points(),
            q.q_per_channel_zero_points(),
        )

        restored = flip_qint8_tensor_bit(
            mutated,
            element_index=index,
            bit_position=bit,
        )

        assert quantized_payload_equal(
            restored,
            q,
        )

    assert torch.equal(
        q.int_repr(),
        original_payload,
    )

    return {
        "target":
            "conv_2.0.weight",

        "dtype":
            str(
                q.dtype
            ),

        "numel":
            q.numel(),

        "qscheme":
            str(
                q.qscheme()
            ),

        "cases":
            len(
                cases
            ),

        "source_unchanged":
            True,

        "quantization_metadata_preserved":
            True,

        "double_flip_restoration":
            True,
    }


def resolve_fp32_checkpoint(
    freeze_manifest: Path,
) -> tuple[Path, str]:
    obj = json.loads(
        freeze_manifest.read_text()
    )

    row = next(
        x
        for x in obj[
            "checkpoints"
        ]
        if x["seed"] == 42
        and x["fold"] == 1
    )

    path = Path(
        row[
            "checkpoint"
        ]
    )

    assert path.is_file()

    assert sha256(
        path
    ) == row[
        "sha256"
    ]

    return (
        path,
        row["sha256"],
    )


def qualify_real_fp32_checkpoint(
    freeze_manifest: Path,
) -> dict:
    checkpoint, digest = resolve_fp32_checkpoint(
        freeze_manifest
    )

    obj = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = obj[
        "state_dict"
    ]

    tensor_names = [
        "model.conv_1.0.weight",
        "model.conv_2.0.weight",
        "model.fc.1.weight",
        "model.fc.4.weight",
    ]

    bit_positions = [
        0,
        22,
        23,
        31,
    ]

    rows = []

    for tensor_name, bit in zip(
        tensor_names,
        bit_positions,
        strict=True,
    ):
        tensor = state[
            tensor_name
        ]

        assert (
            tensor.dtype
            == torch.float32
        )

        original_bytes = (
            tensor
            .contiguous()
            .numpy()
            .tobytes()
        )

        index = deterministic_index(
            namespace=(
                "phase5a-real-fp32-"
                + tensor_name
            ),
            upper_bound=tensor.numel(),
            fields={
                "seed": 42,
                "fold": 1,
                "tensor_name": tensor_name,
                "bit_position": bit,
            },
        )

        mutated = flip_fp32_tensor_bit(
            tensor,
            element_index=index,
            bit_position=bit,
        )

        assert (
            tensor
            .contiguous()
            .numpy()
            .tobytes()
            == original_bytes
        )

        before_payloads = (
            tensor
            .contiguous()
            .numpy()
            .view(
                np.uint32
            )
            .reshape(
                -1
            )
        )

        after_payloads = (
            mutated
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
            before_payloads
            != after_payloads
        )

        assert changed.tolist() == [
            index
        ]

        assert int(
            after_payloads[
                index
            ]
        ) == (
            int(
                before_payloads[
                    index
                ]
            )
            ^ (
                1 << bit
            )
        )

        restored = flip_fp32_tensor_bit(
            mutated,
            element_index=index,
            bit_position=bit,
        )

        assert fp32_payload_equal(
            restored,
            tensor,
        )

        rows.append(
            {
                "tensor":
                    tensor_name,

                "element_index":
                    index,

                "bit_position":
                    bit,

                "numel":
                    tensor.numel(),

                "source_unchanged":
                    True,

                "double_flip_restoration":
                    True,
            }
        )

    return {
        "checkpoint":
            str(
                checkpoint
            ),

        "checkpoint_sha256":
            digest,

        "seed":
            42,

        "fold":
            1,

        "cases":
            rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--success",
        required=True,
    )

    parser.add_argument(
        "--fp32-freeze",
        required=True,
    )

    parser.add_argument(
        "--ptq-state",
        required=True,
    )

    parser.add_argument(
        "--ptq-artifact-manifest",
        required=True,
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    success = Path(
        args.success
    )

    fp32_freeze = Path(
        args.fp32_freeze
    )

    ptq_state = Path(
        args.ptq_state
    )

    ptq_artifact_manifest = Path(
        args.ptq_artifact_manifest
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ptq_manifest = json.loads(
        ptq_artifact_manifest.read_text()
    )

    expected_ptq_sha = ptq_manifest[
        "seed_42/fold_1/v7_state_dict.pt"
    ][
        "sha256"
    ]

    assert sha256(
        ptq_state
    ) == expected_ptq_sha

    int8_scalar_cases = qualify_int8_scalars()
    qint8_tensor_cases = qualify_qint8_tensor()
    quint8_tensor_cases = qualify_quint8_tensor()
    fp32_tensor_cases = qualify_fp32_tensor()

    nonfinite = qualify_nonfinite_preservation()
    schedule = qualify_schedule_semantics()

    real_ptq = qualify_real_ptq_state(
        ptq_state
    )

    real_fp32 = qualify_real_fp32_checkpoint(
        fp32_freeze
    )

    payload = {
        "schema_version":
            "phase5a_compute_fi_operator_qualification_v1",

        "status":
            "QUALIFIED_BIT_EXACT_P0_OPERATORS",

        "qualification_date":
            "2026-10-05",

        "evidence_tier":
            "P0",

        "synthetic_qualification": {
            "int8_scalar_bit_flip_cases":
                int8_scalar_cases,

            "qint8_tensor_bit_flip_cases":
                qint8_tensor_cases,

            "quint8_tensor_bit_flip_cases":
                quint8_tensor_cases,

            "fp32_tensor_bit_flip_cases":
                fp32_tensor_cases,

            "representation_coverage": {
                "persistent_weight": {
                    "torch_dtype":
                        "torch.qint8",

                    "integer_payload_dtype":
                        "torch.int8",

                    "role":
                        "INT8 persistent weight",
                },

                "quantized_activation_buffer": {
                    "torch_dtype":
                        "torch.quint8",

                    "integer_payload_dtype":
                        "torch.uint8",

                    "role":
                        "quantized activation/intermediate buffer",
                },
            },

            "nonfinite_preservation":
                nonfinite,

            "schedule_semantics":
                schedule,
        },

        "frozen_artifact_qualification": {
            "ptq_state": {
                "path":
                    str(
                        ptq_state
                    ),

                "sha256":
                    expected_ptq_sha,

                "result":
                    real_ptq,
            },

            "fp32_checkpoint":
                real_fp32,
        },

        "acceptance": {
            "exactly_one_selected_payload_bit_changes":
                True,

            "all_nonselected_payload_elements_unchanged":
                True,

            "double_flip_restores_exact_payload":
                True,

            "source_tensor_unchanged":
                True,

            "quantized_metadata_preserved":
                True,

            "FP32_nan_inf_not_sanitized":
                True,

            "transient_schedule_exact":
                True,

            "persistent_schedule_exact":
                True,

            "cross_trial_state_leakage":
                False,

            "deterministic_target_mapping":
                True,
        },

        "scientific_boundary": {
            "dataset_read":
                False,

            "model_forward_passes":
                0,

            "faulted_model_inference":
                False,

            "outer_test_read":
                False,

            "onfield_read":
                False,

            "CC_result_generated":
                False,

            "CSC_result_generated":
                False,

            "physical_fault_equivalence":
                False,

            "hardware_claim":
                False,
        },
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    success_payload = {
        "status":
            "PASS",

        "qualification_status":
            payload[
                "status"
            ],

        "operator_qualification_sha256":
            sha256(
                output
            ),
    }

    success.write_text(
        json.dumps(
            success_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "INT8_SCALAR_CASES=",
        int8_scalar_cases,
        sep="",
    )

    print(
        "QINT8_TENSOR_CASES=",
        qint8_tensor_cases,
        sep="",
    )

    print(
        "QUINT8_TENSOR_CASES=",
        quint8_tensor_cases,
        sep="",
    )

    print(
        "FP32_TENSOR_CASES=",
        fp32_tensor_cases,
        sep="",
    )

    print(
        "REAL_PTQ_CASES=",
        real_ptq[
            "cases"
        ],
        sep="",
    )

    print(
        "REAL_FP32_CASES=",
        len(
            real_fp32[
                "cases"
            ]
        ),
        sep="",
    )

    print(
        "QUALIFICATION_STATUS=",
        payload[
            "status"
        ],
        sep="",
    )

    print(
        "RESULT_SHA256=",
        sha256(
            output
        ),
        sep="",
    )


if __name__ == "__main__":
    main()
