from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_contract import FaultIdentity  # noqa: E402
from compute_fi_execution_harness import (  # noqa: E402
    FP32_BOUNDARY_TO_MODULE,
    PTQ_BOUNDARY_TO_NODE,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
    run_fp32_paired,
)


FP32_FREEZE = (
    ROOT
    / "manifests"
    / "phase_4f_prospective_300ms_fp32_freeze_v1.json"
)


def identity(
    *,
    target,
    representation,
    family,
    role,
    onset=0,
    persistence="transient_one_inference",
):
    return FaultIdentity(
        protocol="phase5a_compute_fi_representation_protocol_v1",
        model_variant="fp32",
        checkpoint_seed=42,
        fold=1,
        fault_family=family,
        representation_class=representation,
        target_name=target,
        target_role=role,
        element_index=0,
        bit_position=0,
        inference_index=onset,
        persistence=persistence,
        multiplicity=1,
        replicate_index=0,
    )


def test_fp32_boundary_map_has_frozen_fp32_targets():
    expected = {
        "front_end_output_fp32",
        "conv2_dequantized_output_fp32",
        "post_conv2_prelu_fp32",
        "post_conv2_pool_fp32",
        "post_conv2_dropout_fp32",
        "flatten_buffer_fp32",
        "fc1_output_fp32",
        "fc2_prelu_output_fp32",
        "post_fc2_dequantized_buffer_fp32",
        "classifier_output_fp32",
    }

    assert expected.issubset(
        FP32_BOUNDARY_TO_MODULE
    )


def test_ptq_boundary_map_has_quantized_targets():
    assert (
        PTQ_BOUNDARY_TO_NODE[
            "conv2_quantized_input"
        ]
        == "quantize_per_tensor"
    )

    assert (
        PTQ_BOUNDARY_TO_NODE[
            "conv2_quantized_output"
        ]
        == "conv_2_0"
    )

    assert (
        PTQ_BOUNDARY_TO_NODE[
            "post_fc2_quantized_buffer"
        ]
        == "quantize_per_tensor_2"
    )


def test_fp32_transient_pair_inactive_before_onset_is_clean():
    model, _ = load_frozen_fp32_model(
        FP32_FREEZE,
        seed=42,
        fold=1,
    )

    torch.manual_seed(
        8101
    )

    x = torch.randn(
        1,
        30,
        9,
        dtype=torch.float32,
    )

    fault = identity(
        target="fc1_output_fp32",
        representation="fp32_activation",
        family="fp32_activation_single_bit_flip",
        role="activation",
        onset=2,
    )

    pair = run_fp32_paired(
        model,
        x,
        fault,
        current_inference_index=1,
    )

    assert pair.mutation.active is False

    assert outputs_bitwise_equal(
        pair.clean_output,
        pair.faulted_output,
    )


def test_fp32_transient_pair_mutates_exact_selected_element_at_onset():
    model, _ = load_frozen_fp32_model(
        FP32_FREEZE,
        seed=42,
        fold=1,
    )

    torch.manual_seed(
        8102
    )

    x = torch.randn(
        1,
        30,
        9,
        dtype=torch.float32,
    )

    fault = identity(
        target="flatten_buffer_fp32",
        representation="fp32_buffer",
        family="fp32_buffer_single_bit_flip",
        role="intermediate_buffer",
    )

    pair = run_fp32_paired(
        model,
        x,
        fault,
        current_inference_index=0,
    )

    assert pair.mutation.active is True

    assert pair.mutation.changed_indices == (
        0,
    )

    assert pair.mutation.tensor_dtype == "torch.float32"


def test_fp32_fault_replay_is_deterministic():
    model, _ = load_frozen_fp32_model(
        FP32_FREEZE,
        seed=42,
        fold=1,
    )

    torch.manual_seed(
        8103
    )

    x = torch.randn(
        1,
        30,
        9,
        dtype=torch.float32,
    )

    fault = identity(
        target="fc1_output_fp32",
        representation="fp32_activation",
        family="fp32_activation_single_bit_flip",
        role="activation",
    )

    a = run_fp32_paired(
        model,
        x,
        fault,
        current_inference_index=0,
    )

    b = run_fp32_paired(
        model,
        x,
        fault,
        current_inference_index=0,
    )

    assert a.fault_id == b.fault_id
    assert a.mutation == b.mutation

    assert outputs_bitwise_equal(
        a.clean_output,
        b.clean_output,
    )

    assert outputs_bitwise_equal(
        a.faulted_output,
        b.faulted_output,
    )
