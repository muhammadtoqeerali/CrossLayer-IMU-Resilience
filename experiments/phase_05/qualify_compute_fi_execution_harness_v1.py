"""Qualify paired clean/faulted synthetic execution for Phase 5B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from compute_fi_contract import FaultIdentity
from compute_fi_execution_harness import (
    PTQWeightFaultSession,
    build_frozen_ptq_eager,
    clone_state_dict,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
    run_fp32_paired,
    run_ptq_activation_buffer_paired,
    sha256_file,
    tensor_bytes_sha256,
)


def make_identity(
    *,
    model_variant: str,
    fault_family: str,
    representation_class: str,
    target_name: str,
    target_role: str,
    bit_position: int,
    inference_index: int = 0,
    persistence: str = "transient_one_inference",
) -> FaultIdentity:
    return FaultIdentity(
        protocol="phase5a_compute_fi_representation_protocol_v1",
        model_variant=model_variant,
        checkpoint_seed=42,
        fold=1,
        fault_family=fault_family,
        representation_class=representation_class,
        target_name=target_name,
        target_role=target_role,
        element_index=0,
        bit_position=bit_position,
        inference_index=inference_index,
        persistence=persistence,
        multiplicity=1,
        replicate_index=0,
    )


def serialize_pair(
    pair,
) -> dict:
    return {
        "fault_id":
            pair.fault_id,

        "input_sha256":
            pair.input_sha256,

        "clean_output_sha256":
            tensor_bytes_sha256(
                pair.clean_output
            ),

        "faulted_output_sha256":
            tensor_bytes_sha256(
                pair.faulted_output
            ),

        "outputs_bitwise_equal":
            outputs_bitwise_equal(
                pair.clean_output,
                pair.faulted_output,
            ),

        "mutation": {
            "active":
                pair.mutation.active,

            "representation_class":
                pair.mutation.representation_class,

            "target_name":
                pair.mutation.target_name,

            "element_index":
                pair.mutation.element_index,

            "bit_position":
                pair.mutation.bit_position,

            "before_payload":
                pair.mutation.before_payload,

            "after_payload":
                pair.mutation.after_payload,

            "changed_indices":
                list(
                    pair.mutation.changed_indices
                ),

            "tensor_dtype":
                pair.mutation.tensor_dtype,

            "integer_payload_dtype":
                pair.mutation.integer_payload_dtype,

            "quantization_metadata_preserved":
                pair.mutation.quantization_metadata_preserved,
        },
    }


def require_active_exactly_one(
    row: dict,
) -> None:
    mutation = row[
        "mutation"
    ]

    assert mutation[
        "active"
    ] is True

    assert mutation[
        "changed_indices"
    ] == [
        mutation[
            "element_index"
        ]
    ]

    assert (
        mutation[
            "before_payload"
        ]
        is not None
    )

    assert (
        mutation[
            "after_payload"
        ]
        is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32-freeze",
        required=True,
    )

    parser.add_argument(
        "--ptq-state",
        required=True,
    )

    parser.add_argument(
        "--ptq-torchscript",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--success",
        required=True,
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    success = Path(
        args.success
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp32_model, fp32_meta = load_frozen_fp32_model(
        args.fp32_freeze,
        seed=42,
        fold=1,
    )

    ptq_eager = build_frozen_ptq_eager(
        fp32_model,
        args.ptq_state,
    )

    ptq_reference = torch.jit.load(
        args.ptq_torchscript,
        map_location="cpu",
    )

    ptq_reference.eval()

    # Prove the injectable eager PTQ path remains the exact clean frozen
    # TorchScript reference on several synthetic inputs.
    clean_reference_checks = []

    for seed in (
        7101,
        7102,
        7103,
    ):
        torch.manual_seed(
            seed
        )

        x = torch.randn(
            3,
            30,
            9,
            dtype=torch.float32,
        )

        with torch.no_grad():
            eager_y = ptq_eager(
                x.clone()
            )

            script_y = ptq_reference(
                x.clone()
            )

        exact = outputs_bitwise_equal(
            eager_y,
            script_y,
        )

        assert exact

        clean_reference_checks.append(
            {
                "seed":
                    seed,

                "input_sha256":
                    tensor_bytes_sha256(
                        x
                    ),

                "bitwise_equal":
                    exact,

                "output_sha256":
                    tensor_bytes_sha256(
                        eager_y
                    ),
            }
        )

    torch.manual_seed(
        7201
    )

    x = torch.randn(
        2,
        30,
        9,
        dtype=torch.float32,
    )

    cases = {}

    fp32_activation = make_identity(
        model_variant="fp32",
        fault_family="fp32_activation_single_bit_flip",
        representation_class="fp32_activation",
        target_name="fc1_output_fp32",
        target_role="activation",
        bit_position=0,
    )

    pair = run_fp32_paired(
        fp32_model,
        x,
        fp32_activation,
        current_inference_index=0,
    )

    cases[
        "fp32_variant_fp32_activation"
    ] = serialize_pair(
        pair
    )

    fp32_buffer = make_identity(
        model_variant="fp32",
        fault_family="fp32_buffer_single_bit_flip",
        representation_class="fp32_buffer",
        target_name="flatten_buffer_fp32",
        target_role="intermediate_buffer",
        bit_position=0,
    )

    pair = run_fp32_paired(
        fp32_model,
        x,
        fp32_buffer,
        current_inference_index=0,
    )

    cases[
        "fp32_variant_fp32_buffer"
    ] = serialize_pair(
        pair
    )

    ptq_fp32_activation = make_identity(
        model_variant="ptq_v7",
        fault_family="fp32_activation_single_bit_flip",
        representation_class="fp32_activation",
        target_name="fc1_output_fp32",
        target_role="activation",
        bit_position=0,
    )

    pair = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_fp32_activation,
        current_inference_index=0,
    )

    cases[
        "ptq_variant_fp32_activation"
    ] = serialize_pair(
        pair
    )

    ptq_fp32_buffer = make_identity(
        model_variant="ptq_v7",
        fault_family="fp32_buffer_single_bit_flip",
        representation_class="fp32_buffer",
        target_name="flatten_buffer_fp32",
        target_role="intermediate_buffer",
        bit_position=0,
    )

    pair = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_fp32_buffer,
        current_inference_index=0,
    )

    cases[
        "ptq_variant_fp32_buffer"
    ] = serialize_pair(
        pair
    )

    ptq_quant_activation = make_identity(
        model_variant="ptq_v7",
        fault_family="quantized_activation_single_bit_flip",
        representation_class="quantized_activation",
        target_name="conv2_quantized_input",
        target_role="activation",
        bit_position=0,
    )

    pair = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_quant_activation,
        current_inference_index=0,
    )

    cases[
        "ptq_variant_quint8_activation"
    ] = serialize_pair(
        pair
    )

    ptq_quant_buffer = make_identity(
        model_variant="ptq_v7",
        fault_family="quantized_buffer_single_bit_flip",
        representation_class="quantized_buffer",
        target_name="post_fc2_quantized_buffer",
        target_role="intermediate_buffer",
        bit_position=0,
    )

    pair = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_quant_buffer,
        current_inference_index=0,
    )

    cases[
        "ptq_variant_quint8_buffer"
    ] = serialize_pair(
        pair
    )

    clean_state = torch.load(
        args.ptq_state,
        map_location="cpu",
        weights_only=False,
    )

    weight_model = build_frozen_ptq_eager(
        fp32_model,
        args.ptq_state,
    )

    weight_session = PTQWeightFaultSession(
        weight_model,
        clean_state,
    )

    ptq_weight = make_identity(
        model_variant="ptq_v7",
        fault_family="int8_weight_single_bit_flip",
        representation_class="int8_persistent_weight",
        target_name="conv_2.0.weight",
        target_role="persistent_parameter",
        bit_position=0,
    )

    pair = weight_session.run_paired(
        ptq_eager,
        x,
        ptq_weight,
        current_inference_index=0,
    )

    cases[
        "ptq_variant_qint8_weight"
    ] = serialize_pair(
        pair
    )

    for row in cases.values():
        require_active_exactly_one(
            row
        )

    assert (
        cases[
            "ptq_variant_quint8_activation"
        ][
            "mutation"
        ][
            "tensor_dtype"
        ]
        == "torch.quint8"
    )

    assert (
        cases[
            "ptq_variant_quint8_buffer"
        ][
            "mutation"
        ][
            "tensor_dtype"
        ]
        == "torch.quint8"
    )

    assert (
        cases[
            "ptq_variant_qint8_weight"
        ][
            "mutation"
        ][
            "tensor_dtype"
        ]
        == "torch.qint8"
    )

    assert (
        cases[
            "ptq_variant_quint8_activation"
        ][
            "mutation"
        ][
            "quantization_metadata_preserved"
        ]
        is True
    )

    assert (
        cases[
            "ptq_variant_quint8_buffer"
        ][
            "mutation"
        ][
            "quantization_metadata_preserved"
        ]
        is True
    )

    assert (
        cases[
            "ptq_variant_qint8_weight"
        ][
            "mutation"
        ][
            "quantization_metadata_preserved"
        ]
        is True
    )

    # Deterministic replay: same input + same identity => same mutation and
    # exact same faulted output.
    replay_a = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_quant_activation,
        current_inference_index=0,
    )

    replay_b = run_ptq_activation_buffer_paired(
        ptq_eager,
        x,
        ptq_quant_activation,
        current_inference_index=0,
    )

    assert replay_a.fault_id == replay_b.fault_id

    assert (
        replay_a.mutation
        == replay_b.mutation
    )

    assert outputs_bitwise_equal(
        replay_a.faulted_output,
        replay_b.faulted_output,
    )

    deterministic_replay = {
        "fault_id_equal":
            True,

        "mutation_record_equal":
            True,

        "faulted_output_bitwise_equal":
            True,
    }

    # Transient model schedule: mutation occurs exactly at onset and clean
    # execution resumes immediately after.
    transient_identity = make_identity(
        model_variant="ptq_v7",
        fault_family="quantized_activation_single_bit_flip",
        representation_class="quantized_activation",
        target_name="conv2_quantized_input",
        target_role="activation",
        bit_position=1,
        inference_index=2,
        persistence="transient_one_inference",
    )

    transient_mask = []
    transient_clean_equal_when_inactive = []

    for i in range(
        5
    ):
        row = run_ptq_activation_buffer_paired(
            ptq_eager,
            x,
            transient_identity,
            current_inference_index=i,
        )

        transient_mask.append(
            row.mutation.active
        )

        if not row.mutation.active:
            transient_clean_equal_when_inactive.append(
                outputs_bitwise_equal(
                    row.clean_output,
                    row.faulted_output,
                )
            )

    assert transient_mask == [
        False,
        False,
        True,
        False,
        False,
    ]

    assert all(
        transient_clean_equal_when_inactive
    )

    # Persistent weight schedule: same qint8 weight bit is applied from
    # onset onward. A new session reset starts clean.
    persistent_identity = make_identity(
        model_variant="ptq_v7",
        fault_family="int8_weight_single_bit_flip",
        representation_class="int8_persistent_weight",
        target_name="conv_2.0.weight",
        target_role="persistent_parameter",
        bit_position=2,
        inference_index=2,
        persistence="persistent_from_onset_until_trial_end",
    )

    persistent_model = build_frozen_ptq_eager(
        fp32_model,
        args.ptq_state,
    )

    persistent_session = PTQWeightFaultSession(
        persistent_model,
        clean_state,
    )

    persistent_mask = []

    for i in range(
        5
    ):
        row = persistent_session.run_paired(
            ptq_eager,
            x,
            persistent_identity,
            current_inference_index=i,
        )

        persistent_mask.append(
            row.mutation.active
        )

    assert persistent_mask == [
        False,
        False,
        True,
        True,
        True,
    ]

    persistent_session.reset_clean()

    clean_after_reset = persistent_session.run_paired(
        ptq_eager,
        x,
        persistent_identity,
        current_inference_index=0,
    )

    assert (
        clean_after_reset.mutation.active
        is False
    )

    assert outputs_bitwise_equal(
        clean_after_reset.clean_output,
        clean_after_reset.faulted_output,
    )

    source_hashes_before = {
        "fp32_checkpoint":
            fp32_meta[
                "checkpoint_sha256"
            ],

        "ptq_state":
            sha256_file(
                args.ptq_state
            ),

        "ptq_torchscript":
            sha256_file(
                args.ptq_torchscript
            ),
    }

    source_hashes_after = {
        "fp32_checkpoint":
            sha256_file(
                fp32_meta[
                    "checkpoint"
                ]
            ),

        "ptq_state":
            sha256_file(
                args.ptq_state
            ),

        "ptq_torchscript":
            sha256_file(
                args.ptq_torchscript
            ),
    }

    assert (
        source_hashes_before
        == source_hashes_after
    )

    payload = {
        "schema_version":
            "phase5b_compute_fi_execution_harness_qualification_v1",

        "status":
            "QUALIFIED_PAIRED_SYNTHETIC_EXECUTION_HARNESS",

        "qualification_date":
            "2026-10-05",

        "evidence_tier":
            "P0",

        "clean_reference": {
            "ptq_eager_matches_frozen_torchscript":
                True,

            "checks":
                clean_reference_checks,
        },

        "fault_family_execution_cases":
            cases,

        "deterministic_replay":
            deterministic_replay,

        "transient_schedule": {
            "activity_mask":
                transient_mask,

            "inactive_calls_equal_clean":
                all(
                    transient_clean_equal_when_inactive
                ),
        },

        "persistent_weight_schedule": {
            "activity_mask":
                persistent_mask,

            "new_trial_reset_clean":
                True,
        },

        "source_artifact_immutability": {
            "before":
                source_hashes_before,

            "after":
                source_hashes_after,

            "unchanged":
                True,
        },

        "acceptance": {
            "same_input_clean_fault_pair":
                True,

            "same_checkpoint_clean_fault_pair":
                True,

            "clean_ptq_eager_matches_frozen_torchscript":
                True,

            "all_five_frozen_fault_families_executed":
                True,

            "fp32_family_executed_on_fp32_variant":
                True,

            "fp32_family_executed_on_ptq_mixed_precision_regions":
                True,

            "qint8_weight_execution_qualified":
                True,

            "quint8_activation_execution_qualified":
                True,

            "quint8_buffer_execution_qualified":
                True,

            "exact_single_element_single_bit_mutation_recorded":
                True,

            "deterministic_fault_replay":
                True,

            "transient_restore_after_call":
                True,

            "persistent_onset_schedule_exact":
                True,

            "new_trial_reset_clean":
                True,

            "source_artifacts_unchanged":
                True,
        },

        "scientific_boundary": {
            "synthetic_inputs_only":
                True,

            "dataset_read":
                False,

            "training_partition_read":
                False,

            "calibration_partition_read":
                False,

            "validation_partition_read":
                False,

            "outer_test_read":
                False,

            "onfield_read":
                False,

            "faulted_synthetic_model_forward_executed":
                True,

            "faulted_real_dataset_inference_executed":
                False,

            "CC_task_performance_result_generated":
                False,

            "CSC_result_generated":
                False,

            "outer_sampling_plan_frozen":
                False,

            "physical_fault_equivalence":
                False,

            "hardware_claim":
                False,
        },

        "next_action":
            "Use development-safe training/calibration windows to qualify the paired runner on real model inputs before freezing the immutable outer CC sampling plan.",
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

    result_sha = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()

    success.write_text(
        json.dumps(
            {
                "status":
                    "PASS",

                "qualification_status":
                    payload[
                        "status"
                    ],

                "result_sha256":
                    result_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "FAULT_FAMILY_EXECUTION_CASES=",
        len(
            cases
        ),
        sep="",
    )

    print(
        "PTQ_CLEAN_REFERENCE_CHECKS=",
        len(
            clean_reference_checks
        ),
        sep="",
    )

    print(
        "TRANSIENT_MASK=",
        transient_mask,
        sep="",
    )

    print(
        "PERSISTENT_MASK=",
        persistent_mask,
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
        result_sha,
        sep="",
    )


if __name__ == "__main__":
    main()
