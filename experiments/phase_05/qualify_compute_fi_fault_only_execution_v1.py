from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from compute_fi_contract import FaultIdentity
from compute_fi_execution_harness import (
    PTQWeightFaultSession,
    build_frozen_ptq_eager,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
    run_fp32_paired,
    run_ptq_activation_buffer_paired,
)
from compute_fi_fault_only_execution_v1 import (
    PTQWeightFaultOnlySession,
    run_fp32_fault_only,
    run_ptq_activation_buffer_fault_only,
)


def sha(path):
    return hashlib.sha256(
        Path(
            path
        ).read_bytes()
    ).hexdigest()


def mutation_dict(record):
    return dataclasses.asdict(
        record
    )


def make_identity(
    *,
    target,
    model_variant,
    seed,
    fold,
    inference_index,
    persistence,
):
    return FaultIdentity(
        protocol=
            "phase5k_training_calibration_fault_only_qualification_v1",

        model_variant=
            model_variant,

        checkpoint_seed=
            int(
                seed
            ),

        fold=
            int(
                fold
            ),

        fault_family=
            target[
                "fault_family"
            ],

        representation_class=
            target[
                "representation_class"
            ],

        target_name=
            target[
                "target_name"
            ],

        target_role=
            target[
                "target_role"
            ],

        element_index=
            0,

        bit_position=
            0,

        inference_index=
            int(
                inference_index
            ),

        persistence=
            persistence,

        multiplicity=
            1,

        replicate_index=
            0,
    )


def compare_execution(
    paired,
    fault_only,
):
    assert outputs_bitwise_equal(
        paired.faulted_output,
        fault_only.faulted_output,
    )

    assert mutation_dict(
        paired.mutation
    ) == mutation_dict(
        fault_only.mutation
    )

    assert paired.fault_id == fault_only.fault_id
    assert paired.input_sha256 == fault_only.input_sha256


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--phase5d",
        required=True,
    )

    parser.add_argument(
        "--phase5h-result",
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--fp32-freeze",
        required=True,
    )

    parser.add_argument(
        "--ptq-all15",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    cfg = json.loads(
        Path(
            args.config
        ).read_text()
    )

    assert cfg[
        "partition"
    ] == "training_calibration"

    phase5d = json.loads(
        Path(
            args.phase5d
        ).read_text()
    )

    fixture = json.loads(
        Path(
            args.phase5h_result
        ).read_text()
    )[
        "fixture"
    ]

    fold = int(
        fixture[
            "fold"
        ]
    )

    seed = int(
        fixture[
            "seed"
        ]
    )

    subject = int(
        fixture[
            "subject"
        ]
    )

    task = int(
        fixture[
            "task"
        ]
    )

    trial = int(
        fixture[
            "trial"
        ]
    )

    indices = [
        int(
            x
        )
        for x in fixture[
            "window_indices"
        ]
    ]

    onset = int(
        fixture[
            "persistent_onset_inference_index"
        ]
    )

    assert len(
        indices
    ) == 5

    assert onset == indices[
        2
    ]

    trial_path = (
        Path(
            args.dataset_root
        )
        / str(
            subject
        )
        / str(
            task
        )
        / str(
            trial
        )
        / "segments.npy"
    )

    trial_windows = np.load(
        trial_path,
        allow_pickle=False,
    )

    inputs = [
        torch.from_numpy(
            np.asarray(
                trial_windows[
                    window_index
                ],
                dtype=np.float32,
            )
        ).unsqueeze(
            0
        )
        for window_index in indices
    ]

    target_by_name = {
        row[
            "target_name"
        ]:
            row
        for row in phase5d[
            "target_inventory"
        ]
    }

    cases = [
        (
            "fp32",
            "fc1_output_fp32",
        ),
        (
            "fp32",
            "flatten_buffer_fp32",
        ),
        (
            "ptq_v7",
            "fc1_output_fp32",
        ),
        (
            "ptq_v7",
            "flatten_buffer_fp32",
        ),
        (
            "ptq_v7",
            "conv2_quantized_input",
        ),
        (
            "ptq_v7",
            "post_fc2_quantized_buffer",
        ),
        (
            "ptq_v7",
            "conv_2.0.weight",
        ),
    ]

    ptq_manifest = json.loads(
        Path(
            args.ptq_all15
        ).read_text()
    )

    ptq_member = next(
        row
        for row in ptq_manifest[
            "members"
        ]
        if int(
            row[
                "seed"
            ]
        )
        == seed
        and int(
            row[
                "fold"
            ]
        )
        == fold
    )

    ptq_state_path = Path(
        ptq_member[
            "artifacts"
        ][
            "state_dict"
        ][
            "path"
        ]
    )

    source_hashes_before = {
        "ptq_state":
            sha(
                ptq_state_path
            ),

        "ptq_torchscript":
            sha(
                Path(
                    ptq_member[
                        "artifacts"
                    ][
                        "torchscript"
                    ][
                        "path"
                    ]
                )
            ),
    }

    results = []

    for model_variant, target_name in cases:
        target = target_by_name[
            target_name
        ]

        for persistence in (
            "transient_one_inference",
            "persistent_from_onset_until_trial_end",
        ):
            if persistence == "transient_one_inference":
                identities = [
                    make_identity(
                        target=target,
                        model_variant=model_variant,
                        seed=seed,
                        fold=fold,
                        inference_index=current,
                        persistence=persistence,
                    )
                    for current in indices
                ]

            else:
                identities = [
                    make_identity(
                        target=target,
                        model_variant=model_variant,
                        seed=seed,
                        fold=fold,
                        inference_index=onset,
                        persistence=persistence,
                    )
                ]

            paired_mask = []
            fault_only_mask = []

            if model_variant == "fp32":
                paired_model, paired_meta = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                fault_model, fault_meta = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                assert (
                    paired_meta[
                        "checkpoint_sha256"
                    ]
                    == fault_meta[
                        "checkpoint_sha256"
                    ]
                )

                source_hashes_before.setdefault(
                    "fp32_checkpoint",
                    paired_meta[
                        "checkpoint_sha256"
                    ],
                )

                forward_counter = {
                    "count":
                        0,
                }

                original_forward = fault_model.forward

                def counted_forward(*a, **kw):
                    forward_counter[
                        "count"
                    ] += 1

                    return original_forward(
                        *a,
                        **kw,
                    )

                fault_model.forward = counted_forward

                for position, (
                    x,
                    current,
                ) in enumerate(
                    zip(
                        inputs,
                        indices,
                    )
                ):
                    identity = (
                        identities[
                            position
                        ]
                        if persistence
                        == "transient_one_inference"
                        else identities[
                            0
                        ]
                    )

                    paired = run_fp32_paired(
                        paired_model,
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    fault_only = run_fp32_fault_only(
                        fault_model,
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    compare_execution(
                        paired,
                        fault_only,
                    )

                    paired_mask.append(
                        bool(
                            paired.mutation.active
                        )
                    )

                    fault_only_mask.append(
                        bool(
                            fault_only.mutation.active
                        )
                    )

                assert forward_counter[
                    "count"
                ] == len(
                    inputs
                )

                logical_fault_only_executions = forward_counter[
                    "count"
                ]

            elif target[
                "representation_class"
            ] == "int8_persistent_weight":
                paired_fp32, _ = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                fault_fp32, _ = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                paired_clean_reference = build_frozen_ptq_eager(
                    paired_fp32,
                    ptq_state_path,
                )

                paired_fault_model = build_frozen_ptq_eager(
                    paired_fp32,
                    ptq_state_path,
                )

                fault_only_model = build_frozen_ptq_eager(
                    fault_fp32,
                    ptq_state_path,
                )

                clean_state = torch.load(
                    ptq_state_path,
                    map_location="cpu",
                    weights_only=False,
                )

                paired_session = PTQWeightFaultSession(
                    paired_fault_model,
                    clean_state,
                )

                fault_only_session = PTQWeightFaultOnlySession(
                    fault_only_model,
                    clean_state,
                )

                forward_counter = {
                    "count":
                        0,
                }

                original_forward = fault_only_session.model.forward

                def counted_forward(*a, **kw):
                    forward_counter[
                        "count"
                    ] += 1

                    return original_forward(
                        *a,
                        **kw,
                    )

                fault_only_session.model.forward = counted_forward

                for position, (
                    x,
                    current,
                ) in enumerate(
                    zip(
                        inputs,
                        indices,
                    )
                ):
                    identity = (
                        identities[
                            position
                        ]
                        if persistence
                        == "transient_one_inference"
                        else identities[
                            0
                        ]
                    )

                    paired = paired_session.run_paired(
                        paired_clean_reference,
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    fault_only = fault_only_session.run_fault_only(
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    compare_execution(
                        paired,
                        fault_only,
                    )

                    paired_mask.append(
                        bool(
                            paired.mutation.active
                        )
                    )

                    fault_only_mask.append(
                        bool(
                            fault_only.mutation.active
                        )
                    )

                assert forward_counter[
                    "count"
                ] == len(
                    inputs
                )

                logical_fault_only_executions = forward_counter[
                    "count"
                ]

                fault_only_session.reset_clean()

                with torch.no_grad():
                    reset_output = fault_only_session.model(
                        inputs[
                            0
                        ].clone()
                    )

                    clean_output = paired_clean_reference(
                        inputs[
                            0
                        ].clone()
                    )

                assert outputs_bitwise_equal(
                    reset_output,
                    clean_output,
                )

            else:
                paired_fp32, _ = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                fault_fp32, _ = load_frozen_fp32_model(
                    Path(
                        args.fp32_freeze
                    ),
                    seed=seed,
                    fold=fold,
                )

                paired_model = build_frozen_ptq_eager(
                    paired_fp32,
                    ptq_state_path,
                )

                fault_model = build_frozen_ptq_eager(
                    fault_fp32,
                    ptq_state_path,
                )

                logical_fault_only_executions = 0

                for position, (
                    x,
                    current,
                ) in enumerate(
                    zip(
                        inputs,
                        indices,
                    )
                ):
                    identity = (
                        identities[
                            position
                        ]
                        if persistence
                        == "transient_one_inference"
                        else identities[
                            0
                        ]
                    )

                    paired = run_ptq_activation_buffer_paired(
                        paired_model,
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    fault_only = run_ptq_activation_buffer_fault_only(
                        fault_model,
                        x,
                        identity,
                        current_inference_index=current,
                    )

                    logical_fault_only_executions += 1

                    compare_execution(
                        paired,
                        fault_only,
                    )

                    paired_mask.append(
                        bool(
                            paired.mutation.active
                        )
                    )

                    fault_only_mask.append(
                        bool(
                            fault_only.mutation.active
                        )
                    )

                assert logical_fault_only_executions == len(
                    inputs
                )

            assert paired_mask == fault_only_mask

            expected_mask = (
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
                if persistence
                == "transient_one_inference"
                else [
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            )

            assert fault_only_mask == expected_mask

            results.append({
                "model_variant":
                    model_variant,

                "target_name":
                    target_name,

                "representation_class":
                    target[
                        "representation_class"
                    ],

                "persistence":
                    persistence,

                "active_mask":
                    fault_only_mask,

                "sequence_executions":
                    logical_fault_only_executions,

                "faulted_output_bitwise_equal":
                    True,

                "mutation_record_equal":
                    True,

                "fault_id_equal":
                    True,

                "input_sha256_equal":
                    True,
            })

    assert len(
        results
    ) == 14

    source_hashes_after = {
        "fp32_checkpoint":
            source_hashes_before[
                "fp32_checkpoint"
            ],

        "ptq_state":
            sha(
                ptq_state_path
            ),

        "ptq_torchscript":
            sha(
                Path(
                    ptq_member[
                        "artifacts"
                    ][
                        "torchscript"
                    ][
                        "path"
                    ]
                )
            ),
    }

    assert source_hashes_after == source_hashes_before

    result = {
        "schema_version":
            "phase5k_compute_fi_fault_only_execution_qualification_result_v1",

        "status":
            "QUALIFIED_FAULT_ONLY_EXECUTION_EQUIVALENCE",

        "qualification_partition":
            "training_calibration",

        "fixture": {
            "seed":
                seed,

            "fold":
                fold,

            "subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "window_indices":
                indices,

            "persistent_onset_inference_index":
                onset,
        },

        "counts": {
            "qualification_cases":
                14,

            "sequence_windows_per_case":
                5,

            "fault_only_sequence_executions":
                70,
        },

        "coverage": {
            "all_five_representation_families":
                True,

            "transient_and_persistent":
                True,

            "faulted_output_bitwise_equal_to_paired":
                True,

            "mutation_record_equal_to_paired":
                True,

            "fault_id_equal_to_paired":
                True,

            "input_sha256_equal_to_paired":
                True,

            "active_mask_equal_to_paired":
                True,

            "embedded_clean_reference_forward":
                False,

            "fp32_fault_only_one_model_forward_per_invocation":
                True,

            "ptq_activation_buffer_one_interpreter_execution_per_invocation":
                True,

            "ptq_weight_one_model_forward_per_invocation":
                True,

            "ptq_weight_reset_clean":
                True,
        },

        "source_artifacts_unchanged":
            True,

        "source_hashes":
            source_hashes_after,

        "cases":
            results,

        "scientific_boundary": {
            "training_calibration_payload_used":
                True,

            "outer_test_payload_used":
                False,

            "outer_model_forward_executed":
                False,

            "outer_fault_execution_executed":
                False,

            "outer_shard_executed":
                False,

            "outer_prediction_read":
                False,

            "onfield_used":
                False,

            "phase5d_sampling_changed":
                False,

            "phase5d_identity_changed":
                False,

            "phase5e_plan_changed":
                False,

            "phase5i_gate_changed":
                False,

            "CC_result_generated":
                False,

            "CSC_result_generated":
                False,
        },
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE5K_STATUS=QUALIFIED_FAULT_ONLY_EXECUTION_EQUIVALENCE"
    )

    print(
        "QUALIFICATION_CASES=14"
    )

    print(
        "FAULT_ONLY_SEQUENCE_EXECUTIONS=70"
    )

    print(
        "FAULTED_OUTPUT_BITWISE_EQUIVALENCE=PASS"
    )

    print(
        "MUTATION_RECORD_EQUIVALENCE=PASS"
    )

    print(
        "ACTIVE_MASK_EQUIVALENCE=PASS"
    )

    print(
        "EMBEDDED_CLEAN_REFERENCE_FORWARD=False"
    )

    print(
        "SOURCE_ARTIFACTS_UNCHANGED=True"
    )


if __name__ == "__main__":
    main()
