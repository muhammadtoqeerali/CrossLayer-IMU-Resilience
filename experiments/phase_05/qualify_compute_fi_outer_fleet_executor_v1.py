from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from compute_fi_contract import FaultIdentity
from compute_fi_execution_harness import (
    build_frozen_ptq_eager,
    clone_state_dict,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
)
from compute_fi_outer_fleet_executor_v1 import (
    PHASE5A_PROTOCOL,
    execution_active_mask,
    execute_fault_sequence,
    sha256_file,
    shard_authorization,
    validate_gate,
    validate_shard_structure,
)


def load_json(path):
    return json.loads(
        Path(path).read_text()
    )


def make_identity(
    *,
    variant,
    fold,
    target,
    persistence,
    inference_index,
):
    return FaultIdentity(
        protocol=
            PHASE5A_PROTOCOL,

        model_variant=
            variant,

        checkpoint_seed=
            42,

        fold=
            fold,

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
            inference_index,

        persistence=
            persistence,

        multiplicity=
            1,

        replicate_index=
            0,
    )


def find_target(
    phase5d,
    name,
):
    matches = [
        row
        for row in phase5d[
            "target_inventory"
        ]
        if row[
            "target_name"
        ]
        == name
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def file_hashes(paths):
    return {
        str(path):
            sha256_file(
                path
            )
        for path in paths
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--executor",
        required=True,
    )

    parser.add_argument(
        "--phase5d",
        required=True,
    )

    parser.add_argument(
        "--phase5e-plan",
        required=True,
    )

    parser.add_argument(
        "--phase5k-result",
        required=True,
    )

    parser.add_argument(
        "--fp32-freeze",
        required=True,
    )

    parser.add_argument(
        "--ptq-manifest",
        required=True,
    )

    parser.add_argument(
        "--calibration-identity",
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--outer-result-root",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_json(
        args.config
    )

    phase5d = load_json(
        args.phase5d
    )

    plan = load_json(
        args.phase5e_plan
    )

    phase5k = load_json(
        args.phase5k_result
    )

    ptq_manifest = load_json(
        args.ptq_manifest
    )

    calibration = load_json(
        args.calibration_identity
    )

    assert (
        cfg[
            "qualification_partition"
        ]
        == "training_calibration"
    )

    assert (
        cfg[
            "outer_execution_allowed_during_qualification"
        ]
        is False
    )

    assert (
        phase5k[
            "status"
        ]
        == "QUALIFIED_FAULT_ONLY_EXECUTION_EQUIVALENCE"
    )

    boundary = phase5k[
        "scientific_boundary"
    ]

    assert boundary[
        "training_calibration_payload_used"
    ] is True

    assert boundary[
        "outer_test_payload_used"
    ] is False

    assert boundary[
        "onfield_used"
    ] is False

    outer_root = Path(
        args.outer_result_root
    )

    initial_outer_dirs = sorted(
        path.name
        for path in outer_root.iterdir()
        if path.is_dir()
    )

    assert len(
        initial_outer_dirs
    ) == 2

    validated = validate_gate(
        output_root=outer_root,
    )

    # Validate representative metadata-only shards from all four classes.
    representatives = {}

    for row in plan[
        "shards"
    ]:
        key = (
            row[
                "model_variant"
            ],
            row[
                "persistence"
            ],
        )

        if key not in representatives:
            representatives[
                key
            ] = row

    assert set(
        representatives
    ) == {
        (
            "fp32",
            "transient_one_inference",
        ),
        (
            "fp32",
            "persistent_from_onset_until_trial_end",
        ),
        (
            "ptq_v7",
            "transient_one_inference",
        ),
        (
            "ptq_v7",
            "persistent_from_onset_until_trial_end",
        ),
    }

    representative_checks = []

    for key, shard in sorted(
        representatives.items()
    ):
        authorization = shard_authorization(
            validated,
            shard[
                "shard_id"
            ],
        )

        structure = validate_shard_structure(
            phase5d=phase5d,
            plan=plan,
            shard=authorization[
                "shard"
            ],
        )

        representative_checks.append({
            "model_variant":
                key[
                    0
                ],

            "persistence":
                key[
                    1
                ],

            "shard_id":
                shard[
                    "shard_id"
                ],

            "outer_instance_ids":
                structure[
                    "expected_outer_instance_ids"
                ],

            "faulted_model_window_evaluations":
                structure[
                    "expected_faulted_model_window_evaluations"
                ],
        })

    # Frozen training-only calibration identity: first lexicographic row fold 1.
    fold_rec = next(
        row
        for row in calibration[
            "folds"
        ]
        if int(
            row[
                "fold"
            ]
        ) == 1
    )

    assert int(
        fold_rec[
            "calibration_window_count"
        ]
    ) == 4096

    assert fold_rec[
        "outer_test_subject_leakage"
    ] is False

    assert fold_rec[
        "validation_subject_leakage"
    ] is False

    assert fold_rec[
        "onfield_used"
    ] is False

    selection = sorted(
        fold_rec[
            "selection"
        ],
        key=lambda row: (
            int(
                row[
                    "subject"
                ]
            ),
            int(
                row[
                    "task"
                ]
            ),
            int(
                row[
                    "trial"
                ]
            ),
            int(
                row[
                    "window_index"
                ]
            ),
        ),
    )

    identity = selection[
        0
    ]

    trial_path = (
        Path(
            args.dataset_root
        )
        / str(
            int(
                identity[
                    "subject"
                ]
            )
        )
        / str(
            int(
                identity[
                    "task"
                ]
            )
        )
        / str(
            int(
                identity[
                    "trial"
                ]
            )
        )
        / "segments.npy"
    )

    assert trial_path.is_file()

    trial_windows = np.load(
        trial_path,
        allow_pickle=False,
    )

    assert (
        trial_windows.ndim
        == 3
    )

    assert tuple(
        trial_windows.shape[
            1:
        ]
    ) == (
        30,
        9,
    )

    assert np.isfinite(
        trial_windows
    ).all()

    anchor = int(
        identity[
            "window_index"
        ]
    )

    length = int(
        cfg[
            "qualification_sequence_length"
        ]
    )

    start = min(
        anchor,
        int(
            trial_windows.shape[
                0
            ]
        )
        - length,
    )

    start = max(
        0,
        start,
    )

    indices = list(
        range(
            start,
            start
            + length,
        )
    )

    assert len(
        indices
    ) == 5

    inputs = [
        torch.from_numpy(
            np.asarray(
                trial_windows[
                    index
                ],
                dtype=np.float32,
            )
        ).unsqueeze(
            0
        )
        for index in indices
    ]

    # Load seed42/fold1 frozen FP32 and PTQ models.
    fp32_model, fp32_meta = (
        load_frozen_fp32_model(
            args.fp32_freeze,
            seed=42,
            fold=1,
        )
    )

    member = next(
        row
        for row in ptq_manifest[
            "members"
        ]
        if int(
            row[
                "seed"
            ]
        )
        == 42
        and int(
            row[
                "fold"
            ]
        )
        == 1
    )

    ptq_state = Path(
        member[
            "artifacts"
        ][
            "state_dict"
        ][
            "path"
        ]
    )

    ptq_ts = Path(
        member[
            "artifacts"
        ][
            "torchscript"
        ][
            "path"
        ]
    )

    assert sha256_file(
        ptq_state
    ) == member[
        "artifacts"
    ][
        "state_dict"
    ][
        "sha256"
    ]

    assert sha256_file(
        ptq_ts
    ) == member[
        "artifacts"
    ][
        "torchscript"
    ][
        "sha256"
    ]

    source_paths = [
        Path(
            fp32_meta[
                "checkpoint"
            ]
        ),
        ptq_state,
        ptq_ts,
        trial_path,
    ]

    source_hashes_before = file_hashes(
        source_paths
    )

    ptq_model = build_frozen_ptq_eager(
        fp32_model,
        ptq_state,
    )

    ptq_clean_state = clone_state_dict(
        ptq_model.state_dict()
    )

    ts_model = torch.jit.load(
        ptq_ts,
        map_location="cpu",
    )

    ts_model.eval()

    with torch.no_grad():
        eager_probe = ptq_model(
            inputs[
                0
            ].clone()
        )

        ts_probe = ts_model(
            inputs[
                0
            ].clone()
        )

    assert outputs_bitwise_equal(
        eager_probe,
        ts_probe,
    )

    onset = int(
        cfg[
            "persistent_onset_index"
        ]
    )

    qualification_cases = []

    def run_case(
        name,
        *,
        model,
        variant,
        target_name,
        persistence,
        weight=False,
    ):
        target = find_target(
            phase5d,
            target_name,
        )

        if persistence == "transient_one_inference":
            identities = [
                make_identity(
                    variant=variant,
                    fold=1,
                    target=target,
                    persistence=persistence,
                    inference_index=index,
                )
                for index in range(
                    len(
                        inputs
                    )
                )
            ]

            inference_indices = list(
                range(
                    len(
                        inputs
                    )
                )
            )

            expected_mask = [
                True
            ] * len(
                inputs
            )

        else:
            identities = [
                make_identity(
                    variant=variant,
                    fold=1,
                    target=target,
                    persistence=persistence,
                    inference_index=onset,
                )
            ]

            inference_indices = list(
                range(
                    len(
                        inputs
                    )
                )
            )

            expected_mask = [
                index >= onset
                for index in inference_indices
            ]

        executions = execute_fault_sequence(
            model=model,
            inputs=inputs,
            inference_indices=inference_indices,
            identities=identities,
            ptq_clean_state=(
                ptq_clean_state
                if weight
                else None
            ),
        )

        observed_mask = execution_active_mask(
            executions
        )

        assert observed_mask == expected_mask

        assert len(
            executions
        ) == 5

        qualification_cases.append({
            "name":
                name,

            "model_variant":
                variant,

            "target_name":
                target_name,

            "representation_class":
                target[
                    "representation_class"
                ],

            "persistence":
                persistence,

            "sequence_executions":
                len(
                    executions
                ),

            "active_mask":
                observed_mask,

            "embedded_clean_reference_forward":
                False,
        })

    run_case(
        "fp32_transient_activation",
        model=fp32_model,
        variant="fp32",
        target_name="front_end_output_fp32",
        persistence="transient_one_inference",
    )

    run_case(
        "fp32_persistent_buffer",
        model=fp32_model,
        variant="fp32",
        target_name="flatten_buffer_fp32",
        persistence="persistent_from_onset_until_trial_end",
    )

    run_case(
        "ptq_transient_quantized_activation",
        model=ptq_model,
        variant="ptq_v7",
        target_name="conv2_quantized_input",
        persistence="transient_one_inference",
    )

    run_case(
        "ptq_persistent_quantized_buffer",
        model=ptq_model,
        variant="ptq_v7",
        target_name="post_fc2_quantized_buffer",
        persistence="persistent_from_onset_until_trial_end",
    )

    run_case(
        "ptq_transient_qint8_weight",
        model=ptq_model,
        variant="ptq_v7",
        target_name="conv_2.0.weight",
        persistence="transient_one_inference",
        weight=True,
    )

    # Weight-session reset after transient route.
    with torch.no_grad():
        reset_probe = ptq_model(
            inputs[
                0
            ].clone()
        )

    assert outputs_bitwise_equal(
        reset_probe,
        ts_probe,
    )

    run_case(
        "ptq_persistent_qint8_weight",
        model=ptq_model,
        variant="ptq_v7",
        target_name="conv_2.0.weight",
        persistence="persistent_from_onset_until_trial_end",
        weight=True,
    )

    # Weight-session reset after persistent route.
    with torch.no_grad():
        reset_probe_2 = ptq_model(
            inputs[
                0
            ].clone()
        )

    assert outputs_bitwise_equal(
        reset_probe_2,
        ts_probe,
    )

    source_hashes_after = file_hashes(
        source_paths
    )

    assert (
        source_hashes_before
        == source_hashes_after
    )

    final_outer_dirs = sorted(
        path.name
        for path in outer_root.iterdir()
        if path.is_dir()
    )

    assert (
        final_outer_dirs
        == initial_outer_dirs
    )

    result = {
        "schema_version":
            "phase5r_compute_fi_outer_full_fleet_executor_qualification_result_v1",

        "phase":
            "5R",

        "status":
            "QUALIFIED_FULL_FLEET_EXECUTOR_PRE_OUTER",

        "qualification_partition":
            "training_calibration",

        "phase5p_gate_sha256":
            sha256_file(
                cfg[
                    "phase5p_gate"
                ][
                    "path"
                ]
            ),

        "executor_sha256":
            sha256_file(
                args.executor
            ),

        "training_calibration_source": {
            "fold":
                1,

            "subject":
                int(
                    identity[
                        "subject"
                    ]
                ),

            "task":
                int(
                    identity[
                        "task"
                    ]
                ),

            "trial":
                int(
                    identity[
                        "trial"
                    ]
                ),

            "frozen_anchor_window_index":
                int(
                    identity[
                        "window_index"
                    ]
                ),

            "qualification_window_indices":
                indices,

            "trial_segments_sha256":
                sha256_file(
                    trial_path
                ),

            "labels_read":
                False,
        },

        "representative_outer_metadata_checks":
            representative_checks,

        "qualification_cases":
            qualification_cases,

        "qualification_case_count":
            len(
                qualification_cases
            ),

        "fault_only_sequence_executions":
            sum(
                row[
                    "sequence_executions"
                ]
                for row in qualification_cases
            ),

        "expected_active_masks": {
            "transient":
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                ],

            "persistent":
                [
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
        },

        "ptq_eager_torchscript_clean_equal":
            True,

        "ptq_weight_reset_after_transient":
            True,

        "ptq_weight_reset_after_persistent":
            True,

        "source_artifacts_unchanged":
            True,

        "accepted_canary_artifacts_unchanged":
            True,

        "scientific_boundary": {
            "training_calibration_payload_used":
                True,

            "validation_payload_used":
                False,

            "outer_test_payload_used":
                False,

            "onfield_payload_used":
                False,

            "outer_full_fleet_execution_started":
                False,

            "additional_outer_fault_shard_executed":
                False,

            "additional_outer_clean_cache_executed":
                False,

            "prediction_outcomes_used_for_selection":
                False,

            "threshold_applied":
                False,

            "threshold_changed":
                False,

            "metric_computed":
                False,

            "sampling_changed":
                False,

            "identity_changed":
                False,

            "phase5e_plan_changed":
                False,

            "phase5p_gate_changed":
                False,

            "aggregate_CC_result_generated":
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
        "PHASE5R_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    print(
        "TRAINING_CALIBRATION_SUBJECT=",
        result[
            "training_calibration_source"
        ][
            "subject"
        ],
        sep="",
    )

    print(
        "TRAINING_CALIBRATION_TASK=",
        result[
            "training_calibration_source"
        ][
            "task"
        ],
        sep="",
    )

    print(
        "TRAINING_CALIBRATION_TRIAL=",
        result[
            "training_calibration_source"
        ][
            "trial"
        ],
        sep="",
    )

    print(
        "QUALIFICATION_CASES=",
        result[
            "qualification_case_count"
        ],
        sep="",
    )

    print(
        "FAULT_ONLY_SEQUENCE_EXECUTIONS=",
        result[
            "fault_only_sequence_executions"
        ],
        sep="",
    )

    print(
        "TRANSIENT_ACTIVE_MASK=[True, True, True, True, True]"
    )

    print(
        "PERSISTENT_ACTIVE_MASK=[False, False, True, True, True]"
    )

    print(
        "PTQ_WEIGHT_RESET_TRANSIENT=PASS"
    )

    print(
        "PTQ_WEIGHT_RESET_PERSISTENT=PASS"
    )

    print(
        "SOURCE_ARTIFACTS_UNCHANGED=True"
    )

    print(
        "ACCEPTED_CANARY_ARTIFACTS_UNCHANGED=True"
    )

    print(
        "ADDITIONAL_OUTER_EXECUTION=False"
    )

    print(
        "RESULT_SHA256=",
        sha256_file(
            output
        ),
        sep="",
    )


if __name__ == "__main__":
    main()
