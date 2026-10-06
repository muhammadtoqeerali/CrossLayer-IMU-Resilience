"""Phase-5C development-safe real-window compute-FI runner qualification.

Only the frozen training-only calibration identity set is permitted.

No validation, outer-test or OnField execution is supported here.
No classification performance metric participates in acceptance or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

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
    sha256_file,
    tensor_bytes_sha256,
)


def load_phase4_runner(
    path: str | Path,
):
    path = Path(
        path
    )

    spec = importlib.util.spec_from_file_location(
        "phase5c_bound_phase4_sensor_runner",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load authoritative Phase-4 dev/cal runner"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def file_sha(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(
            path
        ).read_bytes()
    ).hexdigest()


def calibration_first_identity_by_fold(
    calibration_path: str | Path,
) -> dict[int, dict[str, Any]]:
    data = json.loads(
        Path(
            calibration_path
        ).read_text()
    )

    result = {}

    for fold_record in data[
        "folds"
    ]:
        fold = int(
            fold_record[
                "fold"
            ]
        )

        if int(
            fold_record[
                "calibration_window_count"
            ]
        ) != 4096:
            raise ValueError(
                f"fold {fold} calibration cardinality changed"
            )

        if bool(
            fold_record[
                "onfield_used"
            ]
        ):
            raise ValueError(
                f"fold {fold} calibration unexpectedly uses OnField"
            )

        if fold_record[
            "validation_subject_leakage"
        ]:
            raise ValueError(
                f"fold {fold} calibration has validation leakage"
            )

        if fold_record[
            "outer_test_subject_leakage"
        ]:
            raise ValueError(
                f"fold {fold} calibration has outer-test leakage"
            )

        selection = sorted(
            fold_record[
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

        if len(
            selection
        ) != 4096:
            raise ValueError(
                f"fold {fold} selection does not contain 4096 windows"
            )

        result[
            fold
        ] = selection[
            0
        ]

    if set(
        result
    ) != {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(
            "calibration fold coverage mismatch"
        )

    return result


def fp32_rows(
    path: str | Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    data = json.loads(
        Path(
            path
        ).read_text()
    )

    result = {}

    for row in data[
        "checkpoints"
    ]:
        key = (
            int(
                row[
                    "seed"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
        )

        if key in result:
            raise ValueError(
                f"duplicate FP32 member {key}"
            )

        result[
            key
        ] = row

    return result


def ptq_rows(
    path: str | Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    data = json.loads(
        Path(
            path
        ).read_text()
    )

    result = {}

    for row in data[
        "members"
    ]:
        key = (
            int(
                row[
                    "seed"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
        )

        if key in result:
            raise ValueError(
                f"duplicate PTQ member {key}"
            )

        result[
            key
        ] = row

    return result


def mutation_dict(
    pair,
) -> dict[str, Any]:
    mutation = pair.mutation

    return {
        "active":
            bool(
                mutation.active
            ),

        "representation_class":
            mutation.representation_class,

        "target_name":
            mutation.target_name,

        "element_index":
            int(
                mutation.element_index
            ),

        "bit_position":
            int(
                mutation.bit_position
            ),

        "before_payload":
            mutation.before_payload,

        "after_payload":
            mutation.after_payload,

        "changed_indices":
            [
                int(
                    x
                )
                for x in mutation.changed_indices
            ],

        "tensor_dtype":
            mutation.tensor_dtype,

        "integer_payload_dtype":
            mutation.integer_payload_dtype,

        "quantization_metadata_preserved":
            mutation.quantization_metadata_preserved,
    }


def make_fault_identity(
    *,
    protocol: dict[str, Any],
    seed: int,
    fold: int,
    model_variant: str,
    case: dict[str, Any],
) -> FaultIdentity:
    fixed = protocol[
        "fixed_fault_coordinates"
    ]

    return FaultIdentity(
        protocol=protocol[
            "schema_version"
        ],
        model_variant=model_variant,
        checkpoint_seed=int(
            seed
        ),
        fold=int(
            fold
        ),
        fault_family=case[
            "fault_family"
        ],
        representation_class=case[
            "representation_class"
        ],
        target_name=case[
            "target_name"
        ],
        target_role=case[
            "target_role"
        ],
        element_index=int(
            fixed[
                "element_index"
            ]
        ),
        bit_position=int(
            fixed[
                "bit_position"
            ]
        ),
        inference_index=int(
            fixed[
                "inference_index"
            ]
        ),
        persistence=fixed[
            "persistence"
        ],
        multiplicity=int(
            fixed[
                "multiplicity"
            ]
        ),
        replicate_index=int(
            fixed[
                "replicate_index"
            ]
        ),
    )


def require_active_exact_mutation(
    pair,
) -> None:
    mutation = pair.mutation

    if not mutation.active:
        raise AssertionError(
            "qualification fault was unexpectedly inactive"
        )

    if mutation.changed_indices != (
        mutation.element_index,
    ):
        raise AssertionError(
            "qualification fault changed the wrong payload element(s)"
        )

    if mutation.before_payload is None:
        raise AssertionError(
            "missing mutation before payload"
        )

    if mutation.after_payload is None:
        raise AssertionError(
            "missing mutation after payload"
        )

    expected = (
        int(
            mutation.before_payload
        )
        ^ (
            1
            << int(
                mutation.bit_position
            )
        )
    )

    if mutation.tensor_dtype in (
        "torch.qint8",
        "torch.quint8",
    ):
        if (
            int(
                mutation.after_payload
            )
            & 0xFF
        ) != (
            expected
            & 0xFF
        ):
            raise AssertionError(
                "quantized mutation is not the selected XOR bit"
            )

        if (
            mutation.quantization_metadata_preserved
            is not True
        ):
            raise AssertionError(
                "quantization metadata changed"
            )

    else:
        if int(
            mutation.after_payload
        ) != expected:
            raise AssertionError(
                "FP32 mutation is not the selected XOR bit"
            )


def source_artifact_ledger(
    fp32: dict[tuple[int, int], dict[str, Any]],
    ptq: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, str]:
    ledger = {}

    for key in sorted(
        fp32
    ):
        seed, fold = key

        row = fp32[
            key
        ]

        checkpoint = Path(
            row[
                "checkpoint"
            ]
        )

        actual = file_sha(
            checkpoint
        )

        if actual != row[
            "sha256"
        ]:
            raise AssertionError(
                f"FP32 source hash mismatch for {key}"
            )

        ledger[
            f"fp32_seed{seed}_fold{fold}"
        ] = actual

        ptq_row = ptq[
            key
        ]

        for kind in (
            "state_dict",
            "torchscript",
        ):
            artifact = ptq_row[
                "artifacts"
            ][
                kind
            ]

            actual = file_sha(
                artifact[
                    "path"
                ]
            )

            if actual != artifact[
                "sha256"
            ]:
                raise AssertionError(
                    f"PTQ {kind} source hash mismatch for {key}"
                )

            ledger[
                f"ptq_{kind}_seed{seed}_fold{fold}"
            ] = actual

    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        required=True,
    )

    parser.add_argument(
        "--calibration",
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
        "--phase4-runner",
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

    protocol_path = Path(
        args.protocol
    )

    protocol = json.loads(
        protocol_path.read_text()
    )

    if (
        protocol[
            "status"
        ]
        != "FROZEN_PRE_EXECUTION_QUALIFICATION_PROTOCOL"
    ):
        raise ValueError(
            "Phase-5C protocol is not frozen pre-execution"
        )

    if (
        protocol[
            "partition"
        ][
            "qualification_partition"
        ]
        != "training_calibration"
    ):
        raise ValueError(
            "Phase-5C runner qualification must use training_calibration"
        )

    if (
        protocol[
            "expected_cardinality"
        ][
            "paired_cases_total"
        ]
        != 105
    ):
        raise ValueError(
            "unexpected Phase-5C qualification cardinality"
        )

    phase4 = load_phase4_runner(
        args.phase4_runner
    )

    phase4.assert_partition(
        "training_calibration",
        execution_stage="qualification",
    )

    rejected = {}

    for prohibited in (
        "validation",
        "outer_test",
        "onfield",
    ):
        did_reject = False

        try:
            phase4.assert_partition(
                prohibited,
                execution_stage="qualification",
            )

        except ValueError:
            did_reject = True

        if not did_reject:
            raise AssertionError(
                f"qualification failed to reject {prohibited}"
            )

        rejected[
            prohibited
        ] = True

    calibration = calibration_first_identity_by_fold(
        args.calibration
    )

    authoritative = phase4.select_calibration_identity_by_fold(
        args.calibration
    )

    for fold in range(
        1,
        6,
    ):
        expected = calibration[
            fold
        ]

        observed = authoritative[
            fold
        ]

        for key in (
            "subject",
            "task",
            "trial",
            "window_index",
        ):
            if int(
                expected[
                    key
                ]
            ) != int(
                observed[
                    key
                ]
            ):
                raise AssertionError(
                    f"authoritative calibration selector mismatch fold={fold}"
                )

    fp32 = fp32_rows(
        args.fp32_freeze
    )

    ptq = ptq_rows(
        args.ptq_all15
    )

    expected_members = {
        (
            seed,
            fold,
        )
        for seed in (
            42,
            123,
            2025,
        )
        for fold in range(
            1,
            6,
        )
    }

    if set(
        fp32
    ) != expected_members:
        raise AssertionError(
            "FP32 member estate mismatch"
        )

    if set(
        ptq
    ) != expected_members:
        raise AssertionError(
            "PTQ member estate mismatch"
        )

    for key in sorted(
        expected_members
    ):
        fp32_row = fp32[
            key
        ]

        ptq_row = ptq[
            key
        ]

        if (
            fp32_row[
                "sha256"
            ]
            != ptq_row[
                "checkpoint"
            ][
                "sha256"
            ]
        ):
            raise AssertionError(
                f"FP32/PTQ parent checkpoint SHA mismatch for {key}"
            )

        if (
            Path(
                fp32_row[
                    "checkpoint"
                ]
            )
            != Path(
                ptq_row[
                    "checkpoint"
                ][
                    "path"
                ]
            )
        ):
            raise AssertionError(
                f"FP32/PTQ parent checkpoint path mismatch for {key}"
            )

        cal = ptq_row[
            "calibration"
        ]

        if int(
            cal[
                "fold"
            ]
        ) != key[
            1
        ]:
            raise AssertionError(
                f"PTQ calibration fold mismatch for {key}"
            )

        if (
            cal[
                "partition"
            ]
            != "training only"
        ):
            raise AssertionError(
                f"PTQ calibration partition changed for {key}"
            )

        if int(
            cal[
                "windows"
            ]
        ) != 4096:
            raise AssertionError(
                f"PTQ calibration cardinality changed for {key}"
            )

    artifact_hashes_before = source_artifact_ledger(
        fp32,
        ptq,
    )

    rows = []
    member_checks = []
    loaded_windows: dict[int, tuple[torch.Tensor, int, dict[str, Any]]] = {}

    for fold in range(
        1,
        6,
    ):
        identity = calibration[
            fold
        ]

        window_np, label = phase4.load_stored_window(
            args.dataset_root,
            identity,
        )

        if tuple(
            window_np.shape
        ) != (
            30,
            9,
        ):
            raise AssertionError(
                f"real calibration window shape mismatch fold={fold}"
            )

        if not np.isfinite(
            window_np
        ).all():
            raise AssertionError(
                f"real calibration window non-finite fold={fold}"
            )

        x = torch.from_numpy(
            np.asarray(
                window_np,
                dtype=np.float32,
            )
        ).unsqueeze(
            0
        )

        if tuple(
            x.shape
        ) != (
            1,
            30,
            9,
        ):
            raise AssertionError(
                "tensorized window shape mismatch"
            )

        loaded_windows[
            fold
        ] = (
            x,
            int(
                label
            ),
            identity,
        )

    for seed, fold in sorted(
        expected_members
    ):
        x, label, calibration_identity = loaded_windows[
            fold
        ]

        fp32_model, fp32_meta = load_frozen_fp32_model(
            args.fp32_freeze,
            seed=seed,
            fold=fold,
        )

        ptq_row = ptq[
            (
                seed,
                fold,
            )
        ]

        state_path = Path(
            ptq_row[
                "artifacts"
            ][
                "state_dict"
            ][
                "path"
            ]
        )

        script_path = Path(
            ptq_row[
                "artifacts"
            ][
                "torchscript"
            ][
                "path"
            ]
        )

        ptq_eager = build_frozen_ptq_eager(
            fp32_model,
            state_path,
        )

        ptq_script = torch.jit.load(
            str(
                script_path
            ),
            map_location="cpu",
        )

        ptq_script.eval()

        with torch.no_grad():
            clean_fp32 = fp32_model(
                x.clone()
            )

            clean_ptq_eager = ptq_eager(
                x.clone()
            )

            clean_ptq_script = ptq_script(
                x.clone()
            )

        if not outputs_bitwise_equal(
            clean_ptq_eager,
            clean_ptq_script,
        ):
            delta = float(
                (
                    clean_ptq_eager
                    - clean_ptq_script
                )
                .abs()
                .max()
                .item()
            )

            raise AssertionError(
                "eager PTQ does not exactly reproduce frozen TorchScript "
                f"for seed={seed} fold={fold}; max_abs_delta={delta}"
            )

        input_sha = tensor_bytes_sha256(
            x
        )

        member_checks.append({
            "seed":
                seed,

            "fold":
                fold,

            "partition":
                "training_calibration",

            "subject_id":
                int(
                    calibration_identity[
                        "subject"
                    ]
                ),

            "task_id":
                int(
                    calibration_identity[
                        "task"
                    ]
                ),

            "trial_id":
                int(
                    calibration_identity[
                        "trial"
                    ]
                ),

            "window_index":
                int(
                    calibration_identity[
                        "window_index"
                    ]
                ),

            "label":
                label,

            "selection_sha256":
                calibration_identity[
                    "selection_sha256"
                ],

            "input_sha256":
                input_sha,

            "fp32_checkpoint_sha256":
                fp32_meta[
                    "checkpoint_sha256"
                ],

            "ptq_state_sha256":
                ptq_row[
                    "artifacts"
                ][
                    "state_dict"
                ][
                    "sha256"
                ],

            "ptq_torchscript_sha256":
                ptq_row[
                    "artifacts"
                ][
                    "torchscript"
                ][
                    "sha256"
                ],

            "ptq_eager_torchscript_clean_bitwise_equal":
                True,

            "clean_fp32_output_sha256":
                tensor_bytes_sha256(
                    clean_fp32
                ),

            "clean_ptq_output_sha256":
                tensor_bytes_sha256(
                    clean_ptq_eager
                ),
        })

        for case in protocol[
            "fp32_cases_per_member"
        ]:
            fault = make_fault_identity(
                protocol=protocol,
                seed=seed,
                fold=fold,
                model_variant="fp32",
                case=case,
            )

            pair = run_fp32_paired(
                fp32_model,
                x,
                fault,
                current_inference_index=0,
            )

            require_active_exact_mutation(
                pair
            )

            if pair.input_sha256 != input_sha:
                raise AssertionError(
                    "FP32 paired input hash mismatch"
                )

            if not outputs_bitwise_equal(
                pair.clean_output,
                clean_fp32,
            ):
                raise AssertionError(
                    "FP32 pair clean output changed relative to clean reference"
                )

            rows.append({
                "protocol_version":
                    protocol[
                        "schema_version"
                    ],

                "model_variant":
                    "fp32",

                "checkpoint_seed":
                    seed,

                "fold":
                    fold,

                "checkpoint_sha256":
                    fp32_meta[
                        "checkpoint_sha256"
                    ],

                "ptq_state_sha256":
                    None,

                "ptq_torchscript_sha256":
                    None,

                "partition":
                    "training_calibration",

                "subject_id":
                    int(
                        calibration_identity[
                            "subject"
                        ]
                    ),

                "task_id":
                    int(
                        calibration_identity[
                            "task"
                        ]
                    ),

                "trial_id":
                    int(
                        calibration_identity[
                            "trial"
                        ]
                    ),

                "window_index":
                    int(
                        calibration_identity[
                            "window_index"
                        ]
                    ),

                "label":
                    label,

                "selection_sha256":
                    calibration_identity[
                        "selection_sha256"
                    ],

                "input_sha256":
                    pair.input_sha256,

                "fault_id":
                    pair.fault_id,

                "fault_family":
                    case[
                        "fault_family"
                    ],

                "representation_class":
                    case[
                        "representation_class"
                    ],

                "target_name":
                    case[
                        "target_name"
                    ],

                "target_role":
                    case[
                        "target_role"
                    ],

                "element_index":
                    int(
                        fault.element_index
                    ),

                "bit_position":
                    int(
                        fault.bit_position
                    ),

                "inference_index":
                    int(
                        fault.inference_index
                    ),

                "persistence":
                    fault.persistence,

                "multiplicity":
                    int(
                        fault.multiplicity
                    ),

                "replicate_index":
                    int(
                        fault.replicate_index
                    ),

                "clean_output_sha256":
                    tensor_bytes_sha256(
                        pair.clean_output
                    ),

                "faulted_output_sha256":
                    tensor_bytes_sha256(
                        pair.faulted_output
                    ),

                "mutation":
                    mutation_dict(
                        pair
                    ),
            })

        weight_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        private_weight_model = build_frozen_ptq_eager(
            fp32_model,
            state_path,
        )

        weight_session = PTQWeightFaultSession(
            private_weight_model,
            weight_state,
        )

        for case in protocol[
            "ptq_cases_per_member"
        ]:
            fault = make_fault_identity(
                protocol=protocol,
                seed=seed,
                fold=fold,
                model_variant="ptq_v7",
                case=case,
            )

            if (
                case[
                    "representation_class"
                ]
                == "int8_persistent_weight"
            ):
                pair = weight_session.run_paired(
                    ptq_eager,
                    x,
                    fault,
                    current_inference_index=0,
                )

                weight_session.reset_clean()

            else:
                pair = run_ptq_activation_buffer_paired(
                    ptq_eager,
                    x,
                    fault,
                    current_inference_index=0,
                )

            require_active_exact_mutation(
                pair
            )

            if pair.input_sha256 != input_sha:
                raise AssertionError(
                    "PTQ paired input hash mismatch"
                )

            if not outputs_bitwise_equal(
                pair.clean_output,
                clean_ptq_eager,
            ):
                raise AssertionError(
                    "PTQ pair clean output changed relative to clean reference"
                )

            rows.append({
                "protocol_version":
                    protocol[
                        "schema_version"
                    ],

                "model_variant":
                    "ptq_v7",

                "checkpoint_seed":
                    seed,

                "fold":
                    fold,

                "checkpoint_sha256":
                    fp32_meta[
                        "checkpoint_sha256"
                    ],

                "ptq_state_sha256":
                    ptq_row[
                        "artifacts"
                    ][
                        "state_dict"
                    ][
                        "sha256"
                    ],

                "ptq_torchscript_sha256":
                    ptq_row[
                        "artifacts"
                    ][
                        "torchscript"
                    ][
                        "sha256"
                    ],

                "partition":
                    "training_calibration",

                "subject_id":
                    int(
                        calibration_identity[
                            "subject"
                        ]
                    ),

                "task_id":
                    int(
                        calibration_identity[
                            "task"
                        ]
                    ),

                "trial_id":
                    int(
                        calibration_identity[
                            "trial"
                        ]
                    ),

                "window_index":
                    int(
                        calibration_identity[
                            "window_index"
                        ]
                    ),

                "label":
                    label,

                "selection_sha256":
                    calibration_identity[
                        "selection_sha256"
                    ],

                "input_sha256":
                    pair.input_sha256,

                "fault_id":
                    pair.fault_id,

                "fault_family":
                    case[
                        "fault_family"
                    ],

                "representation_class":
                    case[
                        "representation_class"
                    ],

                "target_name":
                    case[
                        "target_name"
                    ],

                "target_role":
                    case[
                        "target_role"
                    ],

                "element_index":
                    int(
                        fault.element_index
                    ),

                "bit_position":
                    int(
                        fault.bit_position
                    ),

                "inference_index":
                    int(
                        fault.inference_index
                    ),

                "persistence":
                    fault.persistence,

                "multiplicity":
                    int(
                        fault.multiplicity
                    ),

                "replicate_index":
                    int(
                        fault.replicate_index
                    ),

                "clean_output_sha256":
                    tensor_bytes_sha256(
                        pair.clean_output
                    ),

                "faulted_output_sha256":
                    tensor_bytes_sha256(
                        pair.faulted_output
                    ),

                "mutation":
                    mutation_dict(
                        pair
                    ),
            })

    if len(
        member_checks
    ) != 15:
        raise AssertionError(
            "expected 15 model-estate clean checks"
        )

    if len(
        rows
    ) != 105:
        raise AssertionError(
            f"expected 105 paired cases, got {len(rows)}"
        )

    fp32_count = sum(
        row[
            "model_variant"
        ]
        == "fp32"
        for row in rows
    )

    ptq_count = sum(
        row[
            "model_variant"
        ]
        == "ptq_v7"
        for row in rows
    )

    if fp32_count != 30:
        raise AssertionError(
            f"expected 30 FP32 cases, got {fp32_count}"
        )

    if ptq_count != 75:
        raise AssertionError(
            f"expected 75 PTQ cases, got {ptq_count}"
        )

    fault_ids = [
        row[
            "fault_id"
        ]
        for row in rows
    ]

    if len(
        fault_ids
    ) != len(
        set(
            fault_ids
        )
    ):
        raise AssertionError(
            "qualification fault IDs are not unique"
        )

    for row in rows:
        mutation = row[
            "mutation"
        ]

        if mutation[
            "changed_indices"
        ] != [
            0
        ]:
            raise AssertionError(
                "qualification case changed unexpected element"
            )

        if row[
            "element_index"
        ] != 0:
            raise AssertionError(
                "qualification element index drifted"
            )

        if row[
            "bit_position"
        ] != 0:
            raise AssertionError(
                "qualification bit position drifted"
            )

        if (
            row[
                "persistence"
            ]
            != "transient_one_inference"
        ):
            raise AssertionError(
                "qualification persistence drifted"
            )

    qint8_rows = [
        row
        for row in rows
        if row[
            "representation_class"
        ]
        == "int8_persistent_weight"
    ]

    quint8_rows = [
        row
        for row in rows
        if row[
            "representation_class"
        ]
        in {
            "quantized_activation",
            "quantized_buffer",
        }
    ]

    if len(
        qint8_rows
    ) != 15:
        raise AssertionError(
            "expected 15 qint8 weight cases"
        )

    if len(
        quint8_rows
    ) != 30:
        raise AssertionError(
            "expected 30 quint8 activation/buffer cases"
        )

    if any(
        row[
            "mutation"
        ][
            "tensor_dtype"
        ]
        != "torch.qint8"
        for row in qint8_rows
    ):
        raise AssertionError(
            "qint8 weight representation mismatch"
        )

    if any(
        row[
            "mutation"
        ][
            "tensor_dtype"
        ]
        != "torch.quint8"
        for row in quint8_rows
    ):
        raise AssertionError(
            "quint8 activation/buffer representation mismatch"
        )

    artifact_hashes_after = source_artifact_ledger(
        fp32,
        ptq,
    )

    if (
        artifact_hashes_before
        != artifact_hashes_after
    ):
        raise AssertionError(
            "source artifact hash changed during qualification"
        )

    output_path = Path(
        args.output
    )

    success_path = Path(
        args.success
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "schema_version":
            "phase5c_compute_fi_devcal_runner_qualification_v1",

        "status":
            "QUALIFIED_REAL_TRAINING_CALIBRATION_PAIRED_RUNNER",

        "qualification_date":
            "2026-10-05",

        "evidence_tier":
            "P0",

        "protocol": {
            "path":
                str(
                    protocol_path
                ),

            "sha256":
                file_sha(
                    protocol_path
                ),
        },

        "partition":
            "training_calibration",

        "calibration_identity": {
            "path":
                str(
                    Path(
                        args.calibration
                    )
                ),

            "sha256":
                file_sha(
                    args.calibration
                ),

            "windows_per_fold":
                4096,

            "identity_rule":
                "first lexicographic frozen calibration identity per fold",

            "selected_by_fold": {
                str(
                    fold
                ): {
                    "subject":
                        int(
                            calibration[
                                fold
                            ][
                                "subject"
                            ]
                        ),

                    "task":
                        int(
                            calibration[
                                fold
                            ][
                                "task"
                            ]
                        ),

                    "trial":
                        int(
                            calibration[
                                fold
                            ][
                                "trial"
                            ]
                        ),

                    "window_index":
                        int(
                            calibration[
                                fold
                            ][
                                "window_index"
                            ]
                        ),

                    "label":
                        str(
                            calibration[
                                fold
                            ][
                                "label"
                            ]
                        ),

                    "selection_sha256":
                        calibration[
                            fold
                        ][
                            "selection_sha256"
                        ],
                }
                for fold in range(
                    1,
                    6,
                )
            },
        },

        "qualification_counts": {
            "model_members":
                15,

            "unique_fold_calibration_inputs":
                5,

            "member_clean_checks":
                len(
                    member_checks
                ),

            "fp32_paired_cases":
                fp32_count,

            "ptq_paired_cases":
                ptq_count,

            "qint8_weight_cases":
                len(
                    qint8_rows
                ),

            "quint8_activation_buffer_cases":
                len(
                    quint8_rows
                ),

            "paired_cases_total":
                len(
                    rows
                ),

            "unique_fault_ids":
                len(
                    set(
                        fault_ids
                    )
                ),
        },

        "member_clean_checks":
            member_checks,

        "paired_case_records":
            rows,

        "partition_rejection": {
            "validation":
                rejected[
                    "validation"
                ],

            "outer_test":
                rejected[
                    "outer_test"
                ],

            "onfield":
                rejected[
                    "onfield"
                ],
        },

        "source_artifact_immutability": {
            "artifact_count":
                len(
                    artifact_hashes_before
                ),

            "before":
                artifact_hashes_before,

            "after":
                artifact_hashes_after,

            "unchanged":
                True,
        },

        "acceptance": {
            "all_15_checkpoint_members_executed":
                True,

            "all_15_ptq_members_executed":
                True,

            "all_five_fold_calibration_identities_loaded":
                True,

            "same_fold_identity_reused_across_three_seeds":
                True,

            "all_ptq_eager_models_match_frozen_torchscript_clean":
                True,

            "all_105_predeclared_pairs_executed":
                True,

            "all_fault_ids_unique":
                True,

            "all_mutations_active":
                True,

            "all_mutations_exact_selected_element":
                True,

            "all_mutations_exact_selected_bit":
                True,

            "qint8_weight_representation_preserved":
                True,

            "quint8_activation_buffer_representation_preserved":
                True,

            "quantization_metadata_preserved":
                True,

            "source_artifacts_unchanged":
                True,

            "qualification_rejects_validation":
                True,

            "qualification_rejects_outer_test":
                True,

            "qualification_rejects_onfield":
                True,
        },

        "scientific_boundary": {
            "real_training_calibration_windows_used":
                True,

            "training_only_calibration_identity_used":
                True,

            "faulted_training_calibration_forward_executed":
                True,

            "raw_logits_recorded":
                False,

            "probabilities_recorded":
                False,

            "accuracy_or_recall_computed":
                False,

            "threshold_applied":
                False,

            "threshold_changed":
                False,

            "fault_family_selected_from_results":
                False,

            "target_selected_from_results":
                False,

            "bit_selected_from_results":
                False,

            "element_selected_from_results":
                False,

            "checkpoint_selected_from_results":
                False,

            "replicate_count_selected_from_results":
                False,

            "validation_used":
                False,

            "outer_test_used":
                False,

            "onfield_used":
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
            "Freeze prospective outer compute-FI sampling/cardinality and reporting protocol before any CC outer-test execution.",
    }

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result_sha = file_sha(
        output_path
    )

    success_path.write_text(
        json.dumps(
            {
                "status":
                    "PASS",

                "qualification_status":
                    result[
                        "status"
                    ],

                "paired_cases_total":
                    105,

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
        "MODEL_MEMBERS_EXECUTED=15"
    )

    print(
        "UNIQUE_CALIBRATION_INPUTS=5"
    )

    print(
        "FP32_PAIRED_CASES=30"
    )

    print(
        "PTQ_PAIRED_CASES=75"
    )

    print(
        "TOTAL_PAIRED_CASES=105"
    )

    print(
        "QINT8_WEIGHT_CASES=15"
    )

    print(
        "QUINT8_ACTIVATION_BUFFER_CASES=30"
    )

    print(
        "UNIQUE_FAULT_IDS=105"
    )

    print(
        "ALL_PTQ_EAGER_TORCHSCRIPT_CLEAN_EQUAL=True"
    )

    print(
        "SOURCE_ARTIFACTS_UNCHANGED=True"
    )

    print(
        "QUALIFICATION_STATUS=",
        result[
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
