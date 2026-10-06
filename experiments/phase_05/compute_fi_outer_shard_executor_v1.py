"""Phase-5G frozen outer compute-FI shard controller v1.

This controller binds the qualified Phase-5F atomic/resume core to the frozen
Phase-5E 732-shard plan.

Current qualification gate:
    outer_execution_enabled = False

Allowed:
    dry-run

Prohibited:
    execute-shard

The dry-run performs only:
- JSON/config validation;
- filesystem metadata enumeration;
- segments.npy HEADER-only inventory reconstruction;
- SHA-256 hashing of frozen model artifacts;
- plan/shard/cardinality/replay validation.

It does not call np.load, torch.load, model.forward, or fault operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compute_fi_outer_execution_plan_v1 import build_plan


SCHEMA = "phase5g_compute_fi_outer_shard_executor_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text()
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def validate_dependency_hashes(cfg: dict) -> None:
    failures = []

    for name, record in cfg[
        "frozen_dependencies"
    ].items():
        path = Path(
            record[
                "path"
            ]
        )

        if not path.is_file():
            failures.append(
                f"{name}:missing:{path}"
            )
            continue

        observed = sha256_file(
            path
        )

        if observed != record[
            "sha256"
        ]:
            failures.append(
                f"{name}:hash:{observed}"
            )

    if failures:
        raise ValueError(
            "frozen dependency validation failed: "
            + "; ".join(failures)
        )


def load_fp32_artifact_index(
    freeze_path: str | Path,
) -> dict[tuple[int, int], dict]:
    manifest = load_json(
        freeze_path
    )

    rows = manifest[
        "checkpoints"
    ]

    result = {}

    for row in rows:
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
                f"duplicate FP32 member: {key}"
            )

        checkpoint = Path(
            row[
                "checkpoint"
            ]
        )

        expected = row[
            "sha256"
        ]

        observed = sha256_file(
            checkpoint
        )

        if observed != expected:
            raise ValueError(
                f"FP32 checkpoint hash mismatch: {key}"
            )

        result[
            key
        ] = {
            "checkpoint":
                str(
                    checkpoint
                ),

            "sha256":
                expected,
        }

    expected_keys = {
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
        result
    ) != expected_keys:
        raise ValueError(
            "FP32 all15 member set changed"
        )

    return result


def load_ptq_artifact_index(
    manifest_path: str | Path,
) -> dict[tuple[int, int], dict]:
    manifest = load_json(
        manifest_path
    )

    rows = manifest[
        "members"
    ]

    result = {}

    for row in rows:
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
                f"duplicate PTQ member: {key}"
            )

        state = row[
            "artifacts"
        ][
            "state_dict"
        ]

        script = row[
            "artifacts"
        ][
            "torchscript"
        ]

        state_path = Path(
            state[
                "path"
            ]
        )

        script_path = Path(
            script[
                "path"
            ]
        )

        if sha256_file(
            state_path
        ) != state[
            "sha256"
        ]:
            raise ValueError(
                f"PTQ state hash mismatch: {key}"
            )

        if sha256_file(
            script_path
        ) != script[
            "sha256"
        ]:
            raise ValueError(
                f"PTQ TorchScript hash mismatch: {key}"
            )

        result[
            key
        ] = {
            "state_dict":
                {
                    "path":
                        str(
                            state_path
                        ),

                    "sha256":
                        state[
                            "sha256"
                        ],
                },

            "torchscript":
                {
                    "path":
                        str(
                            script_path
                        ),

                    "sha256":
                        script[
                            "sha256"
                        ],
                },
        }

    expected_keys = {
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
        result
    ) != expected_keys:
        raise ValueError(
            "PTQ all15 member set changed"
        )

    return result


def validate_plan_structure(
    plan: dict,
) -> dict[str, int]:
    if (
        plan[
            "status"
        ]
        != "QUALIFIED_FROZEN_DERIVED_EXECUTION_PLAN"
    ):
        raise ValueError(
            "unexpected Phase-5E plan status"
        )

    shards = plan[
        "shards"
    ]

    clean = plan[
        "clean_caches"
    ]

    if len(
        shards
    ) != 732:
        raise ValueError(
            "shard count changed"
        )

    if len(
        clean
    ) != 366:
        raise ValueError(
            "clean-cache count changed"
        )

    shard_ids = [
        row[
            "shard_id"
        ]
        for row in shards
    ]

    clean_ids = [
        row[
            "clean_cache_id"
        ]
        for row in clean
    ]

    if len(
        shard_ids
    ) != len(
        set(
            shard_ids
        )
    ):
        raise ValueError(
            "shard-id collision"
        )

    if len(
        clean_ids
    ) != len(
        set(
            clean_ids
        )
    ):
        raise ValueError(
            "clean-cache-id collision"
        )

    clean_set = set(
        clean_ids
    )

    subject_shards = {}

    fp32_shards = 0
    ptq_shards = 0
    transient_shards = 0
    persistent_shards = 0

    transient_instances = 0
    persistent_instances = 0

    transient_evals = 0
    persistent_evals = 0

    for shard in shards:
        subject = int(
            shard[
                "subject"
            ]
        )

        subject_shards[
            subject
        ] = (
            subject_shards.get(
                subject,
                0,
            )
            + 1
        )

        if shard[
            "clean_cache_id"
        ] not in clean_set:
            raise ValueError(
                "fault shard references unknown clean cache"
            )

        variant = shard[
            "model_variant"
        ]

        if variant == "fp32":
            fp32_shards += 1

            if int(
                shard[
                    "target_count"
                ]
            ) != 10:
                raise ValueError(
                    "FP32 target count changed"
                )

        elif variant == "ptq_v7":
            ptq_shards += 1

            if int(
                shard[
                    "target_count"
                ]
            ) != 14:
                raise ValueError(
                    "PTQ target count changed"
                )

        else:
            raise ValueError(
                f"unknown variant: {variant}"
            )

        persistence = shard[
            "persistence"
        ]

        instances = int(
            shard[
                "expected_outer_instance_ids"
            ]
        )

        evaluations = int(
            shard[
                "expected_faulted_model_window_evaluations"
            ]
        )

        recomputed_instances = sum(
            int(
                row[
                    "outer_instance_count"
                ]
            )
            for row in shard[
                "target_exposure"
            ]
        )

        recomputed_evaluations = sum(
            int(
                row[
                    "faulted_model_window_evaluations"
                ]
            )
            for row in shard[
                "target_exposure"
            ]
        )

        if instances != recomputed_instances:
            raise ValueError(
                f"instance arithmetic mismatch: {shard['shard_id']}"
            )

        if evaluations != recomputed_evaluations:
            raise ValueError(
                f"evaluation arithmetic mismatch: {shard['shard_id']}"
            )

        if persistence == "transient_one_inference":
            transient_shards += 1
            transient_instances += instances
            transient_evals += evaluations

            if instances != evaluations:
                raise ValueError(
                    "transient identity/evaluation count mismatch"
                )

        elif (
            persistence
            == "persistent_from_onset_until_trial_end"
        ):
            persistent_shards += 1
            persistent_instances += instances
            persistent_evals += evaluations

            if evaluations < instances:
                raise ValueError(
                    "persistent exposure less than identity count"
                )

        else:
            raise ValueError(
                f"unknown persistence: {persistence}"
            )

    if set(
        subject_shards.values()
    ) != {
        12
    }:
        raise ValueError(
            "each subject must have exactly 12 shards"
        )

    clean_evals = sum(
        int(
            row[
                "expected_clean_model_window_evaluations"
            ]
        )
        for row in clean
    )

    result = {
        "shard_count":
            len(
                shards
            ),

        "clean_cache_count":
            len(
                clean
            ),

        "fp32_shards":
            fp32_shards,

        "ptq_shards":
            ptq_shards,

        "transient_shards":
            transient_shards,

        "persistent_shards":
            persistent_shards,

        "transient_outer_instance_ids":
            transient_instances,

        "persistent_outer_instance_ids":
            persistent_instances,

        "total_outer_instance_ids":
            (
                transient_instances
                + persistent_instances
            ),

        "clean_model_window_evaluations":
            clean_evals,

        "transient_faulted_model_window_evaluations":
            transient_evals,

        "persistent_faulted_model_window_evaluations":
            persistent_evals,

        "total_faulted_model_window_evaluations":
            (
                transient_evals
                + persistent_evals
            ),

        "total_model_window_evaluations_including_clean":
            (
                clean_evals
                + transient_evals
                + persistent_evals
            ),
    }

    expected = plan[
        "expected_totals"
    ]

    for key in (
        "transient_outer_instance_ids",
        "persistent_outer_instance_ids",
        "total_outer_instance_ids",
        "clean_model_window_evaluations",
        "transient_faulted_model_window_evaluations",
        "persistent_faulted_model_window_evaluations",
        "total_faulted_model_window_evaluations",
        "total_model_window_evaluations_including_clean",
    ):
        if result[
            key
        ] != int(
            expected[
                key
            ]
        ):
            raise ValueError(
                f"plan total changed: {key}"
            )

    return result


def validate_shard_model_bindings(
    plan: dict,
    fp32_index: dict,
    ptq_index: dict,
) -> None:
    for shard in plan[
        "shards"
    ]:
        key = (
            int(
                shard[
                    "checkpoint_seed"
                ]
            ),
            int(
                shard[
                    "fold"
                ]
            ),
        )

        if shard[
            "model_variant"
        ] == "fp32":
            if key not in fp32_index:
                raise ValueError(
                    f"missing FP32 model binding: {key}"
                )

        elif shard[
            "model_variant"
        ] == "ptq_v7":
            if key not in ptq_index:
                raise ValueError(
                    f"missing PTQ model binding: {key}"
                )

        else:
            raise ValueError(
                "unknown shard model variant"
            )


def dry_run(
    cfg_path: str | Path,
    output_path: str | Path,
) -> dict:
    cfg_path = Path(
        cfg_path
    )

    cfg = load_json(
        cfg_path
    )

    if cfg[
        "status"
    ] != "FROZEN_DRY_RUN_QUALIFICATION":
        raise ValueError(
            "executor config is not frozen for dry-run qualification"
        )

    if cfg[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ] is not False:
        raise ValueError(
            "outer execution gate must remain false during dry-run qualification"
        )

    validate_dependency_hashes(
        cfg
    )

    plan_path = Path(
        cfg[
            "frozen_dependencies"
        ][
            "phase5e_plan"
        ][
            "path"
        ]
    )

    plan = load_json(
        plan_path
    )

    plan_summary = validate_plan_structure(
        plan
    )

    fp32_index = load_fp32_artifact_index(
        cfg[
            "frozen_dependencies"
        ][
            "fp32_freeze"
        ][
            "path"
        ]
    )

    ptq_index = load_ptq_artifact_index(
        cfg[
            "frozen_dependencies"
        ][
            "ptq_all15"
        ][
            "path"
        ]
    )

    validate_shard_model_bindings(
        plan,
        fp32_index,
        ptq_index,
    )

    # Metadata/header-only deterministic replay of Phase-5E plan.
    rederived = build_plan(
        protocol_path=cfg[
            "frozen_dependencies"
        ][
            "phase5d_protocol"
        ][
            "path"
        ],
        split_path=cfg[
            "frozen_dependencies"
        ][
            "primary_split"
        ][
            "path"
        ],
        dataset_root=cfg[
            "execution_locations"
        ][
            "dataset_root"
        ],
    )

    for key in (
        "inventory",
        "sharding",
        "clean_cache",
        "expected_totals",
        "trial_inventory",
        "subject_inventory",
        "clean_caches",
        "shards",
    ):
        if rederived[
            key
        ] != plan[
            key
        ]:
            raise ValueError(
                f"metadata-only plan replay mismatch: {key}"
            )

    result = {
        "schema_version":
            "phase5g_compute_fi_outer_shard_executor_dry_run_result_v1",

        "status":
            "QUALIFIED_732_SHARD_DRY_RUN",

        "executor_config_sha256":
            sha256_file(
                cfg_path
            ),

        "executor_implementation_sha256":
            sha256_file(
                __file__
            ),

        "phase5e_plan_sha256":
            sha256_file(
                plan_path
            ),

        "phase5e_plan_metadata_replay":
            "EXACT",

        "artifact_hash_checks": {
            "fp32_checkpoints":
                len(
                    fp32_index
                ),

            "ptq_state_dicts":
                len(
                    ptq_index
                ),

            "ptq_torchscript":
                len(
                    ptq_index
                ),

            "total_model_artifacts":
                (
                    len(
                        fp32_index
                    )
                    + 2
                    * len(
                        ptq_index
                    )
                ),
        },

        "plan_summary":
            plan_summary,

        "inventory_bindings":
            {
                "trial_inventory_sha256":
                    plan[
                        "inventory"
                    ][
                        "trial_inventory_sha256"
                    ],

                "subject_inventory_sha256":
                    plan[
                        "inventory"
                    ][
                        "subject_inventory_sha256"
                    ],

                "persistent_onset_binding_sha256":
                    plan[
                        "inventory"
                    ][
                        "persistent_onset_binding_sha256"
                    ],
            },

        "execution_gate":
            {
                "outer_execution_enabled":
                    False,

                "execute_shard_permitted":
                    False,
            },

        "scientific_boundary":
            {
                "segments_npy_header_bytes_read":
                    True,

                "segments_npy_payload_bytes_read":
                    False,

                "labels_npy_opened":
                    False,

                "np_load_used":
                    False,

                "torch_load_used":
                    False,

                "model_loaded":
                    False,

                "model_forward_executed":
                    False,

                "fault_operator_invoked":
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

                "CC_result_generated":
                    False,

                "CSC_result_generated":
                    False,
            },
    }

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def execute_shard(
    cfg_path: str | Path,
    shard_id: str,
) -> None:
    """Outer execution entry point, deliberately hard-gated in Phase 5G."""

    cfg = load_json(
        cfg_path
    )

    if not cfg[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ]:
        raise RuntimeError(
            "outer execute-shard gate is FALSE; "
            "Phase-5G qualification permits dry-run only"
        )

    raise RuntimeError(
        "outer execution gate must be enabled by a later governed freeze "
        "after shard-style sequence qualification"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    dry = sub.add_parser(
        "dry-run"
    )

    dry.add_argument(
        "--output",
        required=True,
    )

    execute = sub.add_parser(
        "execute-shard"
    )

    execute.add_argument(
        "--shard-id",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "dry-run":
        result = dry_run(
            args.config,
            args.output,
        )

        print(
            "PHASE5G_DRY_RUN_STATUS=",
            result[
                "status"
            ],
            sep="",
        )

        for key, value in result[
            "plan_summary"
        ].items():
            print(
                f"{key.upper()}={value}"
            )

        print(
            "MODEL_ARTIFACTS_HASHED=",
            result[
                "artifact_hash_checks"
            ][
                "total_model_artifacts"
            ],
            sep="",
        )

        print(
            "PLAN_METADATA_REPLAY=",
            result[
                "phase5e_plan_metadata_replay"
            ],
            sep="",
        )

        print(
            "OUTER_EXECUTION_ENABLED=False"
        )

        return

    execute_shard(
        args.config,
        args.shard_id,
    )


if __name__ == "__main__":
    main()
