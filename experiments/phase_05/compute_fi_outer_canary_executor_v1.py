"""Prospective Phase-5 outer canary executor v1.

IMPORTANT:
- `validate-config` and `validate-shard` never read outer signal arrays,
  load model state, execute a model, or invoke fault operators.
- `execute-canary` is the only command allowed to cross the frozen outer
  execution boundary.
- This executor accepts exactly the single Phase-5L-authorized shard.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_execution_harness import (
    FP32_BOUNDARY_TO_MODULE,
    load_frozen_fp32_model,
    tensor_bytes_sha256,
)
from compute_fi_fault_only_execution_v1 import (
    run_fp32_fault_only,
)
from compute_fi_outer_executor_v1 import (
    artifact_is_reusable,
    clean_output_record,
    commit_atomic_artifact,
    prepare_atomic_artifact,
)
from compute_fi_outer_sampling_v1 import (
    canonical_sampling_payload,
    derive_bit_position,
    derive_element_index,
    outer_instance_id,
    sampling_instance_id,
    transient_inference_index,
)


EXPECTED_PHASE5L_CONFIG_SHA256 = (
    "b169e2997499dfb75927938d98d39018"
    "d01f60a93e04148d2b4fcb70f30692be"
)

EXPECTED_PHASE5E_PLAN_SHA256 = (
    "95aecd14b4aa8dce70492f0c4d6d50a"
    "8bf5a8dafb2ec962aff9c033b8472bb54"
)

EXPECTED_PHASE5F_CORE_SHA256 = (
    "77ee323f71a5dbf4c54cc454a5911849"
    "aa45c5d4bdcc6d63bb908d8cece363d0"
)

EXPECTED_PHASE5K_FAULT_ONLY_SHA256 = (
    "f8bb4095bbfedc69c24fa4cc6e79ccf"
    "4914e14fa502ce080a06de01e740112cb"
)

PHASE5A_PROTOCOL = (
    "phase5a_compute_fi_representation_protocol_v1"
)

AUTHORIZED_SHARD_ID = (
    "p5e-o1-f5-s009-fp32-seed42-transient-"
    "6d97ac3095dc8dc9"
)

AUTHORIZED_CLEAN_CACHE_ID = (
    "p5e-c0-f5-s009-fp32-seed42-"
    "e19b32428112a04b"
)

EXPECTED_TRIAL_COUNT = 44
EXPECTED_WINDOW_COUNT = 2481
EXPECTED_TARGET_COUNT = 10
EXPECTED_CLEAN_FORWARD_COUNT = 2481
EXPECTED_FAULT_FORWARD_COUNT = 24810
EXPECTED_TOTAL_FORWARD_COUNT = 27291

CLEAN_ARTIFACT_KIND = "clean_cache"
FAULT_ARTIFACT_KIND = "fault_shard"

CLEAN_JSONL = "clean_outputs.jsonl"
FAULT_JSONL = "fault_outputs.jsonl"
METADATA_JSON = "metadata.json"


def sha256_file(
    path: str | Path,
) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: str | Path,
) -> dict:
    return json.loads(
        Path(path).read_text()
    )


def write_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: str | Path,
    rows: list[Mapping[str, Any]],
) -> None:
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            handle.write(
                "\n"
            )


def _dependency_record(
    cfg: dict,
    key: str,
) -> dict:
    record = cfg[
        "frozen_dependencies"
    ][
        key
    ]

    path = Path(
        record[
            "path"
        ]
    )

    if not path.is_file():
        raise ValueError(
            f"missing frozen dependency {key}: {path}"
        )

    observed = sha256_file(
        path
    )

    if observed != record[
        "sha256"
    ]:
        raise ValueError(
            f"frozen dependency hash mismatch "
            f"{key}: {observed}"
        )

    return record


def _trial_rows_for_canary(
    plan: dict,
    cfg: dict,
) -> list[dict]:
    shard = cfg[
        "canary"
    ][
        "shard"
    ]

    subject = int(
        shard[
            "subject"
        ]
    )

    fold = int(
        shard[
            "fold"
        ]
    )

    rows = [
        dict(
            row
        )
        for row in plan[
            "trial_inventory"
        ]
        if int(
            row[
                "subject"
            ]
        )
        == subject
        and int(
            row[
                "fold"
            ]
        )
        == fold
    ]

    rows.sort(
        key=lambda row: (
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
        )
    )

    if len(
        rows
    ) != EXPECTED_TRIAL_COUNT:
        raise ValueError(
            "canary trial count changed"
        )

    total_windows = sum(
        int(
            row[
                "window_count"
            ]
        )
        for row in rows
    )

    if total_windows != EXPECTED_WINDOW_COUNT:
        raise ValueError(
            "canary window count changed"
        )

    return rows


def validate_configuration(
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(
        config_path
    )

    observed_config_sha = sha256_file(
        config_path
    )

    if (
        observed_config_sha
        != EXPECTED_PHASE5L_CONFIG_SHA256
    ):
        raise ValueError(
            "Phase-5L config hash mismatch"
        )

    cfg = load_json(
        config_path
    )

    if (
        cfg[
            "status"
        ]
        != "FROZEN_HASH_BOUND_CANARY_ONLY_OUTER_EXECUTION_CONFIG"
    ):
        raise ValueError(
            "unexpected Phase-5L config status"
        )

    authorization = cfg[
        "authorization"
    ]

    if (
        authorization[
            "canary_execution_authorized"
        ]
        is not True
    ):
        raise ValueError(
            "canary is not authorized"
        )

    if (
        authorization[
            "canary_execution_performed"
        ]
        is not False
    ):
        raise ValueError(
            "Phase-5L freeze must predate execution"
        )

    if (
        authorization[
            "full_fleet_execution_authorized"
        ]
        is not False
    ):
        raise ValueError(
            "full-fleet authorization is prohibited"
        )

    if int(
        authorization[
            "authorized_shard_count"
        ]
    ) != 1:
        raise ValueError(
            "exactly one shard must be authorized"
        )

    if int(
        authorization[
            "authorized_clean_cache_count"
        ]
    ) != 1:
        raise ValueError(
            "exactly one clean cache must be authorized"
        )

    if (
        authorization[
            "authorized_shard_id"
        ]
        != AUTHORIZED_SHARD_ID
    ):
        raise ValueError(
            "authorized shard changed"
        )

    if (
        authorization[
            "authorized_clean_cache_id"
        ]
        != AUTHORIZED_CLEAN_CACHE_ID
    ):
        raise ValueError(
            "authorized clean cache changed"
        )

    # Validate every frozen dependency before any execution boundary.
    for key in cfg[
        "frozen_dependencies"
    ]:
        _dependency_record(
            cfg,
            key,
        )

    plan_record = cfg[
        "frozen_dependencies"
    ][
        "phase5e_plan"
    ]

    if (
        plan_record[
            "sha256"
        ]
        != EXPECTED_PHASE5E_PLAN_SHA256
    ):
        raise ValueError(
            "Phase-5E plan binding changed"
        )

    core_record = cfg[
        "frozen_dependencies"
    ][
        "phase5f_atomic_resume_core"
    ]

    if (
        core_record[
            "sha256"
        ]
        != EXPECTED_PHASE5F_CORE_SHA256
    ):
        raise ValueError(
            "Phase-5F core binding changed"
        )

    fault_only_record = cfg[
        "frozen_dependencies"
    ][
        "phase5k_fault_only_module"
    ]

    if (
        fault_only_record[
            "sha256"
        ]
        != EXPECTED_PHASE5K_FAULT_ONLY_SHA256
    ):
        raise ValueError(
            "Phase-5K fault-only binding changed"
        )

    plan = load_json(
        plan_record[
            "path"
        ]
    )

    if (
        plan[
            "status"
        ]
        != "QUALIFIED_FROZEN_DERIVED_EXECUTION_PLAN"
    ):
        raise ValueError(
            "unexpected Phase-5E plan status"
        )

    if len(
        plan[
            "shards"
        ]
    ) != 732:
        raise ValueError(
            "Phase-5E shard count changed"
        )

    if len(
        plan[
            "clean_caches"
        ]
    ) != 366:
        raise ValueError(
            "Phase-5E clean-cache count changed"
        )

    frozen_shard = plan[
        "shards"
    ][
        0
    ]

    if (
        frozen_shard
        != cfg[
            "canary"
        ][
            "shard"
        ]
    ):
        raise ValueError(
            "canary no longer equals Phase-5E shard zero"
        )

    if (
        frozen_shard[
            "shard_id"
        ]
        != AUTHORIZED_SHARD_ID
    ):
        raise ValueError(
            "Phase-5E shard-zero ID changed"
        )

    clean = cfg[
        "canary"
    ][
        "clean_cache"
    ]

    if (
        clean[
            "clean_cache_id"
        ]
        != AUTHORIZED_CLEAN_CACHE_ID
    ):
        raise ValueError(
            "clean cache ID changed"
        )

    if int(
        frozen_shard[
            "subject"
        ]
    ) != 9:
        raise ValueError(
            "canary subject changed"
        )

    if int(
        frozen_shard[
            "fold"
        ]
    ) != 5:
        raise ValueError(
            "canary fold changed"
        )

    if (
        frozen_shard[
            "model_variant"
        ]
        != "fp32"
    ):
        raise ValueError(
            "canary model variant changed"
        )

    if int(
        frozen_shard[
            "checkpoint_seed"
        ]
    ) != 42:
        raise ValueError(
            "canary seed changed"
        )

    if (
        frozen_shard[
            "persistence"
        ]
        != "transient_one_inference"
    ):
        raise ValueError(
            "canary persistence changed"
        )

    if int(
        frozen_shard[
            "target_count"
        ]
    ) != EXPECTED_TARGET_COUNT:
        raise ValueError(
            "canary target count changed"
        )

    targets = cfg[
        "canary"
    ][
        "target_inventory"
    ]

    if len(
        targets
    ) != EXPECTED_TARGET_COUNT:
        raise ValueError(
            "target inventory count changed"
        )

    target_names = [
        row[
            "target_name"
        ]
        for row in targets
    ]

    if target_names != frozen_shard[
        "target_names"
    ]:
        raise ValueError(
            "target ordering changed"
        )

    missing_hook_targets = [
        name
        for name in target_names
        if name
        not in FP32_BOUNDARY_TO_MODULE
    ]

    if missing_hook_targets:
        raise ValueError(
            "FP32 hook mapping missing targets: "
            + repr(
                missing_hook_targets
            )
        )

    if any(
        "fp32"
        not in row[
            "model_variants"
        ]
        for row in targets
    ):
        raise ValueError(
            "non-FP32 target present in canary inventory"
        )

    budget = cfg[
        "forward_budget"
    ]

    if int(
        budget[
            "clean_model_window_evaluations"
        ]
    ) != EXPECTED_CLEAN_FORWARD_COUNT:
        raise ValueError(
            "clean forward budget changed"
        )

    if int(
        budget[
            "faulted_model_window_evaluations"
        ]
    ) != EXPECTED_FAULT_FORWARD_COUNT:
        raise ValueError(
            "fault forward budget changed"
        )

    if int(
        budget[
            "total_model_window_evaluations"
        ]
    ) != EXPECTED_TOTAL_FORWARD_COUNT:
        raise ValueError(
            "total forward budget changed"
        )

    if (
        budget[
            "embedded_clean_reference_forward_per_fault"
        ]
        is not False
    ):
        raise ValueError(
            "embedded clean forward must remain prohibited"
        )

    checkpoint = cfg[
        "canary"
    ][
        "checkpoint"
    ]

    checkpoint_path = Path(
        checkpoint[
            "path"
        ]
    )

    if not checkpoint_path.is_file():
        raise ValueError(
            "frozen canary checkpoint missing"
        )

    if (
        sha256_file(
            checkpoint_path
        )
        != checkpoint[
            "sha256"
        ]
    ):
        raise ValueError(
            "frozen canary checkpoint hash mismatch"
        )

    trial_rows = _trial_rows_for_canary(
        plan,
        cfg,
    )

    return {
        "config":
            cfg,

        "plan":
            plan,

        "trial_rows":
            trial_rows,

        "config_sha256":
            observed_config_sha,

        "plan_sha256":
            plan_record[
                "sha256"
            ],
    }


def validate_requested_shard(
    validated: Mapping[str, Any],
    shard_id: str,
) -> dict:
    cfg = validated[
        "config"
    ]

    if shard_id != AUTHORIZED_SHARD_ID:
        raise ValueError(
            "requested shard is not authorized by Phase 5L"
        )

    if (
        shard_id
        != cfg[
            "authorization"
        ][
            "authorized_shard_id"
        ]
    ):
        raise ValueError(
            "requested shard/config mismatch"
        )

    shard = cfg[
        "canary"
    ][
        "shard"
    ]

    if shard[
        "shard_id"
    ] != shard_id:
        raise ValueError(
            "requested shard/canary binding mismatch"
        )

    return dict(
        shard
    )


def _parent_record(
    *,
    row: Mapping[str, Any],
    window_index: int,
) -> dict[str, Any]:
    return {
        "partition":
            "outer_test",

        "fold":
            int(
                row[
                    "fold"
                ]
            ),

        "canonical_subject":
            row[
                "canonical_subject"
            ],

        "subject":
            int(
                row[
                    "subject"
                ]
            ),

        "task":
            int(
                row[
                    "task"
                ]
            ),

        "trial":
            int(
                row[
                    "trial"
                ]
            ),

        "window_index":
            int(
                window_index
            ),
    }


def _load_trial_signal(
    dataset_root: str | Path,
    row: Mapping[str, Any],
) -> np.ndarray:
    """Execution-boundary function. This is the only outer array loader."""

    path = (
        Path(
            dataset_root
        )
        / str(
            int(
                row[
                    "subject"
                ]
            )
        )
        / str(
            int(
                row[
                    "task"
                ]
            )
        )
        / str(
            int(
                row[
                    "trial"
                ]
            )
        )
        / "segments.npy"
    )

    if not path.is_file():
        raise ValueError(
            f"missing outer segments file: {path}"
        )

    windows = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = (
        int(
            row[
                "window_count"
            ]
        ),
        30,
        9,
    )

    if tuple(
        windows.shape
    ) != expected_shape:
        raise ValueError(
            f"outer trial shape mismatch: "
            f"path={path} shape={windows.shape} "
            f"expected={expected_shape}"
        )

    return windows


def _window_tensor(
    trial_windows: np.ndarray,
    window_index: int,
) -> torch.Tensor:
    array = np.asarray(
        trial_windows[
            int(
                window_index
            )
        ],
        dtype=np.float32,
    )

    return torch.from_numpy(
        array
    ).unsqueeze(
        0
    )


def _float_token(
    value: float,
) -> float | dict[str, str]:
    value = float(
        value
    )

    if math.isnan(
        value
    ):
        return {
            "nonfinite":
                "nan",
        }

    if math.isinf(
        value
    ):
        return {
            "nonfinite":
                (
                    "+inf"
                    if value > 0
                    else "-inf"
                ),
        }

    return value


def _tensor_values(
    tensor: torch.Tensor,
) -> list[
    float
    | dict[str, str]
]:
    flat = (
        tensor
        .detach()
        .cpu()
        .reshape(
            -1
        )
    )

    return [
        _float_token(
            value.item()
        )
        for value in flat
    ]


def _tensor_float32_hex(
    tensor: torch.Tensor,
) -> list[str]:
    flat = (
        tensor
        .detach()
        .cpu()
        .to(
            torch.float32
        )
        .reshape(
            -1
        )
    )

    return [
        struct.pack(
            "<f",
            float(
                value.item()
            ),
        ).hex()
        for value in flat
    ]


def _tensor_has_nonfinite(
    tensor: torch.Tensor,
) -> bool:
    return bool(
        (
            ~torch.isfinite(
                tensor
            )
        )
        .any()
        .item()
    )


def _probability_values(
    output: torch.Tensor,
) -> list[
    float
    | dict[str, str]
]:
    probabilities = torch.softmax(
        output.to(
            torch.float32
        ),
        dim=-1,
    )

    return _tensor_values(
        probabilities
    )


def _mutation_json(
    mutation,
) -> dict[str, Any]:
    return dataclasses.asdict(
        mutation
    )


def _fault_identity(
    *,
    cfg: dict,
    target: Mapping[str, Any],
    row: Mapping[str, Any],
    window_index: int,
) -> tuple[
    FaultIdentity,
    str,
    str,
]:
    shard = cfg[
        "canary"
    ][
        "shard"
    ]

    payload = canonical_sampling_payload(
        partition="outer_test",
        fold=int(
            row[
                "fold"
            ]
        ),
        subject=int(
            row[
                "subject"
            ]
        ),
        task=int(
            row[
                "task"
            ]
        ),
        trial=int(
            row[
                "trial"
            ]
        ),
        parent_kind="window",
        window_index=int(
            window_index
        ),
        representation_class=target[
            "representation_class"
        ],
        target_name=target[
            "target_name"
        ],
        target_role=target[
            "target_role"
        ],
        persistence=shard[
            "persistence"
        ],
        replicate_index=0,
    )

    sid = sampling_instance_id(
        payload
    )

    element = derive_element_index(
        payload,
        target_numel=int(
            target[
                "target_numel_per_inference"
            ]
        ),
    )

    bit = derive_bit_position(
        payload,
        eligible_bits=target[
            "eligible_bit_positions"
        ],
    )

    inference_index = transient_inference_index(
        payload
    )

    identity = FaultIdentity(
        protocol=
            PHASE5A_PROTOCOL,

        model_variant=
            shard[
                "model_variant"
            ],

        checkpoint_seed=
            int(
                shard[
                    "checkpoint_seed"
                ]
            ),

        fold=
            int(
                shard[
                    "fold"
                ]
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
            int(
                element
            ),

        bit_position=
            int(
                bit
            ),

        inference_index=
            int(
                inference_index
            ),

        persistence=
            shard[
                "persistence"
            ],

        multiplicity=
            1,

        replicate_index=
            0,
    )

    validate_fault_identity(
        identity
    )

    fault_id = identity.fault_id()

    execution_id = outer_instance_id(
        sampling_instance_id_value=sid,
        phase5a_fault_id=fault_id,
        model_variant=identity.model_variant,
        checkpoint_seed=identity.checkpoint_seed,
    )

    return (
        identity,
        sid,
        execution_id,
    )


def _fault_output_record(
    *,
    cfg: dict,
    parent: Mapping[str, Any],
    identity: FaultIdentity,
    sampling_instance_id_value: str,
    execution_id: str,
    input_tensor: torch.Tensor,
    execution,
) -> dict[str, Any]:
    output = execution.faulted_output

    return {
        "schema_version":
            "phase5m_compute_fi_fault_only_outer_record_v1",

        "shard_id":
            cfg[
                "authorization"
            ][
                "authorized_shard_id"
            ],

        "clean_cache_id":
            cfg[
                "authorization"
            ][
                "authorized_clean_cache_id"
            ],

        "outer_instance_id":
            execution_id,

        "sampling_instance_id":
            sampling_instance_id_value,

        "phase5a_fault_id":
            identity.fault_id(),

        "model_variant":
            identity.model_variant,

        "checkpoint_seed":
            int(
                identity.checkpoint_seed
            ),

        "fold":
            int(
                identity.fold
            ),

        "fault_family":
            identity.fault_family,

        "representation_class":
            identity.representation_class,

        "target_name":
            identity.target_name,

        "target_role":
            identity.target_role,

        "element_index":
            int(
                identity.element_index
            ),

        "bit_position":
            int(
                identity.bit_position
            ),

        "onset_or_inference_index":
            int(
                identity.inference_index
            ),

        "persistence":
            identity.persistence,

        "multiplicity":
            int(
                identity.multiplicity
            ),

        "replicate_index":
            int(
                identity.replicate_index
            ),

        "parent":
            dict(
                parent
            ),

        "input_sha256":
            tensor_bytes_sha256(
                input_tensor
            ),

        "faulted_output_sha256":
            tensor_bytes_sha256(
                output
            ),

        "faulted_output_nonfinite":
            _tensor_has_nonfinite(
                output
            ),

        "faulted_output_values":
            _tensor_values(
                output
            ),

        "faulted_output_float32_hex":
            _tensor_float32_hex(
                output
            ),

        "faulted_softmax_values":
            _probability_values(
                output
            ),

        "mutation":
            _mutation_json(
                execution.mutation
            ),
    }


def _clean_record(
    *,
    cfg: dict,
    parent: Mapping[str, Any],
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> dict[str, Any]:
    shard = cfg[
        "canary"
    ][
        "shard"
    ]

    base = clean_output_record(
        clean_cache_id=cfg[
            "authorization"
        ][
            "authorized_clean_cache_id"
        ],
        parent=parent,
        model_variant=shard[
            "model_variant"
        ],
        checkpoint_seed=int(
            shard[
                "checkpoint_seed"
            ]
        ),
        fold=int(
            shard[
                "fold"
            ]
        ),
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )

    base[
        "clean_output_values"
    ] = _tensor_values(
        output_tensor
    )

    base[
        "clean_output_float32_hex"
    ] = _tensor_float32_hex(
        output_tensor
    )

    base[
        "clean_softmax_values"
    ] = _probability_values(
        output_tensor
    )

    return base


def _hash_output_files(
    temp_dir: Path,
    relative_paths: list[str],
) -> dict[str, str]:
    return {
        relative:
            sha256_file(
                temp_dir
                / relative
            )
        for relative in relative_paths
    }


def execute_canary(
    *,
    config_path: str | Path,
    shard_id: str,
    dataset_root: str | Path,
    output_root: str | Path,
    recompute_partial: bool,
) -> dict[str, Any]:
    """Execute exactly the one frozen Phase-5L canary.

    Calling this function crosses the outer execution boundary.
    """

    validated = validate_configuration(
        config_path
    )

    shard = validate_requested_shard(
        validated,
        shard_id,
    )

    cfg = validated[
        "config"
    ]

    plan_sha = validated[
        "plan_sha256"
    ]

    executor_sha = sha256_file(
        Path(
            __file__
        )
    )

    output_root = Path(
        output_root
    )

    configured_root = Path(
        cfg[
            "prospective_result_root"
        ]
    )

    if (
        output_root.resolve()
        != configured_root.resolve()
    ):
        raise ValueError(
            "output root differs from frozen Phase-5L prospective root"
        )

    clean_id = cfg[
        "authorization"
    ][
        "authorized_clean_cache_id"
    ]

    fault_id = cfg[
        "authorization"
    ][
        "authorized_shard_id"
    ]

    clean_reusable = artifact_is_reusable(
        output_root=output_root,
        artifact_id=clean_id,
        artifact_kind=CLEAN_ARTIFACT_KIND,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
    )

    fault_reusable = artifact_is_reusable(
        output_root=output_root,
        artifact_id=fault_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
    )

    if fault_reusable and not clean_reusable:
        raise ValueError(
            "fault shard exists without reusable clean cache"
        )

    if clean_reusable and fault_reusable:
        return {
            "status":
                "REUSED_COMPLETE_CANARY",

            "clean_cache_reused":
                True,

            "fault_shard_reused":
                True,

            "new_clean_forwards":
                0,

            "new_fault_forwards":
                0,
        }

    fp32_freeze = cfg[
        "frozen_dependencies"
    ][
        "fp32_freeze_manifest"
    ][
        "path"
    ]

    model, model_meta = load_frozen_fp32_model(
        Path(
            fp32_freeze
        ),
        seed=int(
            shard[
                "checkpoint_seed"
            ]
        ),
        fold=int(
            shard[
                "fold"
            ]
        ),
    )

    model.eval()

    if (
        model_meta[
            "checkpoint_sha256"
        ]
        != cfg[
            "canary"
        ][
            "checkpoint"
        ][
            "sha256"
        ]
    ):
        raise ValueError(
            "loaded checkpoint hash differs from Phase-5L binding"
        )

    forward_counter = {
        "count":
            0,
    }

    original_forward = model.forward

    def counted_forward(
        *args,
        **kwargs,
    ):
        forward_counter[
            "count"
        ] += 1

        return original_forward(
            *args,
            **kwargs,
        )

    model.forward = counted_forward

    trial_rows = validated[
        "trial_rows"
    ]

    initial_forward_count = int(
        forward_counter[
            "count"
        ]
    )

    clean_computed = False

    if not clean_reusable:
        (
            clean_action,
            clean_temp,
            _clean_success,
        ) = prepare_atomic_artifact(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            plan_sha256=plan_sha,
            executor_sha256=executor_sha,
            recompute_partial=recompute_partial,
        )

        if clean_action != "compute":
            raise AssertionError(
                "clean artifact unexpectedly reusable after precheck"
            )

        assert clean_temp is not None

        clean_rows: list[dict] = []

        for trial_row in trial_rows:
            trial_windows = _load_trial_signal(
                dataset_root,
                trial_row,
            )

            for window_index in range(
                int(
                    trial_row[
                        "window_count"
                    ]
                )
            ):
                x = _window_tensor(
                    trial_windows,
                    window_index,
                )

                with torch.no_grad():
                    output = model(
                        x
                    )

                parent = _parent_record(
                    row=trial_row,
                    window_index=window_index,
                )

                clean_rows.append(
                    _clean_record(
                        cfg=cfg,
                        parent=parent,
                        input_tensor=x,
                        output_tensor=output,
                    )
                )

        clean_delta = (
            int(
                forward_counter[
                    "count"
                ]
            )
            - initial_forward_count
        )

        if clean_delta != EXPECTED_CLEAN_FORWARD_COUNT:
            raise ValueError(
                f"clean forward count mismatch: {clean_delta}"
            )

        if len(
            clean_rows
        ) != EXPECTED_CLEAN_FORWARD_COUNT:
            raise ValueError(
                "clean record count mismatch"
            )

        write_jsonl(
            clean_temp
            / CLEAN_JSONL,
            clean_rows,
        )

        clean_metadata = {
            "schema_version":
                "phase5m_compute_fi_outer_clean_cache_metadata_v1",

            "status":
                "COMPLETE",

            "artifact_id":
                clean_id,

            "shard_id":
                fault_id,

            "subject":
                9,

            "fold":
                5,

            "model_variant":
                "fp32",

            "checkpoint_seed":
                42,

            "trial_count":
                EXPECTED_TRIAL_COUNT,

            "window_count":
                EXPECTED_WINDOW_COUNT,

            "model_forward_count":
                clean_delta,

            "executor_sha256":
                executor_sha,

            "phase5l_config_sha256":
                EXPECTED_PHASE5L_CONFIG_SHA256,

            "phase5e_plan_sha256":
                plan_sha,
        }

        write_json(
            clean_temp
            / METADATA_JSON,
            clean_metadata,
        )

        clean_hashes = _hash_output_files(
            clean_temp,
            [
                CLEAN_JSONL,
                METADATA_JSON,
            ],
        )

        commit_atomic_artifact(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            temp_dir=clean_temp,
            plan_sha256=plan_sha,
            executor_sha256=executor_sha,
            output_hashes=clean_hashes,
            coverage={
                "trial_count":
                    EXPECTED_TRIAL_COUNT,

                "window_count":
                    EXPECTED_WINDOW_COUNT,

                "clean_model_window_evaluations":
                    EXPECTED_CLEAN_FORWARD_COUNT,

                "labels_read":
                    False,

                "onfield_read":
                    False,
            },
        )

        clean_computed = True

    if not artifact_is_reusable(
        output_root=output_root,
        artifact_id=clean_id,
        artifact_kind=CLEAN_ARTIFACT_KIND,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
    ):
        raise ValueError(
            "fault execution prohibited without reusable clean cache"
        )

    fault_start_forward_count = int(
        forward_counter[
            "count"
        ]
    )

    if fault_reusable:
        return {
            "status":
                "REUSED_FAULT_SHARD",

            "clean_cache_reused":
                not clean_computed,

            "fault_shard_reused":
                True,

            "new_clean_forwards":
                (
                    EXPECTED_CLEAN_FORWARD_COUNT
                    if clean_computed
                    else 0
                ),

            "new_fault_forwards":
                0,
        }

    (
        fault_action,
        fault_temp,
        _fault_success,
    ) = prepare_atomic_artifact(
        output_root=output_root,
        artifact_id=fault_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        recompute_partial=recompute_partial,
    )

    if fault_action != "compute":
        raise AssertionError(
            "fault artifact unexpectedly reusable after precheck"
        )

    assert fault_temp is not None

    targets = cfg[
        "canary"
    ][
        "target_inventory"
    ]

    fault_rows: list[dict] = []

    seen_outer_instance_ids: set[str] = set()

    for trial_row in trial_rows:
        trial_windows = _load_trial_signal(
            dataset_root,
            trial_row,
        )

        for window_index in range(
            int(
                trial_row[
                    "window_count"
                ]
            )
        ):
            x = _window_tensor(
                trial_windows,
                window_index,
            )

            parent = _parent_record(
                row=trial_row,
                window_index=window_index,
            )

            for target in targets:
                (
                    identity,
                    sid,
                    execution_id,
                ) = _fault_identity(
                    cfg=cfg,
                    target=target,
                    row=trial_row,
                    window_index=window_index,
                )

                if execution_id in seen_outer_instance_ids:
                    raise ValueError(
                        "outer_instance_id collision"
                    )

                seen_outer_instance_ids.add(
                    execution_id
                )

                fault_execution = run_fp32_fault_only(
                    model,
                    x,
                    identity,
                    current_inference_index=int(
                        identity.inference_index
                    ),
                )

                if (
                    fault_execution.fault_id
                    != identity.fault_id()
                ):
                    raise ValueError(
                        "fault-only execution returned wrong fault_id"
                    )

                if (
                    fault_execution.input_sha256
                    != tensor_bytes_sha256(
                        x
                    )
                ):
                    raise ValueError(
                        "fault-only execution returned wrong input hash"
                    )

                if not bool(
                    fault_execution.mutation.active
                ):
                    raise ValueError(
                        "transient canary mutation unexpectedly inactive"
                    )

                fault_rows.append(
                    _fault_output_record(
                        cfg=cfg,
                        parent=parent,
                        identity=identity,
                        sampling_instance_id_value=sid,
                        execution_id=execution_id,
                        input_tensor=x,
                        execution=fault_execution,
                    )
                )

    fault_delta = (
        int(
            forward_counter[
                "count"
            ]
        )
        - fault_start_forward_count
    )

    if fault_delta != EXPECTED_FAULT_FORWARD_COUNT:
        raise ValueError(
            f"fault forward count mismatch: {fault_delta}"
        )

    if len(
        fault_rows
    ) != EXPECTED_FAULT_FORWARD_COUNT:
        raise ValueError(
            "fault output record count mismatch"
        )

    if len(
        seen_outer_instance_ids
    ) != EXPECTED_FAULT_FORWARD_COUNT:
        raise ValueError(
            "outer-instance cardinality mismatch"
        )

    write_jsonl(
        fault_temp
        / FAULT_JSONL,
        fault_rows,
    )

    fault_metadata = {
        "schema_version":
            "phase5m_compute_fi_outer_fault_shard_metadata_v1",

        "status":
            "COMPLETE",

        "artifact_id":
            fault_id,

        "clean_cache_id":
            clean_id,

        "subject":
            9,

        "fold":
            5,

        "model_variant":
            "fp32",

        "checkpoint_seed":
            42,

        "persistence":
            "transient_one_inference",

        "target_count":
            EXPECTED_TARGET_COUNT,

        "trial_count":
            EXPECTED_TRIAL_COUNT,

        "window_count":
            EXPECTED_WINDOW_COUNT,

        "outer_instance_id_count":
            len(
                seen_outer_instance_ids
            ),

        "faulted_model_window_evaluations":
            fault_delta,

        "executor_sha256":
            executor_sha,

        "phase5l_config_sha256":
            EXPECTED_PHASE5L_CONFIG_SHA256,

        "phase5e_plan_sha256":
            plan_sha,

        "labels_read":
            False,

        "onfield_read":
            False,

        "threshold_selection_performed":
            False,

        "metrics_generated":
            False,
    }

    write_json(
        fault_temp
        / METADATA_JSON,
        fault_metadata,
    )

    fault_hashes = _hash_output_files(
        fault_temp,
        [
            FAULT_JSONL,
            METADATA_JSON,
        ],
    )

    commit_atomic_artifact(
        output_root=output_root,
        artifact_id=fault_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        temp_dir=fault_temp,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        output_hashes=fault_hashes,
        coverage={
            "subject":
                9,

            "trial_count":
                EXPECTED_TRIAL_COUNT,

            "window_count":
                EXPECTED_WINDOW_COUNT,

            "target_count":
                EXPECTED_TARGET_COUNT,

            "outer_instance_ids":
                EXPECTED_FAULT_FORWARD_COUNT,

            "faulted_model_window_evaluations":
                EXPECTED_FAULT_FORWARD_COUNT,

            "clean_reference_forward_inside_fault_loop":
                False,

            "labels_read":
                False,

            "onfield_read":
                False,

            "threshold_selection_performed":
                False,

            "metrics_generated":
                False,
        },
    )

    total_new_forwards = (
        int(
            forward_counter[
                "count"
            ]
        )
        - initial_forward_count
    )

    expected_total_new = (
        EXPECTED_TOTAL_FORWARD_COUNT
        if clean_computed
        else EXPECTED_FAULT_FORWARD_COUNT
    )

    if total_new_forwards != expected_total_new:
        raise ValueError(
            f"total new forward mismatch: "
            f"{total_new_forwards} "
            f"expected={expected_total_new}"
        )

    return {
        "status":
            "EXECUTED_FROZEN_PHASE5L_CANARY",

        "clean_cache_reused":
            not clean_computed,

        "fault_shard_reused":
            False,

        "new_clean_forwards":
            (
                EXPECTED_CLEAN_FORWARD_COUNT
                if clean_computed
                else 0
            ),

        "new_fault_forwards":
            EXPECTED_FAULT_FORWARD_COUNT,

        "new_total_forwards":
            total_new_forwards,

        "outer_instance_ids":
            len(
                seen_outer_instance_ids
            ),

        "authorized_shard_id":
            fault_id,

        "authorized_clean_cache_id":
            clean_id,
    }


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

    sub.add_parser(
        "validate-config"
    )

    validate_shard_parser = sub.add_parser(
        "validate-shard"
    )

    validate_shard_parser.add_argument(
        "--shard-id",
        required=True,
    )

    execute_parser = sub.add_parser(
        "execute-canary"
    )

    execute_parser.add_argument(
        "--shard-id",
        required=True,
    )

    execute_parser.add_argument(
        "--dataset-root",
        required=True,
    )

    execute_parser.add_argument(
        "--output-root",
        required=True,
    )

    execute_parser.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "validate-config":
        validated = validate_configuration(
            args.config
        )

        cfg = validated[
            "config"
        ]

        print(
            "PHASE5M_VALIDATE_CONFIG=PASS"
        )

        print(
            "AUTHORIZED_SHARD_ID=",
            cfg[
                "authorization"
            ][
                "authorized_shard_id"
            ],
            sep="",
        )

        print(
            "AUTHORIZED_CLEAN_CACHE_ID=",
            cfg[
                "authorization"
            ][
                "authorized_clean_cache_id"
            ],
            sep="",
        )

        print(
            "TRIAL_COUNT=",
            len(
                validated[
                    "trial_rows"
                ]
            ),
            sep="",
        )

        print(
            "WINDOW_COUNT=",
            sum(
                int(
                    row[
                        "window_count"
                    ]
                )
                for row in validated[
                    "trial_rows"
                ]
            ),
            sep="",
        )

        print(
            "TARGET_COUNT=",
            len(
                cfg[
                    "canary"
                ][
                    "target_inventory"
                ]
            ),
            sep="",
        )

        print(
            "OUTER_PAYLOAD_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "MODEL_FORWARD_EXECUTED=False"
        )

        print(
            "FAULT_EXECUTION_EXECUTED=False"
        )

        return

    validated = validate_configuration(
        args.config
    )

    validate_requested_shard(
        validated,
        args.shard_id,
    )

    if args.command == "validate-shard":
        print(
            "PHASE5M_VALIDATE_SHARD=PASS"
        )

        print(
            "AUTHORIZED_SHARD_ID=",
            args.shard_id,
            sep="",
        )

        print(
            "OUTER_PAYLOAD_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "MODEL_FORWARD_EXECUTED=False"
        )

        print(
            "FAULT_EXECUTION_EXECUTED=False"
        )

        return

    result = execute_canary(
        config_path=args.config,
        shard_id=args.shard_id,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        recompute_partial=bool(
            args.recompute_partial
        ),
    )

    print(
        "PHASE5M_EXECUTE_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    for key, value in result.items():
        if key == "status":
            continue

        print(
            f"{key.upper()}={value}"
        )


if __name__ == "__main__":
    main()
