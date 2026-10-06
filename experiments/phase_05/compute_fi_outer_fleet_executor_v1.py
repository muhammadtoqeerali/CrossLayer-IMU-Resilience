"""Phase-5 prospective compute-FI full-fleet executor v1.

The executor is bound to the frozen Phase-5P completion gate.

Validation and metadata functions do not read outer arrays or load models.
Only execute-shard / execute-fleet cross the outer execution boundary.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_execution_harness import (
    build_frozen_ptq_eager,
    clone_state_dict,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
    tensor_bytes_sha256,
)
from compute_fi_fault_only_execution_v1 import (
    PTQWeightFaultOnlySession,
    run_fp32_fault_only,
    run_ptq_activation_buffer_fault_only,
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
    derive_persistent_onset_index,
    outer_instance_id,
    sampling_instance_id,
    transient_inference_index,
)


CROSS_ROOT = Path(__file__).resolve().parents[2]

PROTECH_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori"
)

DEFAULT_GATE = (
    CROSS_ROOT
    / "configs/evaluation/"
    "phase5p_compute_fi_outer_full_fleet_completion_gate_v1.json"
)

DEFAULT_PHASE5D = (
    CROSS_ROOT
    / "configs/evaluation/"
    "phase5d_compute_fi_outer_protocol_v1.json"
)

DEFAULT_PLAN = (
    CROSS_ROOT
    / "manifests/"
    "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

DEFAULT_FP32_FREEZE = (
    CROSS_ROOT
    / "manifests/"
    "phase_4f_prospective_300ms_fp32_freeze_v1.json"
)

DEFAULT_PHASE5F_CORE = (
    CROSS_ROOT
    / "experiments/phase_05/"
    "compute_fi_outer_executor_v1.py"
)

DEFAULT_PHASE5K_MODULE = (
    CROSS_ROOT
    / "experiments/phase_05/"
    "compute_fi_fault_only_execution_v1.py"
)

DEFAULT_PHASE5M_EXECUTOR = (
    CROSS_ROOT
    / "experiments/phase_05/"
    "compute_fi_outer_canary_executor_v1.py"
)

DEFAULT_PHASE5O_RESULT = (
    PROTECH_ROOT
    / "results/"
    "phase5o_compute_fi_outer_canary_technical_acceptance_v1/"
    "technical_acceptance.json"
)

DEFAULT_PTQ_MANIFEST = (
    PROTECH_ROOT
    / "results/"
    "phase4g_ptq_300ms_v7_fc1_fp32_all15_training_calibration_v1/"
    "all15_qualification_manifest.json"
)

DEFAULT_DATASET_ROOT = (
    PROTECH_ROOT
    / "data/UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

DEFAULT_OUTPUT_ROOT = (
    PROTECH_ROOT
    / "results/phase5_outer_compute_fi_prospective_v1"
)

EXPECTED_PHASE5P_GATE_SHA256 = (
    "1f8d6dbd6491cbacf21c0be25053201e0f38a9ad8e987261468611d220af3719"
)

EXPECTED_PHASE5E_PLAN_SHA256 = (
    "95aecd14b4aa8dce70492f0c4d6d50a8bf5a8dafb2ec962aff9c033b8472bb54"
)

EXPECTED_PHASE5D_CONFIG_SHA256 = (
    "0ba30c9d336433748378f8d3eed1ed45b673fb9690250b1bd3bbbc79f88c9c36"
)

EXPECTED_PHASE5F_CORE_SHA256 = (
    "77ee323f71a5dbf4c54cc454a5911849aa45c5d4bdcc6d63bb908d8cece363d0"
)

EXPECTED_PHASE5K_MODULE_SHA256 = (
    "f8bb4095bbfedc69c24fa4cc6e79ccf4914e14fa502ce080a06de01e740112cb"
)

EXPECTED_PHASE5M_EXECUTOR_SHA256 = (
    "aebc8e6d9d89ee44eb31e8b4bb39afa1efe46f597b0dacdfd1d9d0efc9a1a645"
)

EXPECTED_PHASE5O_RESULT_SHA256 = (
    "d67dba6513a62c8192137c3b19d1806db91e4b257cb17b6481ad11a86a897d72"
)

EXPECTED_FP32_FREEZE_SHA256 = (
    "6a07fa5ff4f62dbd61ce7be50cbfe72eed713cf00a789e565deefaca45aa5cdd"
)

EXPECTED_PTQ_MANIFEST_SHA256 = (
    "811481415964bb2dc964116600a0ab43af407b4cddef76422a0f0d1a85819316"
)

PHASE5A_PROTOCOL = (
    "phase5a_compute_fi_representation_protocol_v1"
)

ACCEPTED_CANARY_SHARD_ID = (
    "p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9"
)

ACCEPTED_CANARY_CLEAN_ID = (
    "p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b"
)

CLEAN_ARTIFACT_KIND = "clean_cache"
FAULT_ARTIFACT_KIND = "fault_shard"

CLEAN_JSONL = "clean_outputs.jsonl"
FAULT_JSONL = "fault_outputs.jsonl"
METADATA_JSON = "metadata.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def list_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        canonical_json(
            list(values)
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: str | Path) -> dict:
    return json.loads(
        Path(path).read_text()
    )


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
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
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _require_hash(path: str | Path, expected: str, label: str) -> None:
    path = Path(path)

    if not path.is_file():
        raise ValueError(
            f"missing {label}: {path}"
        )

    observed = sha256_file(path)

    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: {observed}"
        )


def _accepted_canary_hash_validation(
    *,
    output_root: str | Path,
    phase5o_result: Mapping[str, Any],
) -> None:
    root = Path(output_root)

    integrity = phase5o_result[
        "artifact_integrity"
    ]

    specs = (
        (
            "clean_cache",
            "clean_outputs.jsonl",
        ),
        (
            "fault_shard",
            "fault_outputs.jsonl",
        ),
    )

    for key, records_name in specs:
        rec = integrity[key]

        directory = (
            root
            / rec["artifact_id"]
        )

        if not directory.is_dir():
            raise ValueError(
                f"accepted canary artifact missing: {directory}"
            )

        observed = {
            "success_marker_sha256":
                sha256_file(
                    directory
                    / "_SUCCESS.json"
                ),

            "metadata_sha256":
                sha256_file(
                    directory
                    / "metadata.json"
                ),

            "records_sha256":
                sha256_file(
                    directory
                    / records_name
                ),
        }

        for field, value in observed.items():
            if value != rec[field]:
                raise ValueError(
                    f"accepted canary {key} {field} mismatch"
                )


def trial_rows_for_shard(
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> list[dict]:
    rows = [
        dict(row)
        for row in plan["trial_inventory"]
        if int(row["subject"]) == int(shard["subject"])
        and int(row["fold"]) == int(shard["fold"])
    ]

    rows.sort(
        key=lambda row: (
            int(row["task"]),
            int(row["trial"]),
        )
    )

    expected = shard[
        "subject_inventory"
    ]

    if len(rows) != int(
        expected["trial_count"]
    ):
        raise ValueError(
            "trial-count mismatch for shard"
        )

    window_count = sum(
        int(row["window_count"])
        for row in rows
    )

    if window_count != int(
        expected["window_count"]
    ):
        raise ValueError(
            "window-count mismatch for shard"
        )

    return rows


def target_rows_for_shard(
    phase5d: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> list[dict]:
    variant = shard[
        "model_variant"
    ]

    rows = [
        dict(row)
        for row in phase5d["target_inventory"]
        if variant in row["model_variants"]
    ]

    by_name = {
        row["target_name"]:
            row
        for row in rows
    }

    ordered = [
        by_name[name]
        for name in shard["target_names"]
    ]

    if len(ordered) != int(
        shard["target_count"]
    ):
        raise ValueError(
            "target-count mismatch for shard"
        )

    if [
        row["target_name"]
        for row in ordered
    ] != list(
        shard["target_names"]
    ):
        raise ValueError(
            "target ordering mismatch"
        )

    return ordered


def persistent_target_binding(
    *,
    shard: Mapping[str, Any],
    target: Mapping[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    digest = hashlib.sha256()

    exposure = 0
    instances = 0

    for trial_row in trial_rows:
        payload = canonical_sampling_payload(
            partition="outer_test",
            fold=int(shard["fold"]),
            subject=int(shard["subject"]),
            task=int(trial_row["task"]),
            trial=int(trial_row["trial"]),
            parent_kind="trial",
            window_index=None,
            representation_class=target["representation_class"],
            target_name=target["target_name"],
            target_role=target["target_role"],
            persistence="persistent_from_onset_until_trial_end",
            replicate_index=0,
        )

        trial_windows = int(
            trial_row["window_count"]
        )

        onset = derive_persistent_onset_index(
            payload,
            trial_window_count=trial_windows,
        )

        sid = sampling_instance_id(
            payload
        )

        digest.update(
            (
                f"{sid}|"
                f"{trial_windows}|"
                f"{onset}\n"
            ).encode("utf-8")
        )

        exposure += (
            trial_windows
            - onset
        )

        instances += 1

    return {
        "outer_instance_count":
            instances,

        "faulted_model_window_evaluations":
            exposure,

        "subject_target_onset_binding_sha256":
            digest.hexdigest(),
    }


def validate_shard_structure(
    *,
    phase5d: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    trial_rows = trial_rows_for_shard(
        plan,
        shard,
    )

    targets = target_rows_for_shard(
        phase5d,
        shard,
    )

    exposure_by_name = {
        row["target_name"]:
            row
        for row in shard["target_exposure"]
    }

    expected_outer_ids = 0
    expected_fault_evals = 0

    if (
        shard["persistence"]
        == "transient_one_inference"
    ):
        windows = int(
            shard[
                "subject_inventory"
            ][
                "window_count"
            ]
        )

        for target in targets:
            exposure = exposure_by_name[
                target["target_name"]
            ]

            if int(
                exposure[
                    "outer_instance_count"
                ]
            ) != windows:
                raise ValueError(
                    "transient target outer-instance mismatch"
                )

            if int(
                exposure[
                    "faulted_model_window_evaluations"
                ]
            ) != windows:
                raise ValueError(
                    "transient target forward-count mismatch"
                )

            expected_outer_ids += windows
            expected_fault_evals += windows

    elif (
        shard["persistence"]
        == "persistent_from_onset_until_trial_end"
    ):
        for target in targets:
            derived = persistent_target_binding(
                shard=shard,
                target=target,
                trial_rows=trial_rows,
            )

            exposure = exposure_by_name[
                target["target_name"]
            ]

            for key in (
                "outer_instance_count",
                "faulted_model_window_evaluations",
                "subject_target_onset_binding_sha256",
            ):
                if derived[key] != exposure[key]:
                    raise ValueError(
                        f"persistent target binding mismatch "
                        f"{target['target_name']} {key}"
                    )

            expected_outer_ids += int(
                derived[
                    "outer_instance_count"
                ]
            )

            expected_fault_evals += int(
                derived[
                    "faulted_model_window_evaluations"
                ]
            )

    else:
        raise ValueError(
            "unknown persistence mode"
        )

    if expected_outer_ids != int(
        shard[
            "expected_outer_instance_ids"
        ]
    ):
        raise ValueError(
            "shard outer-instance cardinality mismatch"
        )

    if expected_fault_evals != int(
        shard[
            "expected_faulted_model_window_evaluations"
        ]
    ):
        raise ValueError(
            "shard fault-forward cardinality mismatch"
        )

    return {
        "trial_rows":
            trial_rows,

        "targets":
            targets,

        "expected_outer_instance_ids":
            expected_outer_ids,

        "expected_faulted_model_window_evaluations":
            expected_fault_evals,
    }


def validate_gate(
    gate_path: str | Path = DEFAULT_GATE,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    gate_path = Path(gate_path)

    _require_hash(
        gate_path,
        EXPECTED_PHASE5P_GATE_SHA256,
        "Phase-5P gate",
    )

    _require_hash(
        DEFAULT_PLAN,
        EXPECTED_PHASE5E_PLAN_SHA256,
        "Phase-5E plan",
    )

    _require_hash(
        DEFAULT_PHASE5D,
        EXPECTED_PHASE5D_CONFIG_SHA256,
        "Phase-5D config",
    )

    _require_hash(
        DEFAULT_PHASE5F_CORE,
        EXPECTED_PHASE5F_CORE_SHA256,
        "Phase-5F atomic core",
    )

    _require_hash(
        DEFAULT_PHASE5K_MODULE,
        EXPECTED_PHASE5K_MODULE_SHA256,
        "Phase-5K fault-only module",
    )

    _require_hash(
        DEFAULT_PHASE5M_EXECUTOR,
        EXPECTED_PHASE5M_EXECUTOR_SHA256,
        "Phase-5M canary executor",
    )

    _require_hash(
        DEFAULT_PHASE5O_RESULT,
        EXPECTED_PHASE5O_RESULT_SHA256,
        "Phase-5O technical acceptance",
    )

    _require_hash(
        DEFAULT_FP32_FREEZE,
        EXPECTED_FP32_FREEZE_SHA256,
        "Phase-4F FP32 freeze",
    )

    _require_hash(
        DEFAULT_PTQ_MANIFEST,
        EXPECTED_PTQ_MANIFEST_SHA256,
        "Phase-4G PTQ all15 manifest",
    )

    gate = load_json(
        gate_path
    )

    plan = load_json(
        DEFAULT_PLAN
    )

    phase5d = load_json(
        DEFAULT_PHASE5D
    )

    acceptance = load_json(
        DEFAULT_PHASE5O_RESULT
    )

    if (
        gate["status"]
        != "FROZEN_PROSPECTIVE_FULL_FLEET_COMPLETION_AUTHORIZATION"
    ):
        raise ValueError(
            "unexpected Phase-5P gate status"
        )

    authorization = gate[
        "authorization"
    ]

    if (
        authorization[
            "full_fleet_completion_authorized"
        ]
        is not True
    ):
        raise ValueError(
            "full-fleet completion is not authorized"
        )

    if (
        authorization[
            "full_fleet_execution_started"
        ]
        is not False
    ):
        raise ValueError(
            "frozen gate must predate full-fleet execution"
        )

    if (
        authorization[
            "authorization_uses_prediction_outcome"
        ]
        is not False
    ):
        raise ValueError(
            "outcome-based authorization prohibited"
        )

    if len(plan["shards"]) != 732:
        raise ValueError(
            "Phase-5E shard count changed"
        )

    if len(
        plan["clean_caches"]
    ) != 366:
        raise ValueError(
            "Phase-5E clean-cache count changed"
        )

    if (
        plan["shards"][0]["shard_id"]
        != ACCEPTED_CANARY_SHARD_ID
    ):
        raise ValueError(
            "accepted canary is no longer plan shard zero"
        )

    _accepted_canary_hash_validation(
        output_root=output_root,
        phase5o_result=acceptance,
    )

    remaining_shards = [
        row
        for row in plan["shards"]
        if row["shard_id"]
        != ACCEPTED_CANARY_SHARD_ID
    ]

    remaining_clean = [
        row
        for row in plan["clean_caches"]
        if row["clean_cache_id"]
        != ACCEPTED_CANARY_CLEAN_ID
    ]

    if len(
        remaining_shards
    ) != 731:
        raise ValueError(
            "remaining shard count changed"
        )

    if len(
        remaining_clean
    ) != 365:
        raise ValueError(
            "remaining clean-cache count changed"
        )

    remaining_gate = gate[
        "remaining_authorized_estate"
    ]

    if (
        list_digest(
            [
                row["shard_id"]
                for row in remaining_shards
            ]
        )
        != remaining_gate[
            "remaining_shard_ids_sha256"
        ]
    ):
        raise ValueError(
            "remaining shard-set digest mismatch"
        )

    if (
        list_digest(
            [
                row["clean_cache_id"]
                for row in remaining_clean
            ]
        )
        != remaining_gate[
            "remaining_clean_cache_ids_sha256"
        ]
    ):
        raise ValueError(
            "remaining clean-cache-set digest mismatch"
        )

    return {
        "gate":
            gate,

        "plan":
            plan,

        "phase5d":
            phase5d,

        "acceptance":
            acceptance,

        "remaining_shards":
            remaining_shards,

        "remaining_clean_caches":
            remaining_clean,
    }


def shard_authorization(
    validated: Mapping[str, Any],
    shard_id: str,
) -> dict[str, Any]:
    plan = validated[
        "plan"
    ]

    row = next(
        (
            dict(item)
            for item in plan["shards"]
            if item["shard_id"] == shard_id
        ),
        None,
    )

    if row is None:
        raise ValueError(
            "requested shard is not in frozen Phase-5E plan"
        )

    if shard_id == ACCEPTED_CANARY_SHARD_ID:
        return {
            "mode":
                "accepted_canary_reuse_only",

            "shard":
                row,
        }

    remaining_ids = {
        item["shard_id"]
        for item in validated[
            "remaining_shards"
        ]
    }

    if shard_id not in remaining_ids:
        raise ValueError(
            "requested shard is not authorized by Phase 5P"
        )

    return {
        "mode":
            "remaining_authorized_execution",

        "shard":
            row,
    }


def build_outer_identity(
    *,
    shard: Mapping[str, Any],
    target: Mapping[str, Any],
    trial_row: Mapping[str, Any],
    window_index: int | None,
) -> dict[str, Any]:
    persistence = shard[
        "persistence"
    ]

    if persistence == "transient_one_inference":
        if window_index is None:
            raise ValueError(
                "transient identity requires window_index"
            )

        parent_kind = "window"
        payload_window = int(
            window_index
        )

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        if window_index is not None:
            raise ValueError(
                "persistent identity is trial-bound"
            )

        parent_kind = "trial"
        payload_window = None

    else:
        raise ValueError(
            "unknown persistence mode"
        )

    payload = canonical_sampling_payload(
        partition="outer_test",
        fold=int(shard["fold"]),
        subject=int(shard["subject"]),
        task=int(trial_row["task"]),
        trial=int(trial_row["trial"]),
        parent_kind=parent_kind,
        window_index=payload_window,
        representation_class=target["representation_class"],
        target_name=target["target_name"],
        target_role=target["target_role"],
        persistence=persistence,
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

    if persistence == "transient_one_inference":
        onset = transient_inference_index(
            payload
        )
    else:
        onset = derive_persistent_onset_index(
            payload,
            trial_window_count=int(
                trial_row["window_count"]
            ),
        )

    identity = FaultIdentity(
        protocol=
            PHASE5A_PROTOCOL,

        model_variant=
            shard["model_variant"],

        checkpoint_seed=
            int(shard["checkpoint_seed"]),

        fold=
            int(shard["fold"]),

        fault_family=
            target["fault_family"],

        representation_class=
            target["representation_class"],

        target_name=
            target["target_name"],

        target_role=
            target["target_role"],

        element_index=
            int(element),

        bit_position=
            int(bit),

        inference_index=
            int(onset),

        persistence=
            persistence,

        multiplicity=
            1,

        replicate_index=
            0,
    )

    validate_fault_identity(
        identity
    )

    phase5a_fault_id = identity.fault_id()

    oid = outer_instance_id(
        sampling_instance_id_value=sid,
        phase5a_fault_id=phase5a_fault_id,
        model_variant=identity.model_variant,
        checkpoint_seed=identity.checkpoint_seed,
    )

    return {
        "payload":
            payload,

        "sampling_instance_id":
            sid,

        "phase5a_fault_id":
            phase5a_fault_id,

        "outer_instance_id":
            oid,

        "identity":
            identity,

        "onset_index":
            int(onset),
    }


def execute_fault_sequence(
    *,
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    inference_indices: Sequence[int],
    identities: Sequence[FaultIdentity],
    ptq_clean_state: Mapping[str, Any] | None = None,
) -> list[Any]:
    if not inputs:
        raise ValueError(
            "fault sequence must not be empty"
        )

    if len(inputs) != len(
        inference_indices
    ):
        raise ValueError(
            "input/index length mismatch"
        )

    if list(
        map(int, inference_indices)
    ) != sorted(
        int(x)
        for x in inference_indices
    ):
        raise ValueError(
            "inference indices must be monotonic"
        )

    if not identities:
        raise ValueError(
            "identities must not be empty"
        )

    for identity in identities:
        validate_fault_identity(
            identity
        )

    persistence = identities[0].persistence

    if any(
        identity.persistence
        != persistence
        for identity in identities
    ):
        raise ValueError(
            "mixed persistence modes prohibited"
        )

    variant = identities[0].model_variant

    if any(
        identity.model_variant
        != variant
        for identity in identities
    ):
        raise ValueError(
            "mixed model variants prohibited"
        )

    if persistence == "transient_one_inference":
        if len(
            identities
        ) != len(
            inputs
        ):
            raise ValueError(
                "transient sequence requires one identity per input"
            )

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        if len(
            identities
        ) != 1:
            raise ValueError(
                "persistent sequence requires exactly one identity"
            )

    else:
        raise ValueError(
            "unknown persistence mode"
        )

    weight_route = (
        identities[0].representation_class
        == "int8_persistent_weight"
    )

    if weight_route:
        if variant != "ptq_v7":
            raise ValueError(
                "qint8 weight route requires ptq_v7"
            )

        if ptq_clean_state is None:
            raise ValueError(
                "PTQ weight route requires clean state"
            )

        session = PTQWeightFaultOnlySession(
            model,
            dict(ptq_clean_state),
        )

        executions = []

        try:
            for index, (
                x,
                current_inference_index,
            ) in enumerate(
                zip(
                    inputs,
                    inference_indices,
                )
            ):
                identity = (
                    identities[index]
                    if persistence
                    == "transient_one_inference"
                    else identities[0]
                )

                executions.append(
                    session.run_fault_only(
                        x,
                        identity,
                        current_inference_index=int(
                            current_inference_index
                        ),
                    )
                )

        finally:
            session.reset_clean()

        return executions

    executions = []

    for index, (
        x,
        current_inference_index,
    ) in enumerate(
        zip(
            inputs,
            inference_indices,
        )
    ):
        identity = (
            identities[index]
            if persistence
            == "transient_one_inference"
            else identities[0]
        )

        if variant == "fp32":
            execution = run_fp32_fault_only(
                model,
                x,
                identity,
                current_inference_index=int(
                    current_inference_index
                ),
            )

        elif variant == "ptq_v7":
            execution = (
                run_ptq_activation_buffer_fault_only(
                    model,
                    x,
                    identity,
                    current_inference_index=int(
                        current_inference_index
                    ),
                )
            )

        else:
            raise ValueError(
                f"unknown model variant: {variant}"
            )

        executions.append(
            execution
        )

    return executions


def execution_active_mask(
    executions: Sequence[Any],
) -> list[bool]:
    return [
        bool(
            execution.mutation.active
        )
        for execution in executions
    ]


def _ptq_member(
    *,
    manifest: Mapping[str, Any],
    seed: int,
    fold: int,
) -> dict:
    matches = [
        row
        for row in manifest["members"]
        if int(row["seed"]) == int(seed)
        and int(row["fold"]) == int(fold)
    ]

    if len(matches) != 1:
        raise ValueError(
            "expected exactly one PTQ seed/fold member"
        )

    return dict(
        matches[0]
    )


def load_model_bundle(
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(
        shard[
            "checkpoint_seed"
        ]
    )

    fold = int(
        shard[
            "fold"
        ]
    )

    fp32_model, fp32_meta = (
        load_frozen_fp32_model(
            DEFAULT_FP32_FREEZE,
            seed=seed,
            fold=fold,
        )
    )

    variant = shard[
        "model_variant"
    ]

    if variant == "fp32":
        return {
            "model":
                fp32_model,

            "ptq_clean_state":
                None,

            "metadata": {
                "model_variant":
                    "fp32",

                **fp32_meta,
            },
        }

    if variant != "ptq_v7":
        raise ValueError(
            f"unknown model variant: {variant}"
        )

    ptq_manifest = load_json(
        DEFAULT_PTQ_MANIFEST
    )

    member = _ptq_member(
        manifest=ptq_manifest,
        seed=seed,
        fold=fold,
    )

    state_record = member[
        "artifacts"
    ][
        "state_dict"
    ]

    state_path = Path(
        state_record["path"]
    )

    _require_hash(
        state_path,
        state_record["sha256"],
        "PTQ state_dict",
    )

    ptq_model = build_frozen_ptq_eager(
        fp32_model,
        state_path,
    )

    return {
        "model":
            ptq_model,

        "ptq_clean_state":
            clone_state_dict(
                ptq_model.state_dict()
            ),

        "metadata": {
            "model_variant":
                "ptq_v7",

            "seed":
                seed,

            "fold":
                fold,

            "state_dict":
                dict(state_record),

            "torchscript":
                dict(
                    member[
                        "artifacts"
                    ][
                        "torchscript"
                    ]
                ),

            "checkpoint":
                dict(
                    member[
                        "checkpoint"
                    ]
                ),
        },
    }


def _load_outer_trial_signal(
    dataset_root: str | Path,
    trial_row: Mapping[str, Any],
) -> np.ndarray:
    """Only outer-array loading point in this production executor."""

    path = (
        Path(dataset_root)
        / str(int(trial_row["subject"]))
        / str(int(trial_row["task"]))
        / str(int(trial_row["trial"]))
        / "segments.npy"
    )

    if not path.is_file():
        raise ValueError(
            f"missing outer segments: {path}"
        )

    windows = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected = (
        int(trial_row["window_count"]),
        30,
        9,
    )

    if tuple(
        windows.shape
    ) != expected:
        raise ValueError(
            f"outer trial shape mismatch: "
            f"{windows.shape} != {expected}"
        )

    return windows


def _window_tensor(
    windows: np.ndarray,
    index: int,
) -> torch.Tensor:
    return torch.from_numpy(
        np.asarray(
            windows[int(index)],
            dtype=np.float32,
        )
    ).unsqueeze(0)


def _float_token(
    value: float,
) -> float | dict[str, str]:
    value = float(value)

    if math.isnan(value):
        return {
            "nonfinite":
                "nan"
        }

    if math.isinf(value):
        return {
            "nonfinite":
                (
                    "+inf"
                    if value > 0
                    else "-inf"
                )
        }

    return value


def _tensor_values(
    tensor: torch.Tensor,
) -> list[Any]:
    return [
        _float_token(
            value.item()
        )
        for value in (
            tensor
            .detach()
            .cpu()
            .reshape(-1)
        )
    ]


def _tensor_float32_hex(
    tensor: torch.Tensor,
) -> list[str]:
    return [
        struct.pack(
            "<f",
            float(value.item()),
        ).hex()
        for value in (
            tensor
            .detach()
            .cpu()
            .to(torch.float32)
            .reshape(-1)
        )
    ]


def _tensor_nonfinite(
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


def _softmax_values(
    tensor: torch.Tensor,
) -> list[Any]:
    return _tensor_values(
        torch.softmax(
            tensor.to(
                torch.float32
            ),
            dim=-1,
        )
    )


def _parent_window(
    trial_row: Mapping[str, Any],
    window_index: int,
) -> dict[str, Any]:
    return {
        "partition":
            "outer_test",

        "fold":
            int(
                trial_row["fold"]
            ),

        "canonical_subject":
            trial_row[
                "canonical_subject"
            ],

        "subject":
            int(
                trial_row["subject"]
            ),

        "task":
            int(
                trial_row["task"]
            ),

        "trial":
            int(
                trial_row["trial"]
            ),

        "window_index":
            int(window_index),
    }


def _parent_trial(
    trial_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "partition":
            "outer_test",

        "fold":
            int(
                trial_row["fold"]
            ),

        "canonical_subject":
            trial_row[
                "canonical_subject"
            ],

        "subject":
            int(
                trial_row["subject"]
            ),

        "task":
            int(
                trial_row["task"]
            ),

        "trial":
            int(
                trial_row["trial"]
            ),
    }


def _fault_record(
    *,
    shard: Mapping[str, Any],
    identity_info: Mapping[str, Any],
    execution_window_index: int,
    input_tensor: torch.Tensor,
    execution: Any,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    identity = identity_info[
        "identity"
    ]

    output = execution[
        "faulted_output"
    ] if isinstance(
        execution,
        dict,
    ) else execution.faulted_output

    mutation = execution[
        "mutation"
    ] if isinstance(
        execution,
        dict,
    ) else execution.mutation

    return {
        "schema_version":
            "phase5r_compute_fi_fault_only_outer_record_v1",

        "shard_id":
            shard["shard_id"],

        "clean_cache_id":
            shard["clean_cache_id"],

        "outer_instance_id":
            identity_info[
                "outer_instance_id"
            ],

        "sampling_instance_id":
            identity_info[
                "sampling_instance_id"
            ],

        "phase5a_fault_id":
            identity_info[
                "phase5a_fault_id"
            ],

        "model_variant":
            identity.model_variant,

        "checkpoint_seed":
            int(
                identity.checkpoint_seed
            ),

        "fold":
            int(identity.fold),

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

        "execution_window_index":
            int(
                execution_window_index
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
            dict(parent),

        "input_sha256":
            tensor_bytes_sha256(
                input_tensor
            ),

        "faulted_output_sha256":
            tensor_bytes_sha256(
                output
            ),

        "faulted_output_nonfinite":
            _tensor_nonfinite(
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
            _softmax_values(
                output
            ),

        "mutation":
            dataclasses.asdict(
                mutation
            ),
    }


def _clean_record(
    *,
    shard: Mapping[str, Any],
    parent: Mapping[str, Any],
    x: torch.Tensor,
    output: torch.Tensor,
) -> dict[str, Any]:
    rec = clean_output_record(
        clean_cache_id=shard[
            "clean_cache_id"
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
        input_tensor=x,
        output_tensor=output,
    )

    rec[
        "clean_output_values"
    ] = _tensor_values(
        output
    )

    rec[
        "clean_output_float32_hex"
    ] = _tensor_float32_hex(
        output
    )

    rec[
        "clean_softmax_values"
    ] = _softmax_values(
        output
    )

    return rec


def accepted_canary_clean_reusable(
    validated: Mapping[str, Any],
    clean_cache_id: str,
    *,
    output_root: str | Path,
) -> bool:
    if clean_cache_id != ACCEPTED_CANARY_CLEAN_ID:
        return False

    _accepted_canary_hash_validation(
        output_root=output_root,
        phase5o_result=validated[
            "acceptance"
        ],
    )

    return True


def _hash_outputs(
    temp_dir: Path,
    names: Sequence[str],
) -> dict[str, str]:
    return {
        name:
            sha256_file(
                temp_dir
                / name
            )
        for name in names
    }


def execute_shard(
    *,
    gate_path: str | Path,
    shard_id: str,
    dataset_root: str | Path,
    output_root: str | Path,
    recompute_partial: bool = False,
) -> dict[str, Any]:
    validated = validate_gate(
        gate_path,
        output_root=output_root,
    )

    authorization = shard_authorization(
        validated,
        shard_id,
    )

    shard = authorization[
        "shard"
    ]

    if (
        authorization["mode"]
        == "accepted_canary_reuse_only"
    ):
        _accepted_canary_hash_validation(
            output_root=output_root,
            phase5o_result=validated[
                "acceptance"
            ],
        )

        return {
            "status":
                "REUSED_ACCEPTED_CANARY",

            "new_clean_forwards":
                0,

            "new_fault_forwards":
                0,
        }

    structure = validate_shard_structure(
        phase5d=validated[
            "phase5d"
        ],
        plan=validated[
            "plan"
        ],
        shard=shard,
    )

    executor_sha = sha256_file(
        __file__
    )

    output_root = Path(
        output_root
    )

    clean_id = shard[
        "clean_cache_id"
    ]

    clean_reusable = (
        accepted_canary_clean_reusable(
            validated,
            clean_id,
            output_root=output_root,
        )
        or artifact_is_reusable(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
            executor_sha256=executor_sha,
        )
    )

    fault_reusable = artifact_is_reusable(
        output_root=output_root,
        artifact_id=shard_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
        executor_sha256=executor_sha,
    )

    if fault_reusable:
        return {
            "status":
                "REUSED_COMPLETE_SHARD",

            "new_clean_forwards":
                0,

            "new_fault_forwards":
                0,
        }

    bundle = load_model_bundle(
        shard
    )

    model = bundle[
        "model"
    ]

    model.eval()

    trial_rows = structure[
        "trial_rows"
    ]

    targets = structure[
        "targets"
    ]

    clean_forward_count = 0
    fault_forward_count = 0

    if not clean_reusable:
        (
            action,
            temp_dir,
            _success,
        ) = prepare_atomic_artifact(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
            executor_sha256=executor_sha,
            recompute_partial=recompute_partial,
        )

        if action != "compute":
            raise AssertionError(
                "unexpected clean-cache reuse state"
            )

        assert temp_dir is not None

        clean_rows = []

        for trial_row in trial_rows:
            windows = _load_outer_trial_signal(
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
                    windows,
                    window_index,
                )

                with torch.no_grad():
                    output = model(
                        x
                    )

                clean_forward_count += 1

                clean_rows.append(
                    _clean_record(
                        shard=shard,
                        parent=_parent_window(
                            trial_row,
                            window_index,
                        ),
                        x=x,
                        output=output,
                    )
                )

        expected_clean = int(
            next(
                row[
                    "expected_clean_model_window_evaluations"
                ]
                for row in validated[
                    "plan"
                ][
                    "clean_caches"
                ]
                if row[
                    "clean_cache_id"
                ]
                == clean_id
            )
        )

        if clean_forward_count != expected_clean:
            raise ValueError(
                "clean forward count mismatch"
            )

        write_jsonl(
            temp_dir
            / CLEAN_JSONL,
            clean_rows,
        )

        clean_meta = {
            "schema_version":
                "phase5r_compute_fi_outer_clean_cache_metadata_v1",

            "status":
                "COMPLETE",

            "artifact_id":
                clean_id,

            "subject":
                int(
                    shard["subject"]
                ),

            "fold":
                int(
                    shard["fold"]
                ),

            "model_variant":
                shard[
                    "model_variant"
                ],

            "checkpoint_seed":
                int(
                    shard[
                        "checkpoint_seed"
                    ]
                ),

            "trial_count":
                len(
                    trial_rows
                ),

            "window_count":
                expected_clean,

            "model_forward_count":
                clean_forward_count,

            "labels_read":
                False,

            "onfield_read":
                False,

            "threshold_selection_performed":
                False,

            "metrics_generated":
                False,

            "executor_sha256":
                executor_sha,

            "phase5p_gate_sha256":
                EXPECTED_PHASE5P_GATE_SHA256,

            "phase5e_plan_sha256":
                EXPECTED_PHASE5E_PLAN_SHA256,
        }

        write_json(
            temp_dir
            / METADATA_JSON,
            clean_meta,
        )

        commit_atomic_artifact(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            temp_dir=temp_dir,
            plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
            executor_sha256=executor_sha,
            output_hashes=_hash_outputs(
                temp_dir,
                [
                    CLEAN_JSONL,
                    METADATA_JSON,
                ],
            ),
            coverage={
                "trial_count":
                    len(
                        trial_rows
                    ),

                "window_count":
                    expected_clean,

                "clean_model_window_evaluations":
                    clean_forward_count,

                "labels_read":
                    False,

                "onfield_read":
                    False,
            },
        )

    if not (
        accepted_canary_clean_reusable(
            validated,
            clean_id,
            output_root=output_root,
        )
        or artifact_is_reusable(
            output_root=output_root,
            artifact_id=clean_id,
            artifact_kind=CLEAN_ARTIFACT_KIND,
            plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
            executor_sha256=executor_sha,
        )
    ):
        raise ValueError(
            "fault execution prohibited without valid clean cache"
        )

    (
        action,
        temp_dir,
        _success,
    ) = prepare_atomic_artifact(
        output_root=output_root,
        artifact_id=shard_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
        executor_sha256=executor_sha,
        recompute_partial=recompute_partial,
    )

    if action != "compute":
        raise AssertionError(
            "unexpected fault-shard reuse state"
        )

    assert temp_dir is not None

    fault_rows = []
    seen_outer_ids = set()

    for trial_row in trial_rows:
        windows = _load_outer_trial_signal(
            dataset_root,
            trial_row,
        )

        if (
            shard[
                "persistence"
            ]
            == "transient_one_inference"
        ):
            for target in targets:
                inputs = []
                indices = []
                identities = []
                infos = []

                for window_index in range(
                    int(
                        trial_row[
                            "window_count"
                        ]
                    )
                ):
                    info = build_outer_identity(
                        shard=shard,
                        target=target,
                        trial_row=trial_row,
                        window_index=window_index,
                    )

                    inputs.append(
                        _window_tensor(
                            windows,
                            window_index,
                        )
                    )

                    indices.append(
                        int(
                            info[
                                "onset_index"
                            ]
                        )
                    )

                    identities.append(
                        info[
                            "identity"
                        ]
                    )

                    infos.append(
                        info
                    )

                executions = execute_fault_sequence(
                    model=model,
                    inputs=inputs,
                    inference_indices=indices,
                    identities=identities,
                    ptq_clean_state=bundle[
                        "ptq_clean_state"
                    ],
                )

                for window_index, (
                    x,
                    info,
                    execution,
                ) in enumerate(
                    zip(
                        inputs,
                        infos,
                        executions,
                    )
                ):
                    oid = info[
                        "outer_instance_id"
                    ]

                    if oid in seen_outer_ids:
                        raise ValueError(
                            "transient outer_instance_id collision"
                        )

                    seen_outer_ids.add(
                        oid
                    )

                    fault_forward_count += 1

                    fault_rows.append(
                        _fault_record(
                            shard=shard,
                            identity_info=info,
                            execution_window_index=window_index,
                            input_tensor=x,
                            execution=execution,
                            parent=_parent_window(
                                trial_row,
                                window_index,
                            ),
                        )
                    )

        else:
            for target in targets:
                info = build_outer_identity(
                    shard=shard,
                    target=target,
                    trial_row=trial_row,
                    window_index=None,
                )

                oid = info[
                    "outer_instance_id"
                ]

                if oid in seen_outer_ids:
                    raise ValueError(
                        "persistent outer_instance_id collision"
                    )

                seen_outer_ids.add(
                    oid
                )

                onset = int(
                    info[
                        "onset_index"
                    ]
                )

                indices = list(
                    range(
                        onset,
                        int(
                            trial_row[
                                "window_count"
                            ]
                        ),
                    )
                )

                inputs = [
                    _window_tensor(
                        windows,
                        index,
                    )
                    for index in indices
                ]

                executions = execute_fault_sequence(
                    model=model,
                    inputs=inputs,
                    inference_indices=indices,
                    identities=[
                        info[
                            "identity"
                        ]
                    ],
                    ptq_clean_state=bundle[
                        "ptq_clean_state"
                    ],
                )

                if execution_active_mask(
                    executions
                ) != [
                    True
                ] * len(
                    executions
                ):
                    raise ValueError(
                        "persistent post-onset execution contained inactive row"
                    )

                for x, index, execution in zip(
                    inputs,
                    indices,
                    executions,
                ):
                    fault_forward_count += 1

                    fault_rows.append(
                        _fault_record(
                            shard=shard,
                            identity_info=info,
                            execution_window_index=index,
                            input_tensor=x,
                            execution=execution,
                            parent=_parent_trial(
                                trial_row
                            ),
                        )
                    )

    if len(
        seen_outer_ids
    ) != int(
        shard[
            "expected_outer_instance_ids"
        ]
    ):
        raise ValueError(
            "outer-instance cardinality mismatch"
        )

    if fault_forward_count != int(
        shard[
            "expected_faulted_model_window_evaluations"
        ]
    ):
        raise ValueError(
            "fault-forward cardinality mismatch"
        )

    if len(
        fault_rows
    ) != fault_forward_count:
        raise ValueError(
            "fault-record cardinality mismatch"
        )

    write_jsonl(
        temp_dir
        / FAULT_JSONL,
        fault_rows,
    )

    fault_meta = {
        "schema_version":
            "phase5r_compute_fi_outer_fault_shard_metadata_v1",

        "status":
            "COMPLETE",

        "artifact_id":
            shard_id,

        "clean_cache_id":
            clean_id,

        "subject":
            int(
                shard["subject"]
            ),

        "fold":
            int(
                shard["fold"]
            ),

        "model_variant":
            shard[
                "model_variant"
            ],

        "checkpoint_seed":
            int(
                shard[
                    "checkpoint_seed"
                ]
            ),

        "persistence":
            shard[
                "persistence"
            ],

        "target_count":
            int(
                shard[
                    "target_count"
                ]
            ),

        "trial_count":
            len(
                trial_rows
            ),

        "window_count":
            int(
                shard[
                    "subject_inventory"
                ][
                    "window_count"
                ]
            ),

        "outer_instance_id_count":
            len(
                seen_outer_ids
            ),

        "faulted_model_window_evaluations":
            fault_forward_count,

        "labels_read":
            False,

        "onfield_read":
            False,

        "threshold_selection_performed":
            False,

        "metrics_generated":
            False,

        "executor_sha256":
            executor_sha,

        "phase5p_gate_sha256":
            EXPECTED_PHASE5P_GATE_SHA256,

        "phase5e_plan_sha256":
            EXPECTED_PHASE5E_PLAN_SHA256,
    }

    write_json(
        temp_dir
        / METADATA_JSON,
        fault_meta,
    )

    commit_atomic_artifact(
        output_root=output_root,
        artifact_id=shard_id,
        artifact_kind=FAULT_ARTIFACT_KIND,
        temp_dir=temp_dir,
        plan_sha256=EXPECTED_PHASE5E_PLAN_SHA256,
        executor_sha256=executor_sha,
        output_hashes=_hash_outputs(
            temp_dir,
            [
                FAULT_JSONL,
                METADATA_JSON,
            ],
        ),
        coverage={
            "subject":
                int(
                    shard["subject"]
                ),

            "trial_count":
                len(
                    trial_rows
                ),

            "window_count":
                int(
                    shard[
                        "subject_inventory"
                    ][
                        "window_count"
                    ]
                ),

            "target_count":
                int(
                    shard[
                        "target_count"
                    ]
                ),

            "outer_instance_ids":
                len(
                    seen_outer_ids
                ),

            "faulted_model_window_evaluations":
                fault_forward_count,

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

    return {
        "status":
            "EXECUTED_AUTHORIZED_PHASE5P_SHARD",

        "shard_id":
            shard_id,

        "clean_cache_id":
            clean_id,

        "new_clean_forwards":
            clean_forward_count,

        "new_fault_forwards":
            fault_forward_count,

        "outer_instance_ids":
            len(
                seen_outer_ids
            ),
    }


def execute_fleet(
    *,
    gate_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    recompute_partial: bool = False,
) -> list[dict[str, Any]]:
    validated = validate_gate(
        gate_path,
        output_root=output_root,
    )

    results = []

    for shard in validated[
        "plan"
    ][
        "shards"
    ]:
        results.append(
            execute_shard(
                gate_path=gate_path,
                shard_id=shard[
                    "shard_id"
                ],
                dataset_root=dataset_root,
                output_root=output_root,
                recompute_partial=recompute_partial,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate",
        default=str(
            DEFAULT_GATE
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "validate-config"
    )

    validate = sub.add_parser(
        "validate-shard"
    )

    validate.add_argument(
        "--shard-id",
        required=True,
    )

    execute = sub.add_parser(
        "execute-shard"
    )

    execute.add_argument(
        "--shard-id",
        required=True,
    )

    execute.add_argument(
        "--dataset-root",
        default=str(
            DEFAULT_DATASET_ROOT
        ),
    )

    execute.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    fleet = sub.add_parser(
        "execute-fleet"
    )

    fleet.add_argument(
        "--dataset-root",
        default=str(
            DEFAULT_DATASET_ROOT
        ),
    )

    fleet.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "validate-config":
        validated = validate_gate(
            args.gate,
            output_root=args.output_root,
        )

        print(
            "PHASE5R_VALIDATE_CONFIG=PASS"
        )

        print(
            "COMPLETE_SHARDS=",
            len(
                validated[
                    "plan"
                ][
                    "shards"
                ]
            ),
            sep="",
        )

        print(
            "REMAINING_AUTHORIZED_SHARDS=",
            len(
                validated[
                    "remaining_shards"
                ]
            ),
            sep="",
        )

        print(
            "ACCEPTED_CANARY_REUSE_ONLY=True"
        )

        print(
            "OUTER_ARRAY_READ=False"
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

    validated = validate_gate(
        args.gate,
        output_root=args.output_root,
    )

    if args.command == "validate-shard":
        authorization = shard_authorization(
            validated,
            args.shard_id,
        )

        structure = validate_shard_structure(
            phase5d=validated[
                "phase5d"
            ],
            plan=validated[
                "plan"
            ],
            shard=authorization[
                "shard"
            ],
        )

        print(
            "PHASE5R_VALIDATE_SHARD=PASS"
        )

        print(
            "AUTHORIZATION_MODE=",
            authorization[
                "mode"
            ],
            sep="",
        )

        print(
            "EXPECTED_OUTER_INSTANCE_IDS=",
            structure[
                "expected_outer_instance_ids"
            ],
            sep="",
        )

        print(
            "EXPECTED_FAULTED_MODEL_WINDOW_EVALUATIONS=",
            structure[
                "expected_faulted_model_window_evaluations"
            ],
            sep="",
        )

        print(
            "OUTER_ARRAY_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "FAULT_EXECUTION_EXECUTED=False"
        )

        return

    if args.command == "execute-shard":
        result = execute_shard(
            gate_path=args.gate,
            shard_id=args.shard_id,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            recompute_partial=bool(
                args.recompute_partial
            ),
        )

        print(
            "PHASE5R_EXECUTE_SHARD_STATUS=",
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

        return

    results = execute_fleet(
        gate_path=args.gate,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        recompute_partial=bool(
            args.recompute_partial
        ),
    )

    print(
        "PHASE5R_EXECUTE_FLEET_COMPLETE=True"
    )

    print(
        "SHARD_RESULTS=",
        len(
            results
        ),
        sep="",
    )


if __name__ == "__main__":
    main()
