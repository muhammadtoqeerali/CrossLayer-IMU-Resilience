"""Phase-6J CSC execution runtime skeleton.

Production bindings and runtime semantics are implemented here, but outer
execution remains prospectively gate-blocked.  The orchestration core can be
qualified only through synthetic injected hooks until a later authorization
artifact binds this exact runtime SHA and the complete shard estate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import csc_execution_adapter_v1 as adapter


ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False

OUTER_EXECUTION_BLOCK = (
    "PHASE6J_OUTER_EXECUTION_NOT_AUTHORIZED"
)

CLEAN_CACHE_ABORT = (
    "ABORT_SHARD_NO_PHASE6_RECOMPUTE"
)

ARTIFACT_KIND = (
    "phase6j_csc_subject_family_shard_v1"
)

SUCCESS_SCHEMA = (
    "phase6j_csc_shard_success_v1"
)

SUCCESS_MARKER = "_SUCCESS.json"

PAIR_MEMBERS_JSONL = "pair_members.jsonl"
SENSOR_REFERENCE_JSONL = "sensor_reference.jsonl"
CSC_FAULT_JSONL = "csc_fault.jsonl"
METADATA_JSON = "metadata.json"

OUTPUT_FILES = (
    PAIR_MEMBERS_JSONL,
    SENSOR_REFERENCE_JSONL,
    CSC_FAULT_JSONL,
    METADATA_JSON,
)

MODEL_VARIANT_RANK = {
    "fp32":
        0,

    "ptq_v7":
        1,
}

CHECKPOINT_SEED_RANK = {
    42:
        0,

    123:
        1,

    2025:
        2,
}

SEVERITY_RANK = {
    "L1":
        0,

    "L2":
        1,

    "L3":
        2,
}


PINNED_FILES = {
    "phase6i_binding":
        (
            "configs/evaluation/"
            "phase6i_csc_source_trial_exposure_binding_v1.json",
            "018d1869c4e6ec6cd061aef0d7b769793c626725937bfe7f337de7dafdf82c5b",
        ),

    "phase6i_exposure_helper":
        (
            "experiments/phase_06/csc_source_trial_exposure_v1.py",
            "f4db0c1dfecd5992deed7947665c5a8afc8d60db28f97457ac7ad11bc24b0aeb",
        ),

    "phase6h_architecture":
        (
            "configs/evaluation/"
            "phase6h_csc_execution_architecture_clarification_v1.json",
            "7cb6234876506fe8f127ee823550d235a04063643013eb3348d81583fdb88a26",
        ),

    "phase6g_adapter":
        (
            "experiments/phase_06/csc_execution_adapter_v1.py",
            "2d8d218b0d2214e099dab28b769eeb9fb90d6fa6420b50351b21e54cc4bd1fb5",
        ),

    "phase6e_metadata_executor":
        (
            "experiments/phase_06/csc_outer_executor_v1.py",
            "99320b5811c5e72940bcfc779fa530df88fb8b7a2f8662711b8f56e4cf5d61fd",
        ),

    "phase4h_executor":
        (
            "experiments/phase_04/sensor_fi_outer_executor_v1.py",
            "350f3356b5eed65d7e318effdb2c152f4f022e117fac73fcc0a8ee4d1871e72d",
        ),

    "phase4h_operators":
        (
            "experiments/phase_04/sensor_fi_operators.py",
            "6559ddc1fcdca352fbd09bf003e28919bb355a76b2ef52e6985a99188fd69eaa",
        ),

    "phase4h_runner":
        (
            "experiments/phase_04/sensor_fi_devcal_runner_v2.py",
            "541af4663293ae5fbaf5286fc3dea7679d511fdabb50738863880e4712a9b86a",
        ),

    "phase4h_sampling":
        (
            "experiments/phase_04/sensor_fi_sampling_v3.py",
            "a78c1dbdafdfe1b074f66c4a9bcb493aef89cfd95eb13620bfc0c9f3c5c20f74",
        ),

    "phase5_executor":
        (
            "experiments/phase_05/compute_fi_outer_fleet_executor_v1.py",
            "82424cf132a7272c9dc8a7ffd2271472b19990a209490f9dbc94bec7a7af9188",
        ),

    "phase5_harness":
        (
            "experiments/phase_05/compute_fi_execution_harness.py",
            "68418e44d50f7c8192777ffe6c7f7f5f0dc0cab2243788f627d07879ef19005b",
        ),

    "phase5_fault_only":
        (
            "experiments/phase_05/compute_fi_fault_only_execution_v1.py",
            "f8bb4095bbfedc69c24fa4cc6e79ccf4914e14fa502ce080a06de01e740112cb",
        ),

    "phase5_outer_core":
        (
            "experiments/phase_05/compute_fi_outer_executor_v1.py",
            "77ee323f71a5dbf4c54cc454a5911849aa45c5d4bdcc6d63bb908d8cece363d0",
        ),
}


RUNTIME_SYMBOLS = {
    "phase4h_executor":
        (
            "build_parent_instance_maps",
            "condition_windows",
            "historical_window_ends",
            "make_trial_metadata",
        ),

    "phase4h_operators":
        (
            "apply_fault",
        ),

    "phase4h_runner":
        (
            "rewindow_sequence",
            "load_source_trial",
            "source_trial_path",
        ),

    "phase4h_sampling":
        (
            "generate_sequence_instances",
            "generate_window_instances",
        ),

    "phase6i_exposure_helper":
        (
            "validate_pinned_exposure_contract",
            "source_trial_fault_active_support",
            "source_trial_exposed_window_indices",
            "source_trial_exposure_census",
        ),

    "phase5_executor":
        (
            "_load_outer_trial_signal",
            "_window_tensor",
            "load_model_bundle",
            "execute_fault_sequence",
            "execution_active_mask",
            "accepted_canary_clean_reusable",
        ),

    "phase5_outer_core":
        (
            "validate_success_marker",
        ),

    "phase5_fault_only":
        (
            "run_fp32_fault_only",
            "run_ptq_activation_buffer_fault_only",
            "PTQWeightFaultOnlySession",
        ),
}


MODULE_NAMES = {
    "phase4h_executor":
        "sensor_fi_outer_executor_v1",

    "phase4h_operators":
        "sensor_fi_operators",

    "phase4h_runner":
        "sensor_fi_devcal_runner_v2",

    "phase4h_sampling":
        "sensor_fi_sampling_v3",

    "phase6i_exposure_helper":
        "csc_source_trial_exposure_v1",

    "phase5_executor":
        "compute_fi_outer_fleet_executor_v1",

    "phase5_outer_core":
        "compute_fi_outer_executor_v1",

    "phase5_fault_only":
        "compute_fi_fault_only_execution_v1",
}


PAIR_MEMBER_REQUIRED = (
    "schema_version",
    "shard_id",
    "execution_request_id",
    "fold",
    "subject",
    "task",
    "trial",
    "sensor_parent_kind",
    "parent_local_index",
    "sensor_fault_id",
    "sensor_replay_id",
    "target_name",
    "target_role",
    "representation_class",
    "persistence",
    "model_variant",
    "checkpoint_seed",
    "compute_sampling_instance_id",
    "element_index",
    "bit_position",
    "onset_or_inference_index",
    "trial_window_count",
    "sensor_exposed_window_indices",
    "compute_execution_window_indices",
    "simultaneous_overlap_window_indices",
    "temporal_overlap_window_count",
    "zero_temporal_overlap",
    "sensor_reference_cache_id",
    "phase5_clean_cache_id",
    "sensor_reference_record_count",
    "csc_fault_record_count",
    "execution_status",
    "metrics_generated",
)

SENSOR_REFERENCE_REQUIRED = (
    "schema_version",
    "shard_id",
    "execution_request_id",
    "sensor_reference_cache_id",
    "fold",
    "subject",
    "task",
    "trial",
    "sensor_parent_kind",
    "parent_local_index",
    "sensor_fault_id",
    "sensor_replay_id",
    "model_variant",
    "checkpoint_seed",
    "reference_window_index",
    "input_sha256",
    "output_sha256",
    "output_nonfinite",
    "output_values",
    "output_float32_hex",
    "softmax_values",
)

CSC_FAULT_REQUIRED = (
    "schema_version",
    "shard_id",
    "execution_request_id",
    "sensor_reference_cache_id",
    "phase5_clean_cache_id",
    "fold",
    "subject",
    "task",
    "trial",
    "sensor_parent_kind",
    "parent_local_index",
    "sensor_fault_id",
    "sensor_replay_id",
    "model_variant",
    "checkpoint_seed",
    "target_name",
    "target_role",
    "representation_class",
    "persistence",
    "compute_sampling_instance_id",
    "element_index",
    "bit_position",
    "onset_or_inference_index",
    "execution_window_index",
    "sensor_active_at_execution_window",
    "simultaneous_sensor_compute_active",
    "reference_kind",
    "input_sha256",
    "faulted_output_sha256",
    "faulted_output_nonfinite",
    "faulted_output_values",
    "faulted_output_float32_hex",
    "faulted_softmax_values",
    "mutation",
)


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def sha256_file(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    Path(path).write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                canonical_json(
                    dict(row)
                )
            )
            handle.write(
                "\n"
            )


def validate_pinned_runtime_bindings() -> dict[str, Any]:
    hashes: dict[str, str] = {}

    for name, (
        relative_path,
        expected_sha,
    ) in PINNED_FILES.items():
        path = (
            ROOT
            / relative_path
        )

        actual = sha256_file(
            path
        )

        if actual != expected_sha:
            raise ValueError(
                f"pinned runtime dependency changed: {name}"
            )

        hashes[
            name
        ] = actual

    architecture = json.loads(
        (
            ROOT
            / PINNED_FILES[
                "phase6h_architecture"
            ][0]
        ).read_text(
            encoding="utf-8"
        )
    )

    if architecture[
        "execution_authorized"
    ] is not False:
        raise ValueError(
            "Phase6H unexpectedly authorizes execution"
        )

    if (
        architecture[
            "runtime_work_unit"
        ][
            "unit"
        ]
        != "one subject x one sensor-family shard"
    ):
        raise ValueError(
            "Phase6H runtime work unit changed"
        )

    if (
        architecture[
            "atomic_resume_contract"
        ][
            "success_marker_written"
        ]
        != (
            "inside temporary directory after all output files "
            "and hashes are complete"
        )
    ):
        raise ValueError(
            "Phase6H success-marker ordering changed"
        )

    if (
        architecture[
            "atomic_resume_contract"
        ][
            "commit_operation"
        ]
        != (
            "atomic directory rename/replace from .partial "
            "to final directory"
        )
    ):
        raise ValueError(
            "Phase6H commit ordering changed"
        )

    adapter.validate_pinned_bindings()

    return {
        "status":
            "PHASE6J_RUNTIME_BINDINGS_VALID",

        "hashes":
            hashes,
    }


def load_bound_runtime_modules() -> dict[str, Any]:
    """Lazy production imports.

    This function is deliberately never called before future gate validation.
    """

    validated = (
        validate_pinned_runtime_bindings()
    )

    modules: dict[str, Any] = {}

    for binding_name, module_name in MODULE_NAMES.items():
        module = importlib.import_module(
            module_name
        )

        for symbol in RUNTIME_SYMBOLS[
            binding_name
        ]:
            if not hasattr(
                module,
                symbol,
            ):
                raise ValueError(
                    f"runtime symbol missing: "
                    f"{module_name}.{symbol}"
                )

        modules[
            binding_name
        ] = module

    return {
        "validated":
            validated,

        "modules":
            modules,
    }


def shard_identity_payload(
    *,
    fold: int,
    subject: int,
    sensor_family: str,
) -> dict[str, Any]:
    sensor_family = str(
        sensor_family
    )

    if (
        not sensor_family
        or "/"
        in sensor_family
        or "\\"
        in sensor_family
    ):
        raise ValueError(
            "invalid sensor_family for shard identity"
        )

    return {
        "schema_version":
            "phase6h_csc_runtime_shard_identity_v1",

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "sensor_family":
            sensor_family,
    }


def shard_id(
    *,
    fold: int,
    subject: int,
    sensor_family: str,
) -> str:
    payload = shard_identity_payload(
        fold=fold,
        subject=subject,
        sensor_family=sensor_family,
    )

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"phase6h_csc_f{int(fold)}"
        f"_s{int(subject)}"
        f"_{sensor_family}"
        f"_{digest[:16]}"
    )


def sensor_reference_cache_id(
    *,
    fold: int,
    subject: int,
    task: int,
    trial: int,
    sensor_parent_kind: str,
    parent_local_index: int,
    sensor_fault_id: str,
    sensor_replay_id: str,
    model_variant: str,
    checkpoint_seed: int,
) -> str:
    payload = {
        "schema_version":
            "phase6h_sensor_reference_cache_id_v1",

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "task":
            int(
                task
            ),

        "trial":
            int(
                trial
            ),

        "sensor_parent_kind":
            str(
                sensor_parent_kind
            ),

        "parent_local_index":
            int(
                parent_local_index
            ),

        "sensor_fault_id":
            str(
                sensor_fault_id
            ),

        "sensor_replay_id":
            str(
                sensor_replay_id
            ),

        "model_variant":
            str(
                model_variant
            ),

        "checkpoint_seed":
            int(
                checkpoint_seed
            ),
    }

    return hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def severity_level(
    row: Mapping[str, Any],
) -> str:
    value = row.get(
        "severity"
    )

    if isinstance(
        value,
        Mapping,
    ):
        value = value.get(
            "level"
        )

    value = str(
        value
    )

    if value not in SEVERITY_RANK:
        raise ValueError(
            f"unknown severity: {value}"
        )

    return value


def canonical_pair_sort_key(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    parent_kind = str(
        row[
            "sensor_parent_kind"
        ]
    )

    parent_index = (
        -1
        if parent_kind
        == "source_trial"
        else int(
            row[
                "parent_local_index"
            ]
        )
    )

    level = severity_level(
        row
    )

    return (
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
        parent_index,
        SEVERITY_RANK[
            level
        ],
        str(
            row[
                "sensor_replay_id"
            ]
        ),
    )


def canonical_pair_member_sort_key(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    model_variant = str(
        row[
            "model_variant"
        ]
    )

    seed = int(
        row[
            "checkpoint_seed"
        ]
    )

    if model_variant not in MODEL_VARIANT_RANK:
        raise ValueError(
            "unknown model variant"
        )

    if seed not in CHECKPOINT_SEED_RANK:
        raise ValueError(
            "unknown checkpoint seed"
        )

    return (
        MODEL_VARIANT_RANK[
            model_variant
        ],
        CHECKPOINT_SEED_RANK[
            seed
        ],
        *canonical_pair_sort_key(
            row
        ),
    )


def reference_kind(
    sensor_active: bool,
) -> str:
    return (
        "phase6_sensor_reference"
        if bool(
            sensor_active
        )
        else "phase5_clean_cache"
    )


def _validate_required_fields(
    record: Mapping[str, Any],
    required: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise ValueError(
            f"{label} missing fields: {missing}"
        )


def build_sensor_reference_record(
    *,
    common: Mapping[str, Any],
    reference_window_index: int,
    output_summary: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version":
            "phase6h_csc_sensor_reference_record_v1",

        **dict(
            common
        ),

        "reference_window_index":
            int(
                reference_window_index
            ),

        "input_sha256":
            output_summary[
                "input_sha256"
            ],

        "output_sha256":
            output_summary[
                "output_sha256"
            ],

        "output_nonfinite":
            bool(
                output_summary[
                    "output_nonfinite"
                ]
            ),

        "output_values":
            output_summary[
                "output_values"
            ],

        "output_float32_hex":
            output_summary[
                "output_float32_hex"
            ],

        "softmax_values":
            output_summary[
                "softmax_values"
            ],
    }

    _validate_required_fields(
        record,
        SENSOR_REFERENCE_REQUIRED,
        label="sensor reference record",
    )

    return record


def build_csc_fault_record(
    *,
    common: Mapping[str, Any],
    request: Mapping[str, Any],
    execution_window_index: int,
    sensor_active: bool,
    fault_summary: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version":
            "phase6h_csc_fault_window_record_v1",

        **dict(
            common
        ),

        "target_name":
            request[
                "target_name"
            ],

        "target_role":
            request[
                "target_role"
            ],

        "representation_class":
            request[
                "representation_class"
            ],

        "persistence":
            request[
                "persistence"
            ],

        "compute_sampling_instance_id":
            request[
                "compute_sampling_instance_id"
            ],

        "element_index":
            int(
                request[
                    "element_index"
                ]
            ),

        "bit_position":
            int(
                request[
                    "bit_position"
                ]
            ),

        "onset_or_inference_index":
            int(
                request[
                    "onset_or_inference_index"
                ]
            ),

        "execution_window_index":
            int(
                execution_window_index
            ),

        "sensor_active_at_execution_window":
            bool(
                sensor_active
            ),

        "simultaneous_sensor_compute_active":
            bool(
                sensor_active
            ),

        "reference_kind":
            reference_kind(
                sensor_active
            ),

        "input_sha256":
            fault_summary[
                "input_sha256"
            ],

        "faulted_output_sha256":
            fault_summary[
                "faulted_output_sha256"
            ],

        "faulted_output_nonfinite":
            bool(
                fault_summary[
                    "faulted_output_nonfinite"
                ]
            ),

        "faulted_output_values":
            fault_summary[
                "faulted_output_values"
            ],

        "faulted_output_float32_hex":
            fault_summary[
                "faulted_output_float32_hex"
            ],

        "faulted_softmax_values":
            fault_summary[
                "faulted_softmax_values"
            ],

        "mutation":
            fault_summary[
                "mutation"
            ],
    }

    _validate_required_fields(
        record,
        CSC_FAULT_REQUIRED,
        label="CSC fault record",
    )

    return record


def build_pair_member_record(
    *,
    common: Mapping[str, Any],
    request: Mapping[str, Any],
    sensor_reference_record_count: int,
    csc_fault_record_count: int,
) -> dict[str, Any]:
    overlap = [
        int(
            value
        )
        for value in request[
            "simultaneous_overlap_window_indices"
        ]
    ]

    record = {
        "schema_version":
            "phase6h_csc_pair_member_record_v1",

        **dict(
            common
        ),

        "target_name":
            request[
                "target_name"
            ],

        "target_role":
            request[
                "target_role"
            ],

        "representation_class":
            request[
                "representation_class"
            ],

        "persistence":
            request[
                "persistence"
            ],

        "compute_sampling_instance_id":
            request[
                "compute_sampling_instance_id"
            ],

        "element_index":
            int(
                request[
                    "element_index"
                ]
            ),

        "bit_position":
            int(
                request[
                    "bit_position"
                ]
            ),

        "onset_or_inference_index":
            int(
                request[
                    "onset_or_inference_index"
                ]
            ),

        "trial_window_count":
            int(
                request[
                    "trial_window_count"
                ]
            ),

        "sensor_exposed_window_indices":
            [
                int(
                    value
                )
                for value in request[
                    "sensor_exposed_window_indices"
                ]
            ],

        "compute_execution_window_indices":
            [
                int(
                    value
                )
                for value in request[
                    "compute_execution_window_indices"
                ]
            ],

        "simultaneous_overlap_window_indices":
            overlap,

        "temporal_overlap_window_count":
            len(
                overlap
            ),

        "zero_temporal_overlap":
            not bool(
                overlap
            ),

        "sensor_reference_record_count":
            int(
                sensor_reference_record_count
            ),

        "csc_fault_record_count":
            int(
                csc_fault_record_count
            ),

        "execution_status":
            "COMPLETE",

        "metrics_generated":
            False,
    }

    _validate_required_fields(
        record,
        PAIR_MEMBER_REQUIRED,
        label="pair member record",
    )

    return record



def resolve_phase5_clean_cache(
    *,
    phase5_plan: Mapping[str, Any],
    fold: int,
    subject: int,
    model_variant: str,
    checkpoint_seed: int,
    phase5_output_root: str | Path,
    expected_plan_sha256: str,
    expected_executor_sha256: str,
    phase5_outer_core: Any,
) -> dict[str, Any]:
    """Resolve and validate exactly one frozen Phase-5 clean cache.

    Phase-6 never regenerates C0. Missing, ambiguous, or invalid Phase-5
    clean material aborts the shard with the frozen Phase6H reason.
    """

    matches = [
        row
        for row in phase5_plan[
            "clean_caches"
        ]
        if (
            int(
                row[
                    "fold"
                ]
            )
            == int(
                fold
            )
            and int(
                row[
                    "subject"
                ]
            )
            == int(
                subject
            )
            and str(
                row[
                    "model_variant"
                ]
            )
            == str(
                model_variant
            )
            and int(
                row[
                    "checkpoint_seed"
                ]
            )
            == int(
                checkpoint_seed
            )
        )
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            CLEAN_CACHE_ABORT
        )

    row = matches[
        0
    ]

    clean_cache_id_value = str(
        row[
            "clean_cache_id"
        ]
    )

    final_dir = (
        Path(
            phase5_output_root
        )
        / clean_cache_id_value
    )

    try:
        success = (
            phase5_outer_core.validate_success_marker(
                final_dir=final_dir,
                expected_artifact_id=clean_cache_id_value,
                expected_artifact_kind="clean_cache",
                expected_plan_sha256=str(
                    expected_plan_sha256
                ),
                expected_executor_sha256=str(
                    expected_executor_sha256
                ),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            CLEAN_CACHE_ABORT
        ) from exc

    if success is None:
        raise RuntimeError(
            CLEAN_CACHE_ABORT
        )

    return {
        "clean_cache_id":
            clean_cache_id_value,

        "plan_row":
            dict(
                row
            ),

        "final_dir":
            str(
                final_dir
            ),

        "success":
            dict(
                success
            ),
    }


def derive_sensor_exposed_window_indices(
    *,
    parent_kind: str,
    instance: Mapping[str, Any],
    parent_local_index: int,
    trial_window_count: int,
    historical_window_ends: Sequence[int] | None,
    source_length: int | None,
    exposure_helper: Any,
) -> list[int]:
    """Bind retained CSC sensor exposure to frozen Phase6I geometry."""

    exposure_helper.validate_pinned_exposure_contract()

    parent_kind = str(
        parent_kind
    )

    trial_window_count = int(
        trial_window_count
    )

    if trial_window_count <= 0:
        raise ValueError(
            "trial_window_count must be positive"
        )

    if parent_kind == "stored_window":
        index = int(
            parent_local_index
        )

        if not (
            0
            <= index
            < trial_window_count
        ):
            raise ValueError(
                "stored-window parent index outside trial"
            )

        return [
            index
        ]

    if parent_kind != "source_trial":
        raise ValueError(
            "unsupported sensor parent kind"
        )

    if (
        historical_window_ends is None
        or source_length is None
    ):
        raise ValueError(
            "source-trial exposure requires historical ends and source length"
        )

    exposed = [
        int(
            value
        )
        for value in (
            exposure_helper.source_trial_exposed_window_indices(
                instance=instance,
                historical_window_ends=historical_window_ends,
                source_length=int(
                    source_length
                ),
            )
        )
    ]

    if not exposed:
        raise ValueError(
            "retained CSC source-trial pair has no sensor exposure"
        )

    if exposed != sorted(
        set(
            exposed
        )
    ):
        raise ValueError(
            "source-trial exposed indices must be unique ascending"
        )

    if (
        exposed[
            0
        ] < 0
        or exposed[
            -1
        ] >= trial_window_count
    ):
        raise ValueError(
            "source-trial exposed index outside retained trial"
        )

    return exposed


def condition_sensor_parent(
    *,
    parent_kind: str,
    instance: Mapping[str, Any],
    parent_local_index: int,
    trial_window_count: int,
    stored_windows: Sequence[Any] | None,
    source_trial: Any | None,
    stored_labels: Any | None,
    fall_start_frame: int | None,
    historical_window_ends: Sequence[int] | None,
    source_length: int | None,
    reference_scales: Mapping[str, Any] | None,
    operators: Any,
    runner: Any,
    exposure_helper: Any,
) -> dict[str, Any]:
    """Apply the sensor fault before Phase-5 tensor conversion.

    Stored-window faults modify only their selected retained parent.
    Source-trial faults modify the source sequence and then reuse the pinned
    Phase4H-v2 historical rewindowing path.
    """

    parent_kind = str(
        parent_kind
    )

    exposed = derive_sensor_exposed_window_indices(
        parent_kind=parent_kind,
        instance=instance,
        parent_local_index=parent_local_index,
        trial_window_count=trial_window_count,
        historical_window_ends=historical_window_ends,
        source_length=source_length,
        exposure_helper=exposure_helper,
    )

    if parent_kind == "stored_window":
        if stored_windows is None:
            raise ValueError(
                "stored-window conditioning requires retained windows"
            )

        index = exposed[
            0
        ]

        corrupted, audit = operators.apply_fault(
            stored_windows[
                index
            ],
            instance,
            reference_scales=reference_scales,
        )

        conditioned = {
            index:
                corrupted,
        }

    elif parent_kind == "source_trial":
        if (
            source_trial is None
            or stored_labels is None
        ):
            raise ValueError(
                "source-trial conditioning requires source and labels"
            )

        corrupted_source, audit = operators.apply_fault(
            source_trial,
            instance,
            reference_scales=reference_scales,
        )

        rebuilt = runner.rewindow_sequence(
            corrupted_source,
            stored_labels,
            fall_start_frame=fall_start_frame,
        )

        if len(
            rebuilt
        ) != int(
            trial_window_count
        ):
            raise ValueError(
                "source-trial rewindow count differs from retained trial"
            )

        conditioned = {
            int(
                index
            ):
                rebuilt[
                    int(
                        index
                    )
                ]
            for index in exposed
        }

    else:
        raise ValueError(
            "unsupported sensor parent kind"
        )

    expected_family = str(
        instance.get(
            "family",
            "",
        )
    )

    if (
        expected_family
        and str(
            audit.get(
                "family",
                "",
            )
        )
        != expected_family
    ):
        raise ValueError(
            "sensor operator audit family mismatch"
        )

    return {
        "sensor_exposed_window_indices":
            exposed,

        "conditioned_windows_by_index":
            conditioned,

        "operator_audit":
            dict(
                audit
            ),
    }


def execute_bound_compute_fault_sequence(
    *,
    request: Mapping[str, Any],
    trial_windows: Any,
    identities: Sequence[Any],
    model_bundle: Mapping[str, Any],
    phase5_executor: Any,
) -> dict[str, Any]:
    """Execute the pinned Phase-5 compute sequence against supplied windows.

    This helper is not reachable from execute_shard while Phase6J remains
    unauthorized. Synthetic tests inject a fake Phase-5 executor.
    """

    adapter.validate_execution_request(
        request
    )

    indices = [
        int(
            value
        )
        for value in request[
            "compute_execution_window_indices"
        ]
    ]

    inputs = [
        phase5_executor._window_tensor(
            trial_windows,
            index,
        )
        for index in indices
    ]

    executions = list(
        phase5_executor.execute_fault_sequence(
            model=model_bundle[
                "model"
            ],
            inputs=inputs,
            inference_indices=indices,
            identities=identities,
            ptq_clean_state=model_bundle.get(
                "ptq_clean_state"
            ),
        )
    )

    if len(
        executions
    ) != len(
        indices
    ):
        raise ValueError(
            "Phase-5 execution cardinality mismatch"
        )

    active_mask = [
        bool(
            value
        )
        for value in (
            phase5_executor.execution_active_mask(
                executions
            )
        )
    ]

    if len(
        active_mask
    ) != len(
        indices
    ):
        raise ValueError(
            "Phase-5 active-mask cardinality mismatch"
        )

    if (
        request[
            "persistence"
        ]
        == "persistent_from_onset_until_trial_end"
        and request[
            "phase5_fault_sequence_contract"
        ][
            "persistent_all_post_onset_mutations_must_be_active"
        ]
        and not all(
            active_mask
        )
    ):
        raise ValueError(
            "persistent compute sequence contains inactive mutation"
        )

    return {
        "execution_window_indices":
            indices,

        "executions":
            executions,

        "execution_active_mask":
            active_mask,
    }


def run_model_member_stream(
    *,
    fold: int,
    subject: int,
    model_variant: str,
    checkpoint_seed: int,
    pair_jobs: Sequence[Mapping[str, Any]],
    load_model_bundle_hook: Callable[
        [Mapping[str, Any]],
        Mapping[str, Any],
    ],
    pair_executor_hook: Callable[
        [Mapping[str, Any], Mapping[str, Any]],
        Any,
    ],
) -> list[Any]:
    """Load one model bundle and execute one already-canonical member stream.

    The caller owns canonical pair enumeration. Every request is checked
    against the stream variant/seed. The frozen Phase-5
    execute_fault_sequence owns PTQ persistent-weight reset in its pinned
    finally block; this runtime does not duplicate that state machine.
    """

    stream_identity = {
        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "model_variant":
            str(
                model_variant
            ),

        "checkpoint_seed":
            int(
                checkpoint_seed
            ),
    }

    for job in pair_jobs:
        request = job[
            "request"
        ]

        if (
            str(
                request[
                    "model_variant"
                ]
            )
            != stream_identity[
                "model_variant"
            ]
            or int(
                request[
                    "checkpoint_seed"
                ]
            )
            != stream_identity[
                "checkpoint_seed"
            ]
        ):
            raise ValueError(
                "pair job does not belong to model-member stream"
            )

    bundle = load_model_bundle_hook(
        stream_identity
    )

    outputs = []

    try:
        for job in pair_jobs:
            outputs.append(
                pair_executor_hook(
                    bundle,
                    job,
                )
            )

    finally:
        release = bundle.get(
            "release"
        )

        if callable(
            release
        ):
            release()

    return outputs


@dataclasses.dataclass(
    frozen=True
)
class SyntheticHooks:
    """Injected qualification-only hooks.

    Real Phase4H/Phase5 calls are not made by synthetic qualification.
    """

    reference_forward: Callable[
            [
                int,
                Any,
            ],
            Mapping[str, Any],
        ]

    fault_sequence: Callable[
            [
                Sequence[int],
                Sequence[Any],
                Mapping[str, Any],
            ],
            Sequence[
                Mapping[str, Any]
            ],
        ]


def synthetic_execute_pair_member(
    *,
    request: Mapping[str, Any],
    shard_id_value: str,
    fold: int,
    subject: int,
    task: int,
    trial: int,
    parent_local_index: int,
    phase5_clean_cache_id: str,
    sensor_reference_cache_id_value: str,
    windows_by_index: Mapping[int, Any],
    hooks: SyntheticHooks,
) -> dict[str, Any]:
    """Exercise frozen orchestration with synthetic injected hooks only."""

    adapter.validate_execution_request(
        request
    )

    exposed = [
        int(
            value
        )
        for value in request[
            "sensor_exposed_window_indices"
        ]
    ]

    compute_indices = [
        int(
            value
        )
        for value in request[
            "compute_execution_window_indices"
        ]
    ]

    sensor_mask = [
        bool(
            value
        )
        for value in request[
            "sensor_active_mask_on_compute_sequence"
        ]
    ]

    if len(
        compute_indices
    ) != len(
        sensor_mask
    ):
        raise ValueError(
            "compute/sensor-mask length mismatch"
        )

    required_indices = set(
        exposed
    ) | set(
        compute_indices
    )

    missing_windows = sorted(
        required_indices
        - {
            int(
                key
            )
            for key in windows_by_index
        }
    )

    if missing_windows:
        raise ValueError(
            f"synthetic windows missing indices: {missing_windows}"
        )

    common = {
        "shard_id":
            str(
                shard_id_value
            ),

        "execution_request_id":
            request[
                "execution_request_id"
            ],

        "sensor_reference_cache_id":
            str(
                sensor_reference_cache_id_value
            ),

        "phase5_clean_cache_id":
            str(
                phase5_clean_cache_id
            ),

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "task":
            int(
                task
            ),

        "trial":
            int(
                trial
            ),

        "sensor_parent_kind":
            request[
                "sensor_parent_kind"
            ],

        "parent_local_index":
            int(
                parent_local_index
            ),

        "sensor_fault_id":
            request[
                "sensor_fault_id"
            ],

        "sensor_replay_id":
            request[
                "sensor_replay_id"
            ],

        "model_variant":
            request[
                "model_variant"
            ],

        "checkpoint_seed":
            int(
                request[
                    "checkpoint_seed"
                ]
            ),
    }

    # Phase6H requires all sensor-reference forwards before the
    # compute-fault sequence.
    sensor_reference_rows = []

    for index in exposed:
        summary = hooks.reference_forward(
            index,
            windows_by_index[
                index
            ],
        )

        sensor_reference_rows.append(
            build_sensor_reference_record(
                common={
                    key:
                        value
                    for key, value
                    in common.items()
                    if key
                    != "phase5_clean_cache_id"
                },
                reference_window_index=index,
                output_summary=summary,
            )
        )

    fault_inputs = [
        windows_by_index[
            index
        ]
        for index in compute_indices
    ]

    fault_summaries = list(
        hooks.fault_sequence(
            compute_indices,
            fault_inputs,
            request,
        )
    )

    if len(
        fault_summaries
    ) != len(
        compute_indices
    ):
        raise ValueError(
            "fault sequence result cardinality mismatch"
        )

    csc_fault_rows = [
        build_csc_fault_record(
            common=common,
            request=request,
            execution_window_index=index,
            sensor_active=active,
            fault_summary=summary,
        )
        for index, active, summary
        in zip(
            compute_indices,
            sensor_mask,
            fault_summaries,
        )
    ]

    pair_member = build_pair_member_record(
        common=common,
        request=request,
        sensor_reference_record_count=len(
            sensor_reference_rows
        ),
        csc_fault_record_count=len(
            csc_fault_rows
        ),
    )

    return {
        "pair_member":
            pair_member,

        "sensor_reference_rows":
            sensor_reference_rows,

        "csc_fault_rows":
            csc_fault_rows,
    }


def _success_marker_path(
    directory: Path,
) -> Path:
    return (
        directory
        / SUCCESS_MARKER
    )


def validate_phase6h_success_marker(
    *,
    final_dir: str | Path,
    expected_shard_id: str,
    expected_gate_sha256: str,
    expected_runtime_sha256: str,
) -> dict[str, Any] | None:
    final_dir = Path(
        final_dir
    )

    marker = _success_marker_path(
        final_dir
    )

    if not marker.is_file():
        return None

    try:
        success = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    expected = {
        "schema_version":
            SUCCESS_SCHEMA,

        "status":
            "PASS",

        "artifact_kind":
            ARTIFACT_KIND,

        "shard_id":
            expected_shard_id,

        "gate_sha256":
            expected_gate_sha256,

        "runtime_sha256":
            expected_runtime_sha256,
    }

    for key, value in expected.items():
        if success.get(
            key
        ) != value:
            return None

    output_hashes = success.get(
        "output_hashes"
    )

    if not isinstance(
        output_hashes,
        Mapping,
    ):
        return None

    if set(
        output_hashes
    ) != set(
        OUTPUT_FILES
    ):
        return None

    for relative, expected_hash in output_hashes.items():
        path = (
            final_dir
            / str(
                relative
            )
        )

        if (
            not path.is_file()
            or sha256_file(
                path
            )
            != expected_hash
        ):
            return None

    return dict(
        success
    )


def prepare_phase6h_artifact(
    *,
    output_root: str | Path,
    shard_id_value: str,
    gate_sha256: str,
    runtime_sha256: str,
    recompute_partial: bool,
) -> tuple[
    str,
    Path | None,
    dict[str, Any] | None,
]:
    shard_root = (
        Path(
            output_root
        )
        / "shards"
    )

    final_dir = (
        shard_root
        / shard_id_value
    )

    partial_dir = (
        shard_root
        / (
            shard_id_value
            + ".partial"
        )
    )

    success = validate_phase6h_success_marker(
        final_dir=final_dir,
        expected_shard_id=shard_id_value,
        expected_gate_sha256=gate_sha256,
        expected_runtime_sha256=runtime_sha256,
    )

    if success is not None:
        return (
            "reuse",
            None,
            success,
        )

    for path in (
        final_dir,
        partial_dir,
    ):
        if not path.exists():
            continue

        if not recompute_partial:
            raise ValueError(
                "invalid final/partial artifact exists; "
                "recompute_partial required"
            )

        if path.is_dir():
            shutil.rmtree(
                path
            )
        else:
            path.unlink()

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    return (
        "compute",
        partial_dir,
        None,
    )


def commit_phase6h_artifact(
    *,
    partial_dir: str | Path,
    final_dir: str | Path,
    shard_id_value: str,
    gate_sha256: str,
    runtime_sha256: str,
    output_hashes: Mapping[str, str],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    partial_dir = Path(
        partial_dir
    )

    final_dir = Path(
        final_dir
    )

    if final_dir.exists():
        raise ValueError(
            "final shard exists before commit"
        )

    if not partial_dir.is_dir():
        raise ValueError(
            "partial shard directory missing"
        )

    if set(
        output_hashes
    ) != set(
        OUTPUT_FILES
    ):
        raise ValueError(
            "output hash set differs from frozen Phase6H output set"
        )

    for relative, expected_hash in output_hashes.items():
        path = (
            partial_dir
            / relative
        )

        if not path.is_file():
            raise ValueError(
                f"partial output missing: {relative}"
            )

        if sha256_file(
            path
        ) != expected_hash:
            raise ValueError(
                f"partial output hash mismatch: {relative}"
            )

    success = {
        "schema_version":
            SUCCESS_SCHEMA,

        "status":
            "PASS",

        "artifact_kind":
            ARTIFACT_KIND,

        "shard_id":
            str(
                shard_id_value
            ),

        "gate_sha256":
            str(
                gate_sha256
            ),

        "runtime_sha256":
            str(
                runtime_sha256
            ),

        "output_hashes":
            dict(
                output_hashes
            ),

        "coverage":
            dict(
                coverage
            ),
    }

    marker = _success_marker_path(
        partial_dir
    )

    if marker.exists():
        raise ValueError(
            "success marker unexpectedly exists in partial shard"
        )

    # Phase6H freezes this exact order:
    # 1. complete outputs + hashes
    # 2. write success marker inside .partial
    # 3. rename directory atomically
    write_json(
        marker,
        success,
    )

    if not marker.is_file():
        raise RuntimeError(
            "success marker write failed before atomic rename"
        )

    os.replace(
        partial_dir,
        final_dir,
    )

    return success


def validate_future_gate(
    *,
    gate_path: str | Path,
    runtime_sha256: str,
    requested_shard_id: str,
) -> dict[str, Any]:
    path = Path(
        gate_path
    )

    if not path.is_file():
        raise RuntimeError(
            OUTER_EXECUTION_BLOCK
        )

    gate = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if gate.get(
        "execution_authorized"
    ) is not True:
        raise RuntimeError(
            OUTER_EXECUTION_BLOCK
        )

    if gate.get(
        "runtime_sha256"
    ) != runtime_sha256:
        raise RuntimeError(
            "authorization gate does not bind this runtime"
        )

    authorized = gate.get(
        "authorized_shard_ids"
    )

    if (
        not isinstance(
            authorized,
            list,
        )
        or requested_shard_id
        not in authorized
    ):
        raise RuntimeError(
            "requested shard not authorized by gate"
        )

    return gate


def execute_shard(
    *,
    gate_path: str | Path,
    shard_id_value: str,
    dataset_root: str | Path,
    phase5_output_root: str | Path,
    output_root: str | Path,
    recompute_partial: bool = False,
) -> dict[str, Any]:
    """Public production entrypoint.

    A later gate must bind this exact runtime SHA before this function may
    progress far enough to import the heavy runtime stack.
    """

    del dataset_root
    del phase5_output_root
    del output_root
    del recompute_partial

    runtime_sha = sha256_file(
        Path(
            __file__
        )
    )

    validate_future_gate(
        gate_path=gate_path,
        runtime_sha256=runtime_sha,
        requested_shard_id=shard_id_value,
    )

    # Even a syntactically plausible future gate cannot authorize execution
    # before the runtime implementation itself is frozen and the later
    # authorization phase explicitly changes this constant prospectively.
    if EXECUTION_AUTHORIZED is not True:
        raise RuntimeError(
            OUTER_EXECUTION_BLOCK
        )

    # Heavy imports are deliberately below both gates.
    load_bound_runtime_modules()

    raise RuntimeError(
        "PHASE6J_AUTHORIZED_EXECUTION_BODY_NOT_YET_RELEASED"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "validate-bindings"
    )

    sub.add_parser(
        "execution-status"
    )

    execute = sub.add_parser(
        "execute-shard"
    )

    execute.add_argument(
        "--gate-path",
        required=True,
    )

    execute.add_argument(
        "--shard-id",
        required=True,
    )

    execute.add_argument(
        "--dataset-root",
        required=True,
    )

    execute.add_argument(
        "--phase5-output-root",
        required=True,
    )

    execute.add_argument(
        "--output-root",
        required=True,
    )

    execute.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "validate-bindings":
        observed = validate_pinned_runtime_bindings()

        print(
            "PHASE6J_RUNTIME_BINDINGS=PASS"
        )

        print(
            "PINNED_FILE_COUNT="
            + str(
                len(
                    observed[
                        "hashes"
                    ]
                )
            )
        )

        return 0

    if args.command == "execution-status":
        print(
            "PHASE6J_EXECUTION_AUTHORIZED="
            + str(
                EXECUTION_AUTHORIZED
            )
        )

        print(
            "OUTER_ARRAY_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "SENSOR_FAULT_EXECUTED=False"
        )

        print(
            "COMPUTE_FAULT_EXECUTED=False"
        )

        print(
            "MODEL_FORWARD_EXECUTED=False"
        )

        return 0

    execute_shard(
        gate_path=args.gate_path,
        shard_id_value=args.shard_id,
        dataset_root=args.dataset_root,
        phase5_output_root=args.phase5_output_root,
        output_root=args.output_root,
        recompute_partial=args.recompute_partial,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
