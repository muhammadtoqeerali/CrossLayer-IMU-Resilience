"""Phase-5F compute-FI outer executor/resume core v1.

This module freezes the execution-record identity and atomic resume semantics
that the later Phase-5 outer shard executor must use.

Qualification mode is synthetic/training-calibration only.

This version intentionally does NOT expose an outer execute-shard CLI yet.
No outer-test payload is read by this module during qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from compute_fi_contract import (
    FaultIdentity,
    validate_fault_identity,
)
from compute_fi_execution_harness import (
    MutationRecord,
    PairedExecution,
    tensor_bytes_sha256,
    validate_exact_single_bit_mutation,
)
from compute_fi_outer_sampling_v1 import (
    outer_instance_id,
)


SUCCESS_MARKER = "_SUCCESS.json"
TEMPORARY_SUFFIX = ".partial"

EXECUTOR_SCHEMA = "phase5f_compute_fi_outer_executor_core_v1"
SUCCESS_SCHEMA = "phase5f_compute_fi_artifact_success_v1"


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: str | Path,
) -> str:
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def canonical_jsonl_sha256(
    rows: list[Mapping[str, Any]],
) -> str:
    payload = "".join(
        canonical_json(
            dict(row)
        )
        + "\n"
        for row in rows
    ).encode(
        "utf-8"
    )

    return sha256_bytes(
        payload
    )


def phase5a_fault_id(
    identity: FaultIdentity,
) -> str:
    validate_fault_identity(
        identity
    )

    return identity.fault_id()


def build_outer_instance_id(
    *,
    sampling_instance_id_value: str,
    identity: FaultIdentity,
) -> str:
    return outer_instance_id(
        sampling_instance_id_value=sampling_instance_id_value,
        phase5a_fault_id=phase5a_fault_id(
            identity
        ),
        model_variant=identity.model_variant,
        checkpoint_seed=identity.checkpoint_seed,
    )


def tensor_has_nonfinite(
    tensor: torch.Tensor,
) -> bool:
    if tensor.is_quantized:
        tensor = tensor.dequantize()

    if not tensor.dtype.is_floating_point:
        return False

    return not bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


def mutation_record_to_json(
    mutation: MutationRecord,
) -> dict[str, Any]:
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
                int(x)
                for x in mutation.changed_indices
            ],

        "tensor_dtype":
            mutation.tensor_dtype,

        "integer_payload_dtype":
            mutation.integer_payload_dtype,

        "quantization_metadata_preserved":
            mutation.quantization_metadata_preserved,
    }


def paired_execution_record(
    *,
    shard_id: str,
    clean_cache_id: str,
    sampling_instance_id_value: str,
    parent: Mapping[str, Any],
    identity: FaultIdentity,
    pair: PairedExecution,
) -> dict[str, Any]:
    validate_fault_identity(
        identity
    )

    validate_exact_single_bit_mutation(
        pair.mutation
    )

    fault_id = phase5a_fault_id(
        identity
    )

    execution_id = build_outer_instance_id(
        sampling_instance_id_value=sampling_instance_id_value,
        identity=identity,
    )

    return {
        "schema_version":
            EXECUTOR_SCHEMA,

        "shard_id":
            shard_id,

        "clean_cache_id":
            clean_cache_id,

        "outer_instance_id":
            execution_id,

        "sampling_instance_id":
            sampling_instance_id_value,

        "phase5a_fault_id":
            fault_id,

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
            pair.input_sha256,

        "clean_output_sha256":
            tensor_bytes_sha256(
                pair.clean_output
            ),

        "faulted_output_sha256":
            tensor_bytes_sha256(
                pair.faulted_output
            ),

        "faulted_output_nonfinite":
            tensor_has_nonfinite(
                pair.faulted_output
            ),

        "mutation":
            mutation_record_to_json(
                pair.mutation
            ),
    }


def clean_output_record(
    *,
    clean_cache_id: str,
    parent: Mapping[str, Any],
    model_variant: str,
    checkpoint_seed: int,
    fold: int,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version":
            "phase5f_compute_fi_clean_output_record_v1",

        "clean_cache_id":
            clean_cache_id,

        "model_variant":
            model_variant,

        "checkpoint_seed":
            int(
                checkpoint_seed
            ),

        "fold":
            int(
                fold
            ),

        "parent":
            dict(
                parent
            ),

        "input_sha256":
            tensor_bytes_sha256(
                input_tensor
            ),

        "clean_output_sha256":
            tensor_bytes_sha256(
                output_tensor
            ),

        "clean_output_nonfinite":
            tensor_has_nonfinite(
                output_tensor
            ),
    }


def write_jsonl(
    path: str | Path,
    rows: list[Mapping[str, Any]],
) -> str:
    path = Path(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                canonical_json(
                    dict(row)
                )
                + "\n"
            )

    return sha256_file(
        path
    )


def write_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> str:
    path = Path(
        path
    )

    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return sha256_file(
        path
    )


def validate_success_marker(
    *,
    final_dir: str | Path,
    expected_artifact_id: str,
    expected_artifact_kind: str,
    expected_plan_sha256: str,
    expected_executor_sha256: str,
) -> dict[str, Any] | None:
    final_dir = Path(
        final_dir
    )

    success_path = (
        final_dir
        / SUCCESS_MARKER
    )

    if not success_path.is_file():
        return None

    success = json.loads(
        success_path.read_text()
    )

    required = {
        "schema_version":
            SUCCESS_SCHEMA,

        "status":
            "PASS",

        "artifact_id":
            expected_artifact_id,

        "artifact_kind":
            expected_artifact_kind,

        "plan_sha256":
            expected_plan_sha256,

        "executor_sha256":
            expected_executor_sha256,
    }

    for key, expected in required.items():
        if success.get(
            key
        ) != expected:
            raise ValueError(
                f"existing success marker mismatch: {key}"
            )

    output_hashes = success.get(
        "output_hashes"
    )

    if not isinstance(
        output_hashes,
        dict,
    ) or not output_hashes:
        raise ValueError(
            "success marker missing output_hashes"
        )

    for relative, expected_hash in output_hashes.items():
        target = (
            final_dir
            / relative
        )

        if not target.is_file():
            raise ValueError(
                f"success output missing: {relative}"
            )

        observed = sha256_file(
            target
        )

        if observed != expected_hash:
            raise ValueError(
                f"success output hash mismatch: {relative}"
            )

    return success


def prepare_atomic_artifact(
    *,
    output_root: str | Path,
    artifact_id: str,
    artifact_kind: str,
    plan_sha256: str,
    executor_sha256: str,
    recompute_partial: bool,
) -> tuple[
    str,
    Path | None,
    dict[str, Any] | None,
]:
    output_root = Path(
        output_root
    )

    final_dir = (
        output_root
        / artifact_id
    )

    temp_dir = (
        output_root
        / (
            artifact_id
            + TEMPORARY_SUFFIX
        )
    )

    success = validate_success_marker(
        final_dir=final_dir,
        expected_artifact_id=artifact_id,
        expected_artifact_kind=artifact_kind,
        expected_plan_sha256=plan_sha256,
        expected_executor_sha256=executor_sha256,
    )

    if success is not None:
        return (
            "reuse",
            None,
            success,
        )

    if final_dir.exists():
        if not recompute_partial:
            raise ValueError(
                "partial final artifact exists without valid success marker"
            )

        shutil.rmtree(
            final_dir
        )

    if temp_dir.exists():
        if not recompute_partial:
            raise ValueError(
                "temporary artifact exists without valid success marker"
            )

        shutil.rmtree(
            temp_dir
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    return (
        "compute",
        temp_dir,
        None,
    )


def commit_atomic_artifact(
    *,
    output_root: str | Path,
    artifact_id: str,
    artifact_kind: str,
    temp_dir: str | Path,
    plan_sha256: str,
    executor_sha256: str,
    output_hashes: Mapping[str, str],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = Path(
        output_root
    )

    temp_dir = Path(
        temp_dir
    )

    final_dir = (
        output_root
        / artifact_id
    )

    if final_dir.exists():
        raise ValueError(
            "final artifact unexpectedly exists before atomic commit"
        )

    if not temp_dir.is_dir():
        raise ValueError(
            "temporary artifact directory does not exist"
        )

    for relative, expected_hash in output_hashes.items():
        path = (
            temp_dir
            / relative
        )

        if not path.is_file():
            raise ValueError(
                f"temporary output missing: {relative}"
            )

        observed = sha256_file(
            path
        )

        if observed != expected_hash:
            raise ValueError(
                f"temporary output hash mismatch: {relative}"
            )

    success = {
        "schema_version":
            SUCCESS_SCHEMA,

        "status":
            "PASS",

        "artifact_id":
            artifact_id,

        "artifact_kind":
            artifact_kind,

        "plan_sha256":
            plan_sha256,

        "executor_sha256":
            executor_sha256,

        "output_hashes":
            dict(
                output_hashes
            ),

        "coverage":
            dict(
                coverage
            ),
    }

    os.replace(
        temp_dir,
        final_dir,
    )

    # Success marker is deliberately written LAST.
    success_path = (
        final_dir
        / SUCCESS_MARKER
    )

    if success_path.exists():
        raise ValueError(
            "success marker unexpectedly existed before final write"
        )

    write_json(
        success_path,
        success,
    )

    return success


def artifact_is_reusable(
    *,
    output_root: str | Path,
    artifact_id: str,
    artifact_kind: str,
    plan_sha256: str,
    executor_sha256: str,
) -> bool:
    success = validate_success_marker(
        final_dir=(
            Path(
                output_root
            )
            / artifact_id
        ),
        expected_artifact_id=artifact_id,
        expected_artifact_kind=artifact_kind,
        expected_plan_sha256=plan_sha256,
        expected_executor_sha256=executor_sha256,
    )

    return success is not None


def assert_unique_outer_instance_ids(
    records: list[Mapping[str, Any]],
) -> None:
    values = [
        row[
            "outer_instance_id"
        ]
        for row in records
    ]

    if len(
        values
    ) != len(
        set(
            values
        )
    ):
        raise ValueError(
            "outer_instance_id collision"
        )
