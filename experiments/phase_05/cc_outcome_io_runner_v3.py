"""Prospective CC I/O runner for the frozen Phase-5Z protocol.

The module implements integrity-checked artifact ingestion and deterministic
scenario construction. Importing it performs no I/O.

Actual prospective paths are supplied explicitly by a later execution command.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cc_outcome_analyzer_v3 import (
    AnalysisContractError,
    clean_falling_probability,
    normalise_activity_label,
    reconstruct_fault_scenario,
)


class IOErrorContractError(RuntimeError):
    """Fatal Phase-5Z I/O/integrity contract violation."""


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def _require_file(
    path: Path,
    *,
    role: str,
) -> None:
    if not path.is_file():
        raise IOErrorContractError(
            f"missing {role}: {path}"
        )


def verify_phase5r_artifact(
    artifact_dir: str | Path,
    *,
    record_filename: str,
    expected_plan_sha256: str,
    expected_executor_sha256: str,
) -> dict[str, Any]:
    """Verify hashes and success metadata before parsing record JSON."""

    artifact_dir = Path(
        artifact_dir
    )

    success_path = (
        artifact_dir
        / "_SUCCESS.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    records_path = (
        artifact_dir
        / record_filename
    )

    _require_file(
        success_path,
        role="success marker",
    )

    _require_file(
        metadata_path,
        role="metadata",
    )

    _require_file(
        records_path,
        role="record file",
    )

    success = load_json(
        success_path
    )

    if success.get(
        "status"
    ) != "PASS":
        raise IOErrorContractError(
            "artifact success marker is not PASS"
        )

    if success.get(
        "plan_sha256"
    ) != expected_plan_sha256:
        raise IOErrorContractError(
            "artifact plan hash mismatch"
        )

    if success.get(
        "executor_sha256"
    ) != expected_executor_sha256:
        raise IOErrorContractError(
            "artifact executor hash mismatch"
        )

    output_hashes = success.get(
        "output_hashes"
    )

    if not isinstance(
        output_hashes,
        Mapping,
    ):
        raise IOErrorContractError(
            "success marker missing output_hashes"
        )

    metadata_sha = sha256_file(
        metadata_path
    )

    record_sha = sha256_file(
        records_path
    )

    if output_hashes.get(
        "metadata.json"
    ) != metadata_sha:
        raise IOErrorContractError(
            "metadata hash mismatch"
        )

    if output_hashes.get(
        record_filename
    ) != record_sha:
        raise IOErrorContractError(
            "record-file hash mismatch"
        )

    return {
        "artifact_dir":
            artifact_dir,

        "success":
            success,

        "success_sha256":
            sha256_file(
                success_path
            ),

        "metadata_path":
            metadata_path,

        "metadata_sha256":
            metadata_sha,

        "records_path":
            records_path,

        "records_sha256":
            record_sha,
    }


def verify_frozen_hash_artifact(
    artifact_dir: str | Path,
    *,
    record_filename: str,
    expected_success_sha256: str,
    expected_metadata_sha256: str,
    expected_records_sha256: str,
) -> dict[str, Any]:
    """Verify an accepted canary against externally frozen exact hashes."""

    artifact_dir = Path(
        artifact_dir
    )

    success_path = (
        artifact_dir
        / "_SUCCESS.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    records_path = (
        artifact_dir
        / record_filename
    )

    _require_file(
        success_path,
        role="success marker",
    )

    _require_file(
        metadata_path,
        role="metadata",
    )

    _require_file(
        records_path,
        role="record file",
    )

    success_sha = sha256_file(
        success_path
    )

    metadata_sha = sha256_file(
        metadata_path
    )

    records_sha = sha256_file(
        records_path
    )

    if (
        success_sha
        != expected_success_sha256
    ):
        raise IOErrorContractError(
            "frozen success-marker hash mismatch"
        )

    if (
        metadata_sha
        != expected_metadata_sha256
    ):
        raise IOErrorContractError(
            "frozen metadata hash mismatch"
        )

    if (
        records_sha
        != expected_records_sha256
    ):
        raise IOErrorContractError(
            "frozen record-file hash mismatch"
        )

    return {
        "artifact_dir":
            artifact_dir,

        "success_sha256":
            success_sha,

        "metadata_path":
            metadata_path,

        "metadata_sha256":
            metadata_sha,

        "records_path":
            records_path,

        "records_sha256":
            records_sha,
    }


def read_verified_jsonl(
    verified_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Parse records only after artifact verification has succeeded."""

    path = Path(
        verified_artifact[
            "records_path"
        ]
    )

    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw in enumerate(
            handle,
            start=1,
        ):
            raw = raw.rstrip(
                "\n"
            )

            if not raw:
                raise IOErrorContractError(
                    f"blank JSONL row at line {line_number}"
                )

            try:
                row = json.loads(
                    raw
                )
            except json.JSONDecodeError as exc:
                raise IOErrorContractError(
                    f"malformed JSONL at line {line_number}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise IOErrorContractError(
                    f"non-object JSONL row at line {line_number}"
                )

            rows.append(
                row
            )

    return rows


def load_trial_labels(
    dataset_root: str | Path,
    *,
    subject: int,
    task: int,
    trial: int,
    expected_window_count: int,
) -> np.ndarray:
    path = (
        Path(dataset_root)
        / str(int(subject))
        / str(int(task))
        / str(int(trial))
        / "labels.npy"
    )

    _require_file(
        path,
        role="trial label array",
    )

    raw = np.load(
        path,
        allow_pickle=True,
    )

    if raw.ndim != 1:
        raise IOErrorContractError(
            "labels.npy must be one-dimensional"
        )

    if len(raw) != int(
        expected_window_count
    ):
        raise IOErrorContractError(
            "label length does not match frozen trial window count"
        )

    return np.asarray(
        [
            normalise_activity_label(
                value
            )
            for value in raw
        ],
        dtype=np.int8,
    )


def clean_sequence(
    clean_records: Sequence[Mapping[str, Any]],
    *,
    subject: int,
    task: int,
    trial: int,
    expected_window_count: int,
) -> np.ndarray:
    values: dict[
        int,
        float,
    ] = {}

    for row in clean_records:
        parent = row.get(
            "parent"
        )

        if not isinstance(
            parent,
            Mapping,
        ):
            raise IOErrorContractError(
                "clean record missing parent"
            )

        if (
            int(parent.get("subject"))
            != int(subject)
            or int(parent.get("task"))
            != int(task)
            or int(parent.get("trial"))
            != int(trial)
        ):
            continue

        index = parent.get(
            "window_index"
        )

        if index is None:
            raise IOErrorContractError(
                "clean record missing parent.window_index"
            )

        index = int(
            index
        )

        if index in values:
            raise IOErrorContractError(
                "duplicate clean window index"
            )

        if (
            index < 0
            or index >= int(
                expected_window_count
            )
        ):
            raise IOErrorContractError(
                "clean window index out of range"
            )

        try:
            values[
                index
            ] = clean_falling_probability(
                row
            )
        except AnalysisContractError as exc:
            raise IOErrorContractError(
                "invalid clean softmax record"
            ) from exc

    expected = set(
        range(
            int(
                expected_window_count
            )
        )
    )

    if set(
        values
    ) != expected:
        raise IOErrorContractError(
            "clean trial does not contain every frozen window exactly once"
        )

    return np.asarray(
        [
            values[index]
            for index in range(
                int(
                    expected_window_count
                )
            )
        ],
        dtype=float,
    )


def fault_identity_groups(
    fault_records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in fault_records:
        identity = row.get(
            "outer_instance_id"
        )

        if not isinstance(
            identity,
            str,
        ) or not identity:
            raise IOErrorContractError(
                "fault record missing outer_instance_id"
            )

        groups[
            identity
        ].append(
            dict(row)
        )

    return dict(
        groups
    )


def reconstruct_identity_sequence(
    clean_probabilities: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    *,
    outer_instance_id: str,
    persistence: str,
) -> np.ndarray:
    try:
        return reconstruct_fault_scenario(
            clean_probabilities,
            rows,
            outer_instance_id=outer_instance_id,
            persistence=persistence,
        )
    except AnalysisContractError as exc:
        raise IOErrorContractError(
            str(exc)
        ) from exc


def load_risk_index_csv(
    path: str | Path,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    path = Path(
        path
    )

    _require_file(
        path,
        role="frozen risk index",
    )

    lookup: dict[
        tuple[int, int, int],
        dict[str, Any],
    ] = {}

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        required = {
            "subject_id",
            "task_id",
            "trial_id",
            "fall_start_position",
            "impact_position",
        }

        if reader.fieldnames is None:
            raise IOErrorContractError(
                "risk index has no header"
            )

        if not required.issubset(
            set(
                reader.fieldnames
            )
        ):
            raise IOErrorContractError(
                "risk index missing required columns"
            )

        for row in reader:
            subject = int(
                str(
                    row[
                        "subject_id"
                    ]
                ).rsplit(
                    "_",
                    1,
                )[-1]
            )

            key = (
                subject,
                int(
                    row[
                        "task_id"
                    ]
                ),
                int(
                    row[
                        "trial_id"
                    ]
                ),
            )

            if key in lookup:
                raise IOErrorContractError(
                    "duplicate risk-index trial key"
                )

            lookup[
                key
            ] = dict(
                row
            )

    return lookup


def historical_window_ends(
    labels: Sequence[int],
    *,
    fall_start_position: int,
    window_samples: int = 30,
    stride_samples: int = 15,
) -> np.ndarray:
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    if labels.ndim != 1:
        raise IOErrorContractError(
            "labels must be one-dimensional"
        )

    seen_fall = False

    for value in labels:
        if int(value) == 1:
            seen_fall = True
        elif (
            int(value) == 0
            and seen_fall
        ):
            raise IOErrorContractError(
                "stored labels must be Activity windows followed by Falling windows"
            )
        elif int(value) not in (0, 1):
            raise IOErrorContractError(
                "stored labels are not binary"
            )

    activity_count = int(
        np.sum(
            labels == 0
        )
    )

    falling_count = int(
        np.sum(
            labels == 1
        )
    )

    ends = np.empty(
        len(labels),
        dtype=np.int64,
    )

    if activity_count:
        ends[
            :activity_count
        ] = (
            int(window_samples)
            + np.arange(
                activity_count,
                dtype=np.int64,
            )
            * int(
                stride_samples
            )
        )

    if falling_count:
        base = (
            int(
                fall_start_position
            )
            if int(
                fall_start_position
            ) >= 0
            else (
                int(
                    ends[
                        activity_count
                        - 1
                    ]
                )
                if activity_count
                else 0
            )
        )

        ends[
            activity_count:
        ] = (
            base
            + int(
                window_samples
            )
            + np.arange(
                falling_count,
                dtype=np.int64,
            )
            * int(
                stride_samples
            )
        )

    return ends


def event_row(
    *,
    probabilities: Sequence[float],
    labels: Sequence[int],
    risk_row: Mapping[str, Any] | None,
    sampling_rate_hz: float = 100.0,
    window_samples: int = 30,
    stride_samples: int = 15,
) -> dict[str, object]:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    if probabilities.shape != labels.shape:
        raise IOErrorContractError(
            "probability and label lengths differ"
        )

    risk = (
        dict(
            risk_row
        )
        if risk_row is not None
        else {}
    )

    fall_start = int(
        risk.get(
            "fall_start_position",
            -1,
        )
    )

    impact = int(
        risk.get(
            "impact_position",
            -1,
        )
    )

    falling_count = int(
        np.sum(
            labels == 1
        )
    )

    true_fall = bool(
        falling_count > 0
        or risk
    )

    ends = historical_window_ends(
        labels,
        fall_start_position=fall_start,
        window_samples=window_samples,
        stride_samples=stride_samples,
    )

    return {
        "true_event":
            (
                "FALLING"
                if true_fall
                else "ACTIVITY"
            ),

        "segment_probabilities":
            probabilities,

        "segment_labels":
            labels,

        "window_ends":
            ends,

        "sampling_rate_hz":
            float(
                risk.get(
                    "sampling_rate_hz",
                    sampling_rate_hz,
                )
            ),

        "fall_start_position":
            fall_start,

        "impact_position":
            impact,

        "event_id":
            risk.get(
                "event_id"
            ),

        "dataset_id":
            risk.get(
                "dataset_id"
            ),
    }


def validate_plan_trial(
    trial_row: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    for key in (
        "subject",
        "task",
        "trial",
        "window_count",
    ):
        if key not in trial_row:
            raise IOErrorContractError(
                f"trial inventory missing {key}"
            )

    subject = int(
        trial_row[
            "subject"
        ]
    )

    task = int(
        trial_row[
            "task"
        ]
    )

    trial = int(
        trial_row[
            "trial"
        ]
    )

    window_count = int(
        trial_row[
            "window_count"
        ]
    )

    if window_count < 1:
        raise IOErrorContractError(
            "trial window_count must be positive"
        )

    return (
        subject,
        task,
        trial,
        window_count,
    )
