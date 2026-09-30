from __future__ import annotations

import ast
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TOQEER = Path.home() / "toqeer"

ONFIELD_300 = Path(
    "/mnt/hdd16T/protechto/data/"
    "OnField/segments/"
    "300ms_50ov_npseg_filt_binary"
)

ONFIELD_400 = Path(
    "/mnt/hdd16T/protechto/data/"
    "OnField/segments/"
    "400ms_50ov_npseg_filt_binary"
)

RAW_CANDIDATES = [
    Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/"
        "OnFieldRecordings"
    ),
    Path(
        "/home/shared/data/protechto/"
        "OnFieldRecordings"
    ),
]

PROJECTS = [
    TOQEER / "Protechto-master",
    TOQEER / "Protechto_master",
    TOQEER / "Protechto-master_ori",
    TOQEER / "HR_LR_Fallings",
]

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3i_onfield_external_role_v1.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def numeric_subjects(
    root: Path,
) -> list[str]:
    if not root.is_dir():
        return []

    return sorted(
        [
            child.name
            for child
            in root.iterdir()
            if (
                child.is_dir()
                and child.name.isdigit()
            )
        ],
        key=int,
    )


def find_augmentation_constants() -> list[dict[str, Any]]:
    records = []

    seen = set()

    for project in PROJECTS:
        if not project.is_dir():
            continue

        for path in project.rglob(
            "*.py"
        ):
            if (
                ".git"
                in path.parts
                or "__pycache__"
                in path.parts
            ):
                continue

            try:
                if (
                    path.stat().st_size
                    > 500_000
                ):
                    continue
            except Exception:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            if (
                "DATA_AUGMENTATION_SUBJECTS"
                not in text
            ):
                continue

            resolved = str(
                path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(resolved)

            try:
                tree = ast.parse(
                    text
                )
            except Exception:
                tree = None

            assignments = []

            if tree is not None:
                for node in ast.walk(
                    tree
                ):
                    if not isinstance(
                        node,
                        (
                            ast.Assign,
                            ast.AnnAssign,
                        ),
                    ):
                        continue

                    targets = []

                    if isinstance(
                        node,
                        ast.Assign,
                    ):
                        targets = (
                            node.targets
                        )

                        value_node = (
                            node.value
                        )

                    else:
                        targets = [
                            node.target
                        ]

                        value_node = (
                            node.value
                        )

                    for target in targets:
                        if (
                            isinstance(
                                target,
                                ast.Name,
                            )
                            and target.id
                            == "DATA_AUGMENTATION_SUBJECTS"
                        ):
                            try:
                                value = (
                                    ast.literal_eval(
                                        value_node
                                    )
                                )
                            except Exception:
                                value = None

                            assignments.append(
                                {
                                    "line":
                                        int(
                                            node.lineno
                                        ),

                                    "value":
                                        value,

                                    "source":
                                        (
                                            ast.get_source_segment(
                                                text,
                                                node,
                                            )
                                            or ""
                                        ),
                                }
                            )

            usage = []

            for number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if (
                    "DATA_AUGMENTATION_SUBJECTS"
                    in line
                ):
                    usage.append(
                        {
                            "line":
                                number,

                            "text":
                                line.strip()[
                                    :800
                                ],
                        }
                    )

            records.append(
                {
                    "path":
                        resolved,

                    "sha256":
                        sha256_file(
                            path
                        ),

                    "assignments":
                        assignments,

                    "usage":
                        usage[:30],
                }
            )

    return records


def resolve_augmentation_subjects(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    literal_values = []

    for record in records:
        for item in record[
            "assignments"
        ]:
            value = item[
                "value"
            ]

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                literal_values.append(
                    tuple(
                        sorted(
                            [
                                str(v)
                                for v
                                in value
                            ],
                            key=lambda x:
                                int(x)
                                if x.isdigit()
                                else x,
                        )
                    )
                )

    unique = sorted(
        set(
            literal_values
        )
    )

    if len(
        unique
    ) == 1:
        resolved = list(
            unique[0]
        )

        status = (
            "SOURCE_CONFIRMED"
        )

    elif len(
        unique
    ) > 1:
        resolved = []

        status = (
            "CONFLICTING_SOURCE_VALUES"
        )

    else:
        resolved = []

        status = (
            "UNRESOLVED"
        )

    return {
        "status":
            status,

        "resolved_subjects":
            resolved,

        "literal_values":
            [
                list(
                    value
                )
                for value
                in unique
            ],
    }


def audit_processed_root(
    root: Path,
    expected_window_samples: int,
) -> dict[str, Any]:

    record: dict[str, Any] = {
        "path":
            str(
                root
            ),

        "exists":
            root.is_dir(),
    }

    if not root.is_dir():
        return record

    subjects = numeric_subjects(
        root
    )

    total_trials = 0
    total_windows = 0

    global_labels = Counter()

    subject_records = {}

    shape_errors = []

    length_errors = []

    for subject in subjects:
        subject_root = (
            root
            / subject
        )

        subject_trials = 0
        subject_windows = 0

        subject_labels = Counter()

        for segment_path in sorted(
            subject_root.rglob(
                "segments.npy"
            )
        ):
            label_path = (
                segment_path.parent
                / "labels.npy"
            )

            if not label_path.is_file():
                raise RuntimeError(
                    "Missing labels.npy for "
                    f"{segment_path}"
                )

            segments = np.load(
                segment_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            labels = np.load(
                label_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if (
                segments.size > 0
                and (
                    segments.ndim != 3
                    or tuple(
                        segments.shape[
                            1:
                        ]
                    )
                    != (
                        expected_window_samples,
                        9,
                    )
                )
            ):
                shape_errors.append(
                    {
                        "path":
                            str(
                                segment_path
                            ),

                        "shape":
                            list(
                                segments.shape
                            ),
                    }
                )

            windows = (
                int(
                    segments.shape[0]
                )
                if segments.size > 0
                else 0
            )

            if int(
                labels.shape[0]
            ) != windows:
                length_errors.append(
                    {
                        "path":
                            str(
                                segment_path.parent
                            ),

                        "windows":
                            windows,

                        "labels":
                            int(
                                labels.shape[0]
                            ),
                    }
                )

            values, counts = np.unique(
                labels,
                return_counts=True,
            )

            for value, count in zip(
                values.tolist(),
                counts.tolist(),
            ):
                value = str(
                    value
                )

                count = int(
                    count
                )

                subject_labels[
                    value
                ] += count

                global_labels[
                    value
                ] += count

            subject_trials += 1
            subject_windows += windows

        total_trials += (
            subject_trials
        )

        total_windows += (
            subject_windows
        )

        subject_records[
            subject
        ] = {
            "trial_count":
                subject_trials,

            "window_count":
                subject_windows,

            "label_counts":
                dict(
                    sorted(
                        subject_labels.items()
                    )
                ),

            "contains_activity":
                (
                    subject_labels[
                        "Activity"
                    ]
                    > 0
                ),

            "contains_falling":
                (
                    subject_labels[
                        "Falling"
                    ]
                    > 0
                ),
        }

    if shape_errors:
        raise RuntimeError(
            "Unexpected OnField segment shapes: "
            f"{shape_errors[:10]}"
        )

    if length_errors:
        raise RuntimeError(
            "OnField segment/label mismatch: "
            f"{length_errors[:10]}"
        )

    record.update(
        {
            "subject_count":
                len(
                    subjects
                ),

            "subjects":
                subjects,

            "trial_count":
                total_trials,

            "window_count":
                total_windows,

            "label_counts":
                dict(
                    sorted(
                        global_labels.items()
                    )
                ),

            "subjects_with_activity":
                [
                    subject
                    for subject, item
                    in subject_records.items()
                    if item[
                        "contains_activity"
                    ]
                ],

            "subjects_with_falling":
                [
                    subject
                    for subject, item
                    in subject_records.items()
                    if item[
                        "contains_falling"
                    ]
                ],

            "per_subject":
                subject_records,
        }
    )

    return record


def raw_tree_audit() -> list[dict[str, Any]]:
    records = []

    for root in (
        RAW_CANDIDATES
    ):
        if not root.exists():
            continue

        top_level = []

        if root.is_dir():
            for child in sorted(
                root.iterdir(),
                key=lambda value:
                    value.name,
            ):
                try:
                    top_level.append(
                        {
                            "name":
                                child.name,

                            "is_dir":
                                child.is_dir(),

                            "is_file":
                                child.is_file(),
                        }
                    )
                except Exception:
                    pass

        csv_count = sum(
            1
            for _ in root.rglob(
                "*.csv"
            )
        )

        npy_count = sum(
            1
            for _ in root.rglob(
                "*.npy"
            )
        )

        records.append(
            {
                "path":
                    str(
                        root.resolve()
                    ),

                "top_level":
                    top_level[:100],

                "csv_count":
                    csv_count,

                "npy_count":
                    npy_count,
            }
        )

    return records


def main() -> None:

    constants = (
        find_augmentation_constants()
    )

    augmentation = (
        resolve_augmentation_subjects(
            constants
        )
    )

    onfield_300 = (
        audit_processed_root(
            ONFIELD_300,
            expected_window_samples=30,
        )
    )

    onfield_400 = (
        audit_processed_root(
            ONFIELD_400,
            expected_window_samples=40,
        )
    )

    raw = raw_tree_audit()

    print()
    print(
        "AUGMENTATION CONSTANT STATUS:",
        augmentation[
            "status"
        ],
    )

    print(
        "Resolved augmentation subjects:",
        augmentation[
            "resolved_subjects"
        ],
    )

    print()
    print(
        "300-ms ONFIELD"
    )

    print(
        " exists:",
        onfield_300[
            "exists"
        ],
    )

    if onfield_300[
        "exists"
    ]:
        print(
            " subjects:",
            onfield_300[
                "subject_count"
            ],
            onfield_300[
                "subjects"
            ],
        )

        print(
            " trials:",
            onfield_300[
                "trial_count"
            ],
        )

        print(
            " windows:",
            onfield_300[
                "window_count"
            ],
        )

        print(
            " labels:",
            onfield_300[
                "label_counts"
            ],
        )

        print(
            " subjects with Falling:",
            onfield_300[
                "subjects_with_falling"
            ],
        )

        for subject, item in (
            onfield_300[
                "per_subject"
            ].items()
        ):
            print(
                " ",
                subject,
                "trials=",
                item[
                    "trial_count"
                ],
                "windows=",
                item[
                    "window_count"
                ],
                "labels=",
                item[
                    "label_counts"
                ],
            )

    print()
    print(
        "400-ms ONFIELD"
    )

    print(
        " exists:",
        onfield_400[
            "exists"
        ],
    )

    if onfield_400[
        "exists"
    ]:
        print(
            " subjects:",
            onfield_400[
                "subject_count"
            ],
            onfield_400[
                "subjects"
            ],
        )

        print(
            " trials:",
            onfield_400[
                "trial_count"
            ],
        )

        print(
            " windows:",
            onfield_400[
                "window_count"
            ],
        )

        print(
            " labels:",
            onfield_400[
                "label_counts"
            ],
        )

    augmentation_set = set(
        augmentation[
            "resolved_subjects"
        ]
    )

    processed_set = set(
        onfield_300.get(
            "subjects",
            [],
        )
    )

    external_candidates = sorted(
        processed_set
        - augmentation_set,
        key=int,
    )

    if (
        augmentation[
            "status"
        ]
        == "SOURCE_CONFIRMED"
    ):
        role_status = (
            "ROLE_RESOLVED"
        )

    else:
        role_status = (
            "ROLE_PENDING_AUGMENTATION_ID_CONFIRMATION"
        )

    external_label_capability = (
        "FALL_AND_ACTIVITY_EXTERNAL_EVALUATION"
        if any(
            onfield_300.get(
                "per_subject",
                {},
            ).get(
                subject,
                {},
            ).get(
                "contains_falling",
                False,
            )
            for subject
            in external_candidates
        )
        else "ACTIVITY_FALSE_ALARM_DOMAIN_SHIFT_ONLY"
    )

    manifest = {
        "schema":
            "crosslayer_phase3i_onfield_external_role_v1",

        "generated_utc": (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        ),

        "status":
            "PASS",

        "audit_mode":
            "READ_ONLY",

        "augmentation_constant_sources":
            constants,

        "augmentation_subject_resolution":
            augmentation,

        "onfield_300ms":
            onfield_300,

        "onfield_400ms":
            onfield_400,

        "raw_onfield_sources":
            raw,

        "role_candidate": {
            "status":
                role_status,

            "augmentation_storage_ids":
                augmentation[
                    "resolved_subjects"
                ],

            "external_subject_storage_ids":
                external_candidates,

            "external_subject_count":
                len(
                    external_candidates
                ),

            "primary_fivefold_population":
                False,

            "external_generalization_population":
                True,

            "external_label_capability":
                external_label_capability,

            "allowed_uses": [
                "external_domain_generalization",
                "external_fault_resilience_evaluation",
                "false_alarm_evaluation",
            ],

            "forbidden_before_protocol_freeze": [
                "architecture_selection",
                "hyperparameter_selection",
                "fault_severity_selection",
                "monitor_threshold_selection",
                "supervisor_tuning",
                "recovery_tuning",
                "int8_calibration",
            ],
        },

        "scientific_interpretation": {
            "onfield_required_for_primary_cv":
                False,

            "onfield_recommended_for_final_project":
                True,

            "reason":
                (
                    "Preserving independent field recordings provides "
                    "stronger external-domain evidence than mixing them "
                    "into the UniVR+KFall development folds."
                ),
        },

        "scientific_boundary": {
            "model_retrained":
                False,

            "onfield_predictions_opened":
                False,

            "onfield_used_for_selection":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,
        },
    }

    OUTPUT.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "External candidates:",
        external_candidates,
    )

    print(
        "External candidate count:",
        len(
            external_candidates
        ),
    )

    print(
        "External label capability:",
        external_label_capability,
    )

    print()
    print(
        "PHASE_3I_ONFIELD_EXTERNAL_ROLE=PASS"
    )


if __name__ == "__main__":
    main()
