from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PRIMARY = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

UNIVR_ANNOTATION_CANDIDATES = [
    Path(
        "/mnt/hdd16T/protechto/"
        "UniVrFall_oriented"
    ),
    Path(
        "/mnt/hdd16T/protechto/"
        "UniVrFallOriginalDataset"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "uniVr-dataset"
    ),
]

KFALL_ANNOTATION_CANDIDATES = [
    Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/KFall"
    ),
    Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/KFall_oriented"
    ),
]

EVENT_INDEX_CANDIDATES = [
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "HR_LR_Fallings/risk_data_generated/"
        "FALL_EVENT_INDEX_COMBINED_LABELED.csv"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori/risk_data_generated/"
        "FALL_EVENT_INDEX_COMBINED_LABELED.csv"
    ),
]

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3k_event_annotation_trial_mapping_v1.json"
)

SUBJECT_FILE_PATTERNS = [
    re.compile(
        r"^S(?P<subject>\d+)_label\.xlsx$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^SA(?P<subject>\d+)_label\.xlsx$",
        re.IGNORECASE,
    ),
]

EXPECTED_UNIVR_SUBJECTS = {
    value
    for value
    in range(
        9,
        38,
    )
}

EXPECTED_KFALL_SUBJECTS = {
    6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 35,
    36, 37, 38,
}


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


def normalize_column(
    value: Any,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def subject_from_filename(
    path: Path,
) -> int | None:

    for pattern in (
        SUBJECT_FILE_PATTERNS
    ):
        match = pattern.match(
            path.name
        )

        if match:
            return int(
                match.group(
                    "subject"
                )
            )

    return None


def is_missing(
    value: Any,
) -> bool:

    if value is None:
        return True

    try:
        if math.isnan(
            float(value)
        ):
            return True
    except Exception:
        pass

    text = str(value).strip()

    return (
        text == ""
        or text.lower()
        in {
            "nan",
            "none",
            "null",
        }
    )


def integer_value(
    value: Any,
) -> int | None:

    if is_missing(
        value
    ):
        return None

    try:
        numeric = float(
            value
        )

        rounded = int(
            round(
                numeric
            )
        )

        if abs(
            numeric
            - rounded
        ) > 1e-6:
            return None

        return rounded

    except Exception:
        text = str(
            value
        ).strip()

        match = re.search(
            r"-?\d+",
            text,
        )

        if not match:
            return None

        return int(
            match.group(0)
        )


def detect_columns(
    columns,
) -> dict[str, str | None]:

    normalized = {
        str(column):
            normalize_column(
                column
            )
        for column
        in columns
    }

    def exact_or_contains(
        exact=(),
        contains=(),
    ):
        for original, value in (
            normalized.items()
        ):
            if value in exact:
                return original

        for original, value in (
            normalized.items()
        ):
            if all(
                token
                in value
                for token
                in contains
            ):
                return original

        return None

    task = exact_or_contains(
        exact=(
            "task",
            "taskid",
            "taskcode",
        ),
        contains=(
            "task",
        ),
    )

    trial = exact_or_contains(
        exact=(
            "trial",
            "trialid",
            "trialno",
            "trialnumber",
        ),
        contains=(
            "trial",
        ),
    )

    onset = None

    for original, value in (
        normalized.items()
    ):
        if (
            (
                "onset"
                in value
                and "frame"
                in value
            )
            or (
                "fallonset"
                in value
            )
            or (
                "startfall"
                in value
            )
        ):
            onset = original
            break

    impact = None

    for original, value in (
        normalized.items()
    ):
        if (
            (
                "impact"
                in value
                and "frame"
                in value
            )
            or (
                "fallimpact"
                in value
            )
            or (
                "endfall"
                in value
            )
        ):
            impact = original
            break

    description = None

    for original, value in (
        normalized.items()
    ):
        if (
            "description"
            in value
            or "activityname"
            in value
            or "taskdescription"
            in value
        ):
            description = original
            break

    return {
        "task":
            task,

        "trial":
            trial,

        "onset":
            onset,

        "impact":
            impact,

        "description":
            description,
    }


def workbook_inventory(
    root: Path,
) -> dict[str, Any]:

    record = {
        "root":
            str(
                root
            ),

        "exists":
            root.is_dir(),
    }

    if not root.is_dir():
        return record

    files = []

    for path in sorted(
        root.rglob(
            "*.xlsx"
        )
    ):
        subject = (
            subject_from_filename(
                path
            )
        )

        if subject is None:
            continue

        files.append(
            {
                "path":
                    str(
                        path.resolve()
                    ),

                "subject":
                    subject,

                "sha256":
                    sha256_file(
                        path
                    ),

                "bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

    record[
        "workbooks"
    ] = files

    record[
        "workbook_count"
    ] = len(
        files
    )

    record[
        "subjects"
    ] = sorted(
        {
            item[
                "subject"
            ]
            for item
            in files
        }
    )

    record[
        "subject_count"
    ] = len(
        record[
            "subjects"
        ]
    )

    return record


def choose_annotation_root(
    candidates: list[Path],
    expected_subjects: set[int],
) -> dict[str, Any]:

    inventories = [
        workbook_inventory(
            root
        )
        for root
        in candidates
    ]

    qualifying = []

    for item in inventories:
        if not item.get(
            "exists",
            False,
        ):
            continue

        subjects = set(
            item.get(
                "subjects",
                [],
            )
        )

        if (
            subjects
            == expected_subjects
        ):
            qualifying.append(
                item
            )

    if not qualifying:
        return {
            "status":
                "NO_EXACT_SUBJECT_SET",

            "inventories":
                inventories,
        }

    # Candidate ordering is deliberate provenance preference.
    chosen = qualifying[
        0
    ]

    return {
        "status":
            "SELECTED",

        "selected":
            chosen,

        "all_inventories":
            inventories,
    }


def read_annotation_rows(
    dataset: str,
    selection: dict[str, Any],
) -> dict[str, Any]:

    if selection[
        "status"
    ] != "SELECTED":
        return {
            "status":
                "UNAVAILABLE",
        }

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas required for annotation audit"
        ) from exc

    workbooks = selection[
        "selected"
    ][
        "workbooks"
    ]

    records = []

    workbook_summaries = []

    column_signatures = Counter()

    for workbook in (
        workbooks
    ):
        path = Path(
            workbook[
                "path"
            ]
        )

        subject = int(
            workbook[
                "subject"
            ]
        )

        frame = pd.read_excel(
            path
        )

        columns = [
            str(
                value
            )
            for value
            in frame.columns
        ]

        detected = (
            detect_columns(
                frame.columns
            )
        )

        signature = (
            "|".join(
                columns
            )
        )

        column_signatures[
            signature
        ] += 1

        workbook_summaries.append(
            {
                "path":
                    str(
                        path
                    ),

                "subject":
                    subject,

                "rows":
                    int(
                        len(
                            frame
                        )
                    ),

                "columns":
                    columns,

                "detected_columns":
                    detected,
            }
        )

        task_col = (
            detected[
                "task"
            ]
        )

        trial_col = (
            detected[
                "trial"
            ]
        )

        onset_col = (
            detected[
                "onset"
            ]
        )

        impact_col = (
            detected[
                "impact"
            ]
        )

        if (
            task_col is None
            or trial_col is None
        ):
            raise RuntimeError(
                "Cannot identify task/trial columns in "
                f"{path}: {columns}"
            )

        for index, row in (
            frame.iterrows()
        ):
            task = integer_value(
                row[
                    task_col
                ]
            )

            trial = integer_value(
                row[
                    trial_col
                ]
            )

            if (
                task is None
                or trial is None
            ):
                continue

            onset = (
                integer_value(
                    row[
                        onset_col
                    ]
                )
                if onset_col
                is not None
                else None
            )

            impact = (
                integer_value(
                    row[
                        impact_col
                    ]
                )
                if impact_col
                is not None
                else None
            )

            description = None

            if (
                detected[
                    "description"
                ]
                is not None
            ):
                value = row[
                    detected[
                        "description"
                    ]
                ]

                if not is_missing(
                    value
                ):
                    description = str(
                        value
                    )

            records.append(
                {
                    "dataset":
                        dataset,

                    "subject":
                        subject,

                    "task":
                        task,

                    "trial":
                        trial,

                    "onset":
                        onset,

                    "impact":
                        impact,

                    "description":
                        description,

                    "workbook":
                        str(
                            path
                        ),

                    "excel_row":
                        int(
                            index
                        )
                        + 2,
                }
            )

    event_rows = [
        row
        for row
        in records
        if (
            row[
                "onset"
            ]
            is not None
            and row[
                "impact"
            ]
            is not None
        )
    ]

    partial_rows = [
        row
        for row
        in records
        if (
            (
                row[
                    "onset"
                ]
                is None
            )
            != (
                row[
                    "impact"
                ]
                is None
            )
        )
    ]

    return {
        "status":
            "PASS",

        "workbook_count":
            len(
                workbooks
            ),

        "all_annotation_row_count":
            len(
                records
            ),

        "complete_onset_impact_row_count":
            len(
                event_rows
            ),

        "partial_onset_impact_row_count":
            len(
                partial_rows
            ),

        "column_signatures":
            dict(
                column_signatures
            ),

        "workbook_summaries":
            workbook_summaries,

        "records":
            records,

        "complete_event_rows":
            event_rows,

        "partial_event_rows":
            partial_rows,
    }


def processed_trials() -> dict[str, Any]:

    records = []

    subject_counts = Counter()

    dataset_counts = Counter()

    falling_trial_counts = Counter()

    if not PRIMARY.is_dir():
        raise RuntimeError(
            f"Primary root missing: {PRIMARY}"
        )

    for segment_path in sorted(
        PRIMARY.rglob(
            "segments.npy"
        )
    ):
        trial_dir = (
            segment_path.parent
        )

        relative = (
            trial_dir.relative_to(
                PRIMARY
            )
        )

        if len(
            relative.parts
        ) != 3:
            raise RuntimeError(
                "Unexpected primary trial path: "
                f"{relative}"
            )

        storage_subject, task, trial = (
            relative.parts
        )

        subject_value = int(
            storage_subject
        )

        if (
            9 <= subject_value <= 37
        ):
            dataset = "UNIVR"

            source_subject = (
                subject_value
            )

            canonical_subject = (
                f"UNIVR_{source_subject:02d}"
            )

        else:
            dataset = "KFALL"

            source_subject = (
                subject_value
                - 100
            )

            canonical_subject = (
                f"KFALL_{source_subject:02d}"
            )

        label_path = (
            trial_dir
            / "labels.npy"
        )

        labels = np.load(
            label_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        unique, counts = np.unique(
            labels,
            return_counts=True,
        )

        label_counts = {
            str(
                label
            ):
                int(
                    count
                )
            for label, count
            in zip(
                unique.tolist(),
                counts.tolist(),
            )
        }

        contains_falling = (
            label_counts.get(
                "Falling",
                0,
            )
            > 0
        )

        record = {
            "dataset":
                dataset,

            "storage_subject":
                storage_subject,

            "source_subject":
                source_subject,

            "canonical_subject":
                canonical_subject,

            "task":
                int(
                    task
                ),

            "trial":
                int(
                    trial
                ),

            "trial_key":
                (
                    f"{dataset}:"
                    f"S{source_subject}:"
                    f"T{int(task)}:"
                    f"R{int(trial)}"
                ),

            "processed_path":
                str(
                    relative
                ),

            "window_count":
                int(
                    labels.shape[0]
                ),

            "label_counts":
                label_counts,

            "contains_falling":
                contains_falling,
        }

        records.append(
            record
        )

        subject_counts[
            canonical_subject
        ] += 1

        dataset_counts[
            dataset
        ] += 1

        if contains_falling:
            falling_trial_counts[
                dataset
            ] += 1

    return {
        "trial_count":
            len(
                records
            ),

        "dataset_trial_counts":
            dict(
                dataset_counts
            ),

        "processed_falling_trial_counts":
            dict(
                falling_trial_counts
            ),

        "subject_trial_counts":
            dict(
                sorted(
                    subject_counts.items()
                )
            ),

        "records":
            records,
    }


def trial_key(
    dataset: str,
    subject: int,
    task: int,
    trial: int,
) -> tuple[
    str,
    int,
    int,
    int,
]:

    return (
        dataset,
        int(
            subject
        ),
        int(
            task
        ),
        int(
            trial
        ),
    )


def reconcile(
    processed: dict[str, Any],
    annotations: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:

    processed_map = {}

    for row in (
        processed[
            "records"
        ]
    ):
        key = trial_key(
            row[
                "dataset"
            ],
            row[
                "source_subject"
            ],
            row[
                "task"
            ],
            row[
                "trial"
            ],
        )

        if key in (
            processed_map
        ):
            raise RuntimeError(
                f"Duplicate processed trial key: {key}"
            )

        processed_map[
            key
        ] = row

    annotation_by_key = (
        defaultdict(
            list
        )
    )

    all_event_rows = []

    for dataset, audit in (
        annotations.items()
    ):
        for row in audit.get(
            "complete_event_rows",
            [],
        ):
            key = trial_key(
                dataset,
                row[
                    "subject"
                ],
                row[
                    "task"
                ],
                row[
                    "trial"
                ],
            )

            annotation_by_key[
                key
            ].append(
                row
            )

            all_event_rows.append(
                row
            )

    duplicate_annotation_keys = []

    for key, rows in (
        annotation_by_key.items()
    ):
        if len(
            rows
        ) > 1:
            duplicate_annotation_keys.append(
                {
                    "key":
                        list(
                            key
                        ),

                    "count":
                        len(
                            rows
                        ),

                    "rows":
                        rows,
                }
            )

    mapped_events = []

    annotation_without_processed = []

    invalid_order = []

    event_by_dataset = Counter()

    mapped_by_dataset = Counter()

    for key, rows in (
        annotation_by_key.items()
    ):
        event_by_dataset[
            key[0]
        ] += len(
            rows
        )

        processed_row = (
            processed_map.get(
                key
            )
        )

        if processed_row is None:
            for annotation in rows:
                annotation_without_processed.append(
                    {
                        "key":
                            list(
                                key
                            ),

                        "annotation":
                            annotation,
                    }
                )

            continue

        for annotation in rows:
            if (
                annotation[
                    "onset"
                ]
                >= annotation[
                    "impact"
                ]
            ):
                invalid_order.append(
                    {
                        "key":
                            list(
                                key
                            ),

                        "annotation":
                            annotation,
                    }
                )

            mapped_by_dataset[
                key[0]
            ] += 1

            mapped_events.append(
                {
                    "key":
                        list(
                            key
                        ),

                    "processed":
                        processed_row,

                    "annotation":
                        annotation,

                    "processed_contains_falling":
                        processed_row[
                            "contains_falling"
                        ],
                }
            )

    processed_falling_without_event = []

    processed_falling_counts = Counter()

    processed_falling_mapped = Counter()

    for key, row in (
        processed_map.items()
    ):
        if not row[
            "contains_falling"
        ]:
            continue

        processed_falling_counts[
            key[0]
        ] += 1

        if key in (
            annotation_by_key
        ):
            processed_falling_mapped[
                key[0]
            ] += 1

        else:
            processed_falling_without_event.append(
                row
            )

    mapped_event_without_falling_windows = [
        row
        for row
        in mapped_events
        if not row[
            "processed_contains_falling"
        ]
    ]

    return {
        "annotation_complete_event_counts":
            dict(
                event_by_dataset
            ),

        "mapped_complete_event_counts":
            dict(
                mapped_by_dataset
            ),

        "processed_falling_trial_counts":
            dict(
                processed_falling_counts
            ),

        "processed_falling_trials_with_event":
            dict(
                processed_falling_mapped
            ),

        "mapped_event_count":
            len(
                mapped_events
            ),

        "annotation_without_processed_count":
            len(
                annotation_without_processed
            ),

        "annotation_without_processed":
            annotation_without_processed,

        "duplicate_annotation_trial_key_count":
            len(
                duplicate_annotation_keys
            ),

        "duplicate_annotation_trial_keys":
            duplicate_annotation_keys,

        "invalid_onset_impact_order_count":
            len(
                invalid_order
            ),

        "invalid_onset_impact_order":
            invalid_order,

        "processed_falling_without_event_count":
            len(
                processed_falling_without_event
            ),

        "processed_falling_without_event":
            processed_falling_without_event,

        "mapped_event_without_falling_window_count":
            len(
                mapped_event_without_falling_windows
            ),

        "mapped_event_without_falling_windows":
            mapped_event_without_falling_windows,

        "mapped_events":
            mapped_events,
    }


def inspect_event_indexes() -> list[
    dict[str, Any]
]:

    records = []

    try:
        import pandas as pd
    except Exception:
        return records

    seen = set()

    for candidate in (
        EVENT_INDEX_CANDIDATES
    ):
        if not candidate.is_file():
            continue

        resolved = str(
            candidate.resolve()
        )

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        try:
            frame = pd.read_csv(
                candidate
            )

            columns = [
                str(
                    value
                )
                for value
                in frame.columns
            ]

            normalized = {
                column:
                    normalize_column(
                        column
                    )
                for column
                in columns
            }

            timing_columns = [
                column
                for column, value
                in normalized.items()
                if any(
                    token
                    in value
                    for token
                    in (
                        "onset",
                        "impact",
                        "timestamp",
                        "time",
                        "sample",
                        "frame",
                    )
                )
            ]

            records.append(
                {
                    "path":
                        resolved,

                    "sha256":
                        sha256_file(
                            candidate
                        ),

                    "row_count":
                        int(
                            len(
                                frame
                            )
                        ),

                    "columns":
                        columns,

                    "timing_columns":
                        timing_columns,

                    "first_rows":
                        (
                            frame.head(
                                5
                            )
                            .where(
                                frame.notna(),
                                None,
                            )
                            .to_dict(
                                orient="records"
                            )
                        ),
                }
            )

        except Exception as exc:
            records.append(
                {
                    "path":
                        resolved,

                    "parse_error":
                        repr(
                            exc
                        ),
                }
            )

    return records


def main() -> None:

    print(
        "AUDITING PRIMARY PROCESSED TRIALS"
    )

    processed = (
        processed_trials()
    )

    print(
        "Processed trials:",
        processed[
            "trial_count"
        ],
    )

    print(
        "Dataset trial counts:",
        processed[
            "dataset_trial_counts"
        ],
    )

    print(
        "Processed Falling trial counts:",
        processed[
            "processed_falling_trial_counts"
        ],
    )

    if (
        processed[
            "trial_count"
        ]
        != 6309
    ):
        raise RuntimeError(
            "Expected 6309 primary processed trials"
        )

    if (
        processed[
            "dataset_trial_counts"
        ].get(
            "UNIVR"
        )
        != 1234
    ):
        raise RuntimeError(
            "Expected 1234 UniVR processed trials"
        )

    if (
        processed[
            "dataset_trial_counts"
        ].get(
            "KFALL"
        )
        != 5075
    ):
        raise RuntimeError(
            "Expected 5075 KFall processed trials"
        )

    print()
    print(
        "LOCATING UNIVR ANNOTATIONS"
    )

    univr_selection = (
        choose_annotation_root(
            UNIVR_ANNOTATION_CANDIDATES,
            EXPECTED_UNIVR_SUBJECTS,
        )
    )

    print(
        "UniVR selection status:",
        univr_selection[
            "status"
        ],
    )

    if (
        univr_selection[
            "status"
        ]
        != "SELECTED"
    ):
        print(
            json.dumps(
                univr_selection,
                indent=2,
            )
        )

        raise RuntimeError(
            "Could not select exact 29-subject UniVR annotation root"
        )

    print(
        "UniVR annotation root:",
        univr_selection[
            "selected"
        ][
            "root"
        ],
    )

    print(
        "UniVR workbooks:",
        univr_selection[
            "selected"
        ][
            "workbook_count"
        ],
    )

    print()
    print(
        "LOCATING KFALL ANNOTATIONS"
    )

    kfall_selection = (
        choose_annotation_root(
            KFALL_ANNOTATION_CANDIDATES,
            EXPECTED_KFALL_SUBJECTS,
        )
    )

    print(
        "KFall selection status:",
        kfall_selection[
            "status"
        ],
    )

    if (
        kfall_selection[
            "status"
        ]
        != "SELECTED"
    ):
        print(
            json.dumps(
                kfall_selection,
                indent=2,
            )
        )

        raise RuntimeError(
            "Could not select exact 32-subject KFall annotation root"
        )

    print(
        "KFall annotation root:",
        kfall_selection[
            "selected"
        ][
            "root"
        ],
    )

    print(
        "KFall workbooks:",
        kfall_selection[
            "selected"
        ][
            "workbook_count"
        ],
    )

    print()
    print(
        "READING ANNOTATION WORKBOOKS"
    )

    univr = (
        read_annotation_rows(
            "UNIVR",
            univr_selection,
        )
    )

    kfall = (
        read_annotation_rows(
            "KFALL",
            kfall_selection,
        )
    )

    print(
        "UniVR all annotation rows:",
        univr[
            "all_annotation_row_count"
        ],
    )

    print(
        "UniVR complete onset+impact rows:",
        univr[
            "complete_onset_impact_row_count"
        ],
    )

    print(
        "UniVR partial onset/impact rows:",
        univr[
            "partial_onset_impact_row_count"
        ],
    )

    print(
        "KFall all annotation rows:",
        kfall[
            "all_annotation_row_count"
        ],
    )

    print(
        "KFall complete onset+impact rows:",
        kfall[
            "complete_onset_impact_row_count"
        ],
    )

    print(
        "KFall partial onset/impact rows:",
        kfall[
            "partial_onset_impact_row_count"
        ],
    )

    reconciliation = (
        reconcile(
            processed,
            {
                "UNIVR":
                    univr,

                "KFALL":
                    kfall,
            },
        )
    )

    print()
    print(
        "EVENT / PROCESSED TRIAL RECONCILIATION"
    )

    print(
        "Annotation complete-event counts:",
        reconciliation[
            "annotation_complete_event_counts"
        ],
    )

    print(
        "Mapped event counts:",
        reconciliation[
            "mapped_complete_event_counts"
        ],
    )

    print(
        "Processed Falling trial counts:",
        reconciliation[
            "processed_falling_trial_counts"
        ],
    )

    print(
        "Processed Falling trials with event:",
        reconciliation[
            "processed_falling_trials_with_event"
        ],
    )

    print(
        "Annotation without processed trial:",
        reconciliation[
            "annotation_without_processed_count"
        ],
    )

    print(
        "Duplicate annotation trial keys:",
        reconciliation[
            "duplicate_annotation_trial_key_count"
        ],
    )

    print(
        "Invalid onset >= impact:",
        reconciliation[
            "invalid_onset_impact_order_count"
        ],
    )

    print(
        "Processed Falling trial without event:",
        reconciliation[
            "processed_falling_without_event_count"
        ],
    )

    print(
        "Mapped event but no Falling window:",
        reconciliation[
            "mapped_event_without_falling_window_count"
        ],
    )

    event_indexes = (
        inspect_event_indexes()
    )

    print()
    print(
        "CURATED EVENT INDEX ASSETS:",
        len(
            event_indexes
        ),
    )

    for item in (
        event_indexes
    ):
        print()
        print(
            item[
                "path"
            ]
        )

        if (
            "parse_error"
            in item
        ):
            print(
                "  parse error:",
                item[
                    "parse_error"
                ],
            )

            continue

        print(
            "  rows:",
            item[
                "row_count"
            ],
        )

        print(
            "  columns:",
            item[
                "columns"
            ],
        )

        print(
            "  timing columns:",
            item[
                "timing_columns"
            ],
        )

        print(
            "  first rows:",
            item[
                "first_rows"
            ],
        )

    uni_count = (
        univr[
            "complete_onset_impact_row_count"
        ]
    )

    if uni_count == 573:
        uni_discrepancy = {
            "local_complete_event_rows":
                573,

            "public_summary_events":
                573,

            "difference":
                0,

            "status":
                "RESOLVED_BY_SELECTED_ANNOTATION_ROOT",
        }

    elif uni_count == 574:
        uni_discrepancy = {
            "local_complete_event_rows":
                574,

            "public_summary_events":
                573,

            "difference":
                1,

            "status":
                "OPEN_ONE_EVENT_DIFFERENCE",
        }

    else:
        uni_discrepancy = {
            "local_complete_event_rows":
                uni_count,

            "public_summary_events":
                573,

            "difference":
                uni_count
                - 573,

            "status":
                "OPEN_UNEXPECTED_EVENT_COUNT",
        }

    # This phase establishes trial-key mapping, not yet timing-unit semantics.
    # Even perfectly mapped frame numbers must not automatically be treated
    # as 100-Hz sensor sample indices.
    timing_semantics = {
        "window_duration_ms":
            300,

        "window_samples":
            30,

        "stride_ms":
            150,

        "stride_samples":
            15,

        "candidate_causal_decision_timestamp":
            (
                "timestamp of final sensor sample available "
                "to the 300-ms window"
            ),

        "candidate_lead_time_equation":
            (
                "impact_time - decision_available_time"
            ),

        "annotation_frame_unit_status":
            "NOT_YET_FROZEN",

        "direct_frame_divide_by_100_allowed":
            False,

        "reason":
            (
                "Onset/impact annotation values must first be shown to be "
                "sensor-frame/sample aligned or transformed through a "
                "qualified synchronization mapping."
            ),
    }

    manifest = {
        "schema":
            "crosslayer_phase3k_event_annotation_trial_mapping_v1",

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
            "NO_MODEL_EXECUTION",

        "processed_primary_trials":
            processed,

        "annotation_selection": {
            "univr":
                univr_selection,

            "kfall":
                kfall_selection,
        },

        "annotation_audit": {
            "univr": {
                key:
                    value
                for key, value
                in univr.items()
                if key
                not in {
                    "records",
                    "complete_event_rows",
                }
            },

            "kfall": {
                key:
                    value
                for key, value
                in kfall.items()
                if key
                not in {
                    "records",
                    "complete_event_rows",
                }
            },
        },

        "reconciliation": {
            key:
                value
            for key, value
            in reconciliation.items()
            if key
            != "mapped_events"
        },

        "mapped_events":
            reconciliation[
                "mapped_events"
            ],

        "univr_573_574_discrepancy":
            uni_discrepancy,

        "curated_event_indexes":
            event_indexes,

        "timing_semantics_candidate":
            timing_semantics,

        "scientific_boundary": {
            "fivefold_membership_changed":
                False,

            "model_trained":
                False,

            "model_predictions_executed":
                False,

            "outer_test_outcomes_opened":
                False,

            "onfield_outcomes_opened":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,

            "physical_lead_time_frozen":
                False,
        },

        "next_questions": [
            (
                "Determine annotation frame-to-sensor-sample synchronization "
                "for UniVRFall."
            ),
            (
                "Determine annotation frame-to-sensor-sample synchronization "
                "for KFall."
            ),
            (
                "Resolve any duplicate/unmapped event trial keys."
            ),
            (
                "Resolve UniVR 573-versus-574 event-count difference."
            ),
            (
                "Freeze decision timestamp and physical impact lead-time "
                "equation only after synchronization is qualified."
            ),
        ],
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
        "UNIVR_573_574_STATUS=",
        uni_discrepancy[
            "status"
        ],
        sep="",
    )

    print(
        "PHASE_3K_EVENT_TRIAL_MAPPING_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
