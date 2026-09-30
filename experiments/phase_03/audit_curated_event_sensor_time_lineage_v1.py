from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PRIMARY = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

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

UNIVR_SENSOR_ROOT = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/sensors_data"
)

KFALL_SENSOR_ROOT = Path(
    "/mnt/hdd16T/protechto/"
    "ThirdPartyDatasets/KFall_oriented/sensors_data"
)

UNIVR_LABEL_ROOT = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/labels_data"
)

KFALL_LABEL_ROOT = Path(
    "/mnt/hdd16T/protechto/"
    "ThirdPartyDatasets/KFall/labels_data"
)

SOURCE_CODE_ROOTS = [
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "HR_LR_Fallings"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori"
    ),
]

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3l_curated_event_sensor_time_lineage_v1.json"
)

EVENT_RE = re.compile(
    r"^(?P<dataset>KFALL|UNIVR)_"
    r"(?P<subject>\d+)_"
    r"T(?P<task>\d+)_"
    r"R(?P<trial>\d+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def integer(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("missing integer value")

    numeric = float(value)
    rounded = int(round(numeric))

    if abs(numeric - rounded) > 1e-6:
        raise ValueError(
            f"non-integral value: {value!r}"
        )

    return rounded


def canonical_key(
    dataset: str,
    subject: int,
    task: int,
    trial: int,
) -> tuple[str, int, int, int]:

    return (
        str(dataset).upper(),
        int(subject),
        int(task),
        int(trial),
    )


def processed_inventory() -> dict[str, Any]:
    if not PRIMARY.is_dir():
        raise RuntimeError(
            f"Missing primary root: {PRIMARY}"
        )

    all_trials = {}
    fall_trials = {}

    counts = Counter()
    fall_counts = Counter()

    for label_path in sorted(
        PRIMARY.rglob("labels.npy")
    ):
        trial_dir = label_path.parent
        relative = trial_dir.relative_to(
            PRIMARY
        )

        if len(relative.parts) != 3:
            raise RuntimeError(
                f"Unexpected trial path: {relative}"
            )

        storage_subject, task, trial = (
            relative.parts
        )

        subject = int(storage_subject)

        if subject < 100:
            dataset = "UNIVR"
        else:
            dataset = "KFALL"

        key = canonical_key(
            dataset,
            subject,
            int(task),
            int(trial),
        )

        labels = np.load(
            label_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        unique, n = np.unique(
            labels,
            return_counts=True,
        )

        label_counts = {
            str(label): int(count)
            for label, count
            in zip(
                unique.tolist(),
                n.tolist(),
            )
        }

        contains_falling = (
            label_counts.get(
                "Falling",
                0,
            )
            > 0
        )

        row = {
            "dataset":
                dataset,

            "storage_subject":
                subject,

            "task":
                int(task),

            "trial":
                int(trial),

            "processed_path":
                relative.as_posix(),

            "window_count":
                int(labels.shape[0]),

            "label_counts":
                label_counts,

            "contains_falling":
                contains_falling,
        }

        if key in all_trials:
            raise RuntimeError(
                f"Duplicate processed key: {key}"
            )

        all_trials[key] = row

        counts[dataset] += 1

        if contains_falling:
            fall_trials[key] = row
            fall_counts[dataset] += 1

    return {
        "all_trial_count":
            len(all_trials),

        "fall_trial_count":
            len(fall_trials),

        "dataset_trial_counts":
            dict(counts),

        "dataset_fall_trial_counts":
            dict(fall_counts),

        "all_trials":
            all_trials,

        "fall_trials":
            fall_trials,
    }


def choose_event_index() -> dict[str, Any]:
    existing = [
        path
        for path
        in EVENT_INDEX_CANDIDATES
        if path.is_file()
    ]

    if not existing:
        raise RuntimeError(
            "No curated event index found"
        )

    records = []

    for path in existing:
        records.append(
            {
                "path":
                    str(path.resolve()),

                "sha256":
                    sha256_file(path),

                "bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

    hashes = {
        item["sha256"]
        for item
        in records
    }

    if len(hashes) != 1:
        # If bytes differ, compare the core lineage columns
        required = [
            "event_id",
            "dataset_id",
            "subject_id",
            "source_subject_id",
            "task_id",
            "trial_id",
            "sensor_file",
            "annotation_file",
            "fall_start_frame",
            "impact_frame",
            "fall_start_position",
            "impact_position",
            "sampling_rate_hz",
        ]

        frames = []

        for path in existing:
            frame = pd.read_csv(
                path,
                usecols=required,
            ).sort_values(
                "event_id"
            ).reset_index(
                drop=True
            )

            frames.append(frame)

        first = frames[0]

        if not all(
            first.equals(other)
            for other
            in frames[1:]
        ):
            raise RuntimeError(
                "Curated event-index copies disagree "
                "on core lineage fields"
            )

        content_status = (
            "CORE_LINEAGE_IDENTICAL_BYTES_DIFFER"
        )

    else:
        content_status = (
            "BYTE_IDENTICAL"
        )

    return {
        "status":
            content_status,

        "copies":
            records,

        "selected_path":
            str(
                existing[0].resolve()
            ),
    }


def parse_event_index(
    path: Path,
) -> dict[str, Any]:

    frame = pd.read_csv(path)

    required = {
        "event_id",
        "dataset_id",
        "subject_id",
        "source_subject_id",
        "task_id",
        "trial_id",
        "sensor_file",
        "annotation_file",
        "fall_start_frame",
        "impact_frame",
        "fall_start_position",
        "impact_position",
        "sampling_rate_hz",
        "fall_duration_ms",
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"Event index missing columns: {missing}"
        )

    records = {}
    dataset_counts = Counter()

    malformed_ids = []
    subject_disagreements = []
    invalid_order = []
    frame_position_mismatch = []
    bad_sampling_rates = []
    duration_mismatch = []

    for row_number, row in frame.iterrows():
        event_id = str(
            row["event_id"]
        ).strip()

        match = EVENT_RE.match(
            event_id
        )

        if not match:
            malformed_ids.append(
                {
                    "row":
                        int(row_number) + 2,

                    "event_id":
                        event_id,
                }
            )
            continue

        dataset = match.group(
            "dataset"
        )

        event_subject = int(
            match.group(
                "subject"
            )
        )

        task = int(
            match.group(
                "task"
            )
        )

        trial = int(
            match.group(
                "trial"
            )
        )

        row_dataset = str(
            row[
                "dataset_id"
            ]
        ).upper()

        row_subject_text = str(
            row[
                "subject_id"
            ]
        )

        subject_match = re.search(
            r"(\d+)$",
            row_subject_text,
        )

        if not subject_match:
            subject_disagreements.append(
                {
                    "event_id":
                        event_id,

                    "subject_id":
                        row_subject_text,
                }
            )
            continue

        row_subject = int(
            subject_match.group(1)
        )

        if (
            row_dataset != dataset
            or row_subject != event_subject
            or integer(row["task_id"]) != task
            or integer(row["trial_id"]) != trial
        ):
            subject_disagreements.append(
                {
                    "event_id":
                        event_id,

                    "dataset_id":
                        row_dataset,

                    "subject_id":
                        row_subject,

                    "task_id":
                        integer(
                            row[
                                "task_id"
                            ]
                        ),

                    "trial_id":
                        integer(
                            row[
                                "trial_id"
                            ]
                        ),
                }
            )

        onset_frame = integer(
            row[
                "fall_start_frame"
            ]
        )

        impact_frame = integer(
            row[
                "impact_frame"
            ]
        )

        onset_position = integer(
            row[
                "fall_start_position"
            ]
        )

        impact_position = integer(
            row[
                "impact_position"
            ]
        )

        if onset_frame >= impact_frame:
            invalid_order.append(
                event_id
            )

        if (
            onset_position
            != onset_frame - 1
            or impact_position
            != impact_frame - 1
        ):
            frame_position_mismatch.append(
                {
                    "event_id":
                        event_id,

                    "onset_frame":
                        onset_frame,

                    "onset_position":
                        onset_position,

                    "impact_frame":
                        impact_frame,

                    "impact_position":
                        impact_position,
                }
            )

        fs = float(
            row[
                "sampling_rate_hz"
            ]
        )

        if (
            not math.isfinite(fs)
            or fs <= 0.0
        ):
            bad_sampling_rates.append(
                {
                    "event_id":
                        event_id,

                    "sampling_rate_hz":
                        fs,
                }
            )

        expected_duration_ms = (
            (
                impact_position
                - onset_position
            )
            / fs
            * 1000.0
        )

        recorded_duration_ms = float(
            row[
                "fall_duration_ms"
            ]
        )

        if (
            abs(
                expected_duration_ms
                - recorded_duration_ms
            )
            > 1e-6
        ):
            duration_mismatch.append(
                {
                    "event_id":
                        event_id,

                    "expected_ms":
                        expected_duration_ms,

                    "recorded_ms":
                        recorded_duration_ms,
                }
            )

        key = canonical_key(
            dataset,
            event_subject,
            task,
            trial,
        )

        if key in records:
            raise RuntimeError(
                f"Duplicate curated event key: {key}"
            )

        records[key] = {
            "event_id":
                event_id,

            "dataset":
                dataset,

            "storage_subject":
                event_subject,

            "source_subject_id":
                str(
                    row[
                        "source_subject_id"
                    ]
                ),

            "task":
                task,

            "trial":
                trial,

            "sensor_file_historical":
                str(
                    row[
                        "sensor_file"
                    ]
                ),

            "annotation_file_historical":
                str(
                    row[
                        "annotation_file"
                    ]
                ),

            "fall_start_frame":
                onset_frame,

            "impact_frame":
                impact_frame,

            "fall_start_position":
                onset_position,

            "impact_position":
                impact_position,

            "sampling_rate_hz":
                fs,

            "fall_duration_ms":
                recorded_duration_ms,
        }

        dataset_counts[
            dataset
        ] += 1

    return {
        "row_count":
            int(
                len(frame)
            ),

        "event_count":
            len(records),

        "dataset_counts":
            dict(
                dataset_counts
            ),

        "malformed_event_id_count":
            len(
                malformed_ids
            ),

        "malformed_event_ids":
            malformed_ids,

        "field_disagreement_count":
            len(
                subject_disagreements
            ),

        "field_disagreements":
            subject_disagreements,

        "invalid_onset_impact_count":
            len(
                invalid_order
            ),

        "invalid_onset_impact":
            invalid_order,

        "frame_position_mismatch_count":
            len(
                frame_position_mismatch
            ),

        "frame_position_mismatch":
            frame_position_mismatch,

        "bad_sampling_rate_count":
            len(
                bad_sampling_rates
            ),

        "bad_sampling_rates":
            bad_sampling_rates,

        "duration_mismatch_count":
            len(
                duration_mismatch
            ),

        "duration_mismatch":
            duration_mismatch,

        "records":
            records,
    }


def compare_events_to_processed(
    processed: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:

    all_keys = set(
        processed[
            "all_trials"
        ]
    )

    fall_keys = set(
        processed[
            "fall_trials"
        ]
    )

    event_keys = set(
        events[
            "records"
        ]
    )

    event_and_fall = (
        event_keys
        & fall_keys
    )

    event_not_fall = (
        event_keys
        - fall_keys
    )

    fall_not_event = (
        fall_keys
        - event_keys
    )

    event_missing_processed = {
        key
        for key
        in event_keys
        if key not in all_keys
    }

    event_processed_activity_only = {
        key
        for key
        in event_not_fall
        if key in all_keys
    }

    def serialize(
        keys,
    ):
        return [
            {
                "dataset":
                    key[0],

                "storage_subject":
                    key[1],

                "task":
                    key[2],

                "trial":
                    key[3],

                "event":
                    (
                        events[
                            "records"
                        ].get(
                            key
                        )
                    ),

                "processed":
                    (
                        processed[
                            "all_trials"
                        ].get(
                            key
                        )
                    ),
            }
            for key
            in sorted(
                keys
            )
        ]

    by_dataset = {}

    for dataset in (
        "UNIVR",
        "KFALL",
    ):
        d_events = {
            key
            for key
            in event_keys
            if key[0] == dataset
        }

        d_falls = {
            key
            for key
            in fall_keys
            if key[0] == dataset
        }

        by_dataset[
            dataset
        ] = {
            "event_count":
                len(
                    d_events
                ),

            "processed_fall_trial_count":
                len(
                    d_falls
                ),

            "intersection_count":
                len(
                    d_events
                    & d_falls
                ),

            "event_only_count":
                len(
                    d_events
                    - d_falls
                ),

            "processed_fall_only_count":
                len(
                    d_falls
                    - d_events
                ),
        }

    return {
        "event_count":
            len(
                event_keys
            ),

        "processed_fall_trial_count":
            len(
                fall_keys
            ),

        "exact_event_fall_intersection_count":
            len(
                event_and_fall
            ),

        "event_not_processed_fall_count":
            len(
                event_not_fall
            ),

        "processed_fall_without_event_count":
            len(
                fall_not_event
            ),

        "event_missing_processed_trial_count":
            len(
                event_missing_processed
            ),

        "event_on_processed_activity_only_trial_count":
            len(
                event_processed_activity_only
            ),

        "by_dataset":
            by_dataset,

        "event_not_processed_fall":
            serialize(
                event_not_fall
            ),

        "processed_fall_without_event":
            serialize(
                fall_not_event
            ),

        "event_missing_processed_trial":
            serialize(
                event_missing_processed
            ),

        "event_on_processed_activity_only_trial":
            serialize(
                event_processed_activity_only
            ),
    }


def index_sensor_files(
    root: Path,
) -> dict[str, list[Path]]:

    index = defaultdict(
        list
    )

    if not root.is_dir():
        return {}

    for path in root.rglob(
        "*.csv"
    ):
        index[
            path.name
        ].append(
            path
        )

    return dict(index)


def index_annotation_files(
    root: Path,
) -> dict[str, list[Path]]:

    index = defaultdict(
        list
    )

    if not root.is_dir():
        return {}

    for path in root.rglob(
        "*.xlsx"
    ):
        index[
            path.name
        ].append(
            path
        )

    return dict(index)


def choose_unique_file(
    index: dict[
        str,
        list[Path],
    ],
    basename: str,
) -> Path | None:

    values = index.get(
        basename,
        [],
    )

    if len(values) == 1:
        return values[0]

    return None


def inspect_csv_event_positions(
    path: Path,
    onset_position: int,
    impact_position: int,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        rows = list(
            reader
        )

    result = {
        "row_count":
            len(
                rows
            ),

        "columns":
            fieldnames,
    }

    if (
        onset_position < 0
        or impact_position < 0
        or onset_position >= len(rows)
        or impact_position >= len(rows)
    ):
        result[
            "positions_in_range"
        ] = False

        return result

    result[
        "positions_in_range"
    ] = True

    normalized = {
        re.sub(
            r"[^a-z0-9]+",
            "",
            name.lower(),
        ):
            name
        for name
        in fieldnames
    }

    frame_field = None

    for candidate in (
        "framecounter",
        "frame",
    ):
        if candidate in normalized:
            frame_field = (
                normalized[
                    candidate
                ]
            )
            break

    timestamp_field = None

    for key, name in (
        normalized.items()
    ):
        if (
            "timestamp"
            in key
            or key
            in {
                "time",
                "timems",
                "times",
            }
        ):
            timestamp_field = name
            break

    result[
        "frame_counter_field"
    ] = frame_field

    result[
        "timestamp_field"
    ] = timestamp_field

    for label, position in (
        (
            "onset",
            onset_position,
        ),
        (
            "impact",
            impact_position,
        ),
    ):
        row = rows[
            position
        ]

        result[
            f"{label}_frame_counter"
        ] = (
            row.get(
                frame_field
            )
            if frame_field
            else None
        )

        result[
            f"{label}_timestamp"
        ] = (
            row.get(
                timestamp_field
            )
            if timestamp_field
            else None
        )

    return result


def raw_lineage_audit(
    events: dict[str, Any],
) -> dict[str, Any]:

    sensor_indexes = {
        "UNIVR":
            index_sensor_files(
                UNIVR_SENSOR_ROOT
            ),

        "KFALL":
            index_sensor_files(
                KFALL_SENSOR_ROOT
            ),
    }

    annotation_indexes = {
        "UNIVR":
            index_annotation_files(
                UNIVR_LABEL_ROOT
            ),

        "KFALL":
            index_annotation_files(
                KFALL_LABEL_ROOT
            ),
    }

    sensor_missing = []
    sensor_ambiguous = []

    annotation_missing = []
    annotation_ambiguous = []

    position_out_of_range = []

    kfall_framecounter_missing = 0
    kfall_framecounter_match = 0
    kfall_framecounter_mismatch = []

    univr_framecounter_present = 0

    timestamp_present = Counter()

    local_records = []

    for key, event in (
        events[
            "records"
        ].items()
    ):
        dataset = event[
            "dataset"
        ]

        sensor_basename = Path(
            event[
                "sensor_file_historical"
            ]
        ).name

        annotation_basename = Path(
            event[
                "annotation_file_historical"
            ]
        ).name

        sensor_candidates = (
            sensor_indexes[
                dataset
            ].get(
                sensor_basename,
                [],
            )
        )

        annotation_candidates = (
            annotation_indexes[
                dataset
            ].get(
                annotation_basename,
                [],
            )
        )

        if len(
            sensor_candidates
        ) == 0:
            sensor_missing.append(
                event[
                    "event_id"
                ]
            )
            continue

        if len(
            sensor_candidates
        ) > 1:
            sensor_ambiguous.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "candidates":
                        [
                            str(p)
                            for p
                            in sensor_candidates
                        ],
                }
            )
            continue

        sensor_path = (
            sensor_candidates[
                0
            ]
        )

        if len(
            annotation_candidates
        ) == 0:
            annotation_missing.append(
                event[
                    "event_id"
                ]
            )

            annotation_path = None

        elif len(
            annotation_candidates
        ) > 1:
            annotation_ambiguous.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "candidates":
                        [
                            str(p)
                            for p
                            in annotation_candidates
                        ],
                }
            )

            annotation_path = None

        else:
            annotation_path = (
                annotation_candidates[
                    0
                ]
            )

        sensor_audit = (
            inspect_csv_event_positions(
                sensor_path,
                event[
                    "fall_start_position"
                ],
                event[
                    "impact_position"
                ],
            )
        )

        if not sensor_audit[
            "positions_in_range"
        ]:
            position_out_of_range.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "sensor_path":
                        str(
                            sensor_path
                        ),

                    "row_count":
                        sensor_audit[
                            "row_count"
                        ],

                    "onset_position":
                        event[
                            "fall_start_position"
                        ],

                    "impact_position":
                        event[
                            "impact_position"
                        ],
                }
            )

        frame_field = (
            sensor_audit.get(
                "frame_counter_field"
            )
        )

        if dataset == "KFALL":
            if frame_field is None:
                kfall_framecounter_missing += 1

            elif sensor_audit[
                "positions_in_range"
            ]:
                try:
                    onset_counter = int(
                        round(
                            float(
                                sensor_audit[
                                    "onset_frame_counter"
                                ]
                            )
                        )
                    )

                    impact_counter = int(
                        round(
                            float(
                                sensor_audit[
                                    "impact_frame_counter"
                                ]
                            )
                        )
                    )

                    if (
                        onset_counter
                        == event[
                            "fall_start_frame"
                        ]
                        and impact_counter
                        == event[
                            "impact_frame"
                        ]
                    ):
                        kfall_framecounter_match += 1

                    else:
                        kfall_framecounter_mismatch.append(
                            {
                                "event_id":
                                    event[
                                        "event_id"
                                    ],

                                "annotation_onset":
                                    event[
                                        "fall_start_frame"
                                    ],

                                "csv_onset":
                                    onset_counter,

                                "annotation_impact":
                                    event[
                                        "impact_frame"
                                    ],

                                "csv_impact":
                                    impact_counter,
                            }
                        )

                except Exception:
                    kfall_framecounter_mismatch.append(
                        {
                            "event_id":
                                event[
                                    "event_id"
                                ],

                            "reason":
                                (
                                    "FRAMECOUNTER_PARSE_FAILURE"
                                ),
                        }
                    )

        elif frame_field is not None:
            univr_framecounter_present += 1

        if (
            sensor_audit.get(
                "timestamp_field"
            )
            is not None
        ):
            timestamp_present[
                dataset
            ] += 1

        local_records.append(
            {
                "event_id":
                    event[
                        "event_id"
                    ],

                "dataset":
                    dataset,

                "sensor_path":
                    str(
                        sensor_path
                    ),

                "annotation_path":
                    (
                        str(
                            annotation_path
                        )
                        if annotation_path
                        else None
                    ),

                "sensor_audit":
                    sensor_audit,
            }
        )

    return {
        "sensor_missing_count":
            len(
                sensor_missing
            ),

        "sensor_missing":
            sensor_missing,

        "sensor_ambiguous_count":
            len(
                sensor_ambiguous
            ),

        "sensor_ambiguous":
            sensor_ambiguous,

        "annotation_missing_count":
            len(
                annotation_missing
            ),

        "annotation_missing":
            annotation_missing,

        "annotation_ambiguous_count":
            len(
                annotation_ambiguous
            ),

        "annotation_ambiguous":
            annotation_ambiguous,

        "position_out_of_range_count":
            len(
                position_out_of_range
            ),

        "position_out_of_range":
            position_out_of_range,

        "kfall_framecounter_missing_count":
            kfall_framecounter_missing,

        "kfall_framecounter_exact_match_count":
            kfall_framecounter_match,

        "kfall_framecounter_mismatch_count":
            len(
                kfall_framecounter_mismatch
            ),

        "kfall_framecounter_mismatch":
            kfall_framecounter_mismatch,

        "univr_oriented_framecounter_present_event_count":
            univr_framecounter_present,

        "timestamp_field_present_event_counts":
            dict(
                timestamp_present
            ),

        "local_event_records":
            local_records,
    }


def source_builder_evidence() -> list[
    dict[str, Any]
]:

    needles = (
        "FALL_EVENT_INDEX_COMBINED_LABELED",
        "fall_start_position",
        "impact_position",
        "decision_end_frame_position",
    )

    results = []

    seen = set()

    for root in (
        SOURCE_CODE_ROOTS
    ):
        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(
            root
        ):
            dirs[:] = [
                value
                for value
                in dirs
                if value
                not in {
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    "data",
                    "datasets",
                    "checkpoints",
                    "artifacts",
                    "outputs",
                    "results",
                }
            ]

            for filename in files:
                path = (
                    Path(current)
                    / filename
                )

                if path.suffix.lower() not in {
                    ".py",
                    ".md",
                    ".txt",
                }:
                    continue

                try:
                    if (
                        path.stat().st_size
                        > 2_000_000
                    ):
                        continue

                    text = path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )

                except Exception:
                    continue

                if not any(
                    needle
                    in text
                    for needle
                    in needles
                ):
                    continue

                resolved = str(
                    path.resolve()
                )

                if resolved in seen:
                    continue

                seen.add(resolved)

                matches = []

                lines = text.splitlines()

                for number, line in enumerate(
                    lines,
                    start=1,
                ):
                    if not any(
                        needle
                        in line
                        for needle
                        in needles
                    ):
                        continue

                    start = max(
                        1,
                        number - 6,
                    )

                    end = min(
                        len(lines),
                        number + 10,
                    )

                    matches.append(
                        {
                            "match_line":
                                number,

                            "context":
                                "\n".join(
                                    (
                                        f"{i}: "
                                        f"{lines[i - 1]}"
                                    )
                                    for i
                                    in range(
                                        start,
                                        end + 1,
                                    )
                                ),
                        }
                    )

                results.append(
                    {
                        "path":
                            resolved,

                        "sha256":
                            sha256_file(
                                path
                            ),

                        "matches":
                            matches[:20],
                    }
                )

    return results


def main() -> None:
    processed = (
        processed_inventory()
    )

    print(
        "Processed all trials:",
        processed[
            "all_trial_count"
        ],
    )

    print(
        "Processed fall-positive trials:",
        processed[
            "fall_trial_count"
        ],
    )

    print(
        "Processed fall trials by dataset:",
        processed[
            "dataset_fall_trial_counts"
        ],
    )

    if (
        processed[
            "all_trial_count"
        ]
        != 6309
    ):
        raise RuntimeError(
            "Primary trial count changed"
        )

    if (
        processed[
            "fall_trial_count"
        ]
        != 2918
    ):
        raise RuntimeError(
            "Expected 2918 processed fall-positive trials"
        )

    index_choice = (
        choose_event_index()
    )

    selected = Path(
        index_choice[
            "selected_path"
        ]
    )

    print()
    print(
        "Event-index copy status:",
        index_choice[
            "status"
        ],
    )

    for item in (
        index_choice[
            "copies"
        ]
    ):
        print(
            " ",
            item[
                "path"
            ],
            item[
                "sha256"
            ],
        )

    events = (
        parse_event_index(
            selected
        )
    )

    print()
    print(
        "Curated event rows:",
        events[
            "row_count"
        ],
    )

    print(
        "Unique events:",
        events[
            "event_count"
        ],
    )

    print(
        "Events by dataset:",
        events[
            "dataset_counts"
        ],
    )

    print(
        "Malformed event IDs:",
        events[
            "malformed_event_id_count"
        ],
    )

    print(
        "Field disagreements:",
        events[
            "field_disagreement_count"
        ],
    )

    print(
        "Invalid onset >= impact:",
        events[
            "invalid_onset_impact_count"
        ],
    )

    print(
        "Frame/position mismatches:",
        events[
            "frame_position_mismatch_count"
        ],
    )

    print(
        "Bad sampling rates:",
        events[
            "bad_sampling_rate_count"
        ],
    )

    print(
        "Duration mismatches:",
        events[
            "duration_mismatch_count"
        ],
    )

    comparison = (
        compare_events_to_processed(
            processed,
            events,
        )
    )

    print()
    print(
        "CURATED EVENT / PROCESSED FALL COMPARISON"
    )

    print(
        "Curated events:",
        comparison[
            "event_count"
        ],
    )

    print(
        "Processed fall trials:",
        comparison[
            "processed_fall_trial_count"
        ],
    )

    print(
        "Exact intersection:",
        comparison[
            "exact_event_fall_intersection_count"
        ],
    )

    print(
        "Event not processed-fall:",
        comparison[
            "event_not_processed_fall_count"
        ],
    )

    print(
        "Processed-fall without event:",
        comparison[
            "processed_fall_without_event_count"
        ],
    )

    print(
        "Event missing processed trial:",
        comparison[
            "event_missing_processed_trial_count"
        ],
    )

    print(
        "Event on processed Activity-only trial:",
        comparison[
            "event_on_processed_activity_only_trial_count"
        ],
    )

    print(
        "By dataset:",
        comparison[
            "by_dataset"
        ],
    )

    if comparison[
        "event_not_processed_fall_count"
    ]:
        print()
        print(
            "EVENTS NOT REPRESENTED AS PROCESSED FALL TRIAL"
        )

        for item in comparison[
            "event_not_processed_fall"
        ][:50]:
            print(
                item
            )

    if comparison[
        "processed_fall_without_event_count"
    ]:
        print()
        print(
            "PROCESSED FALL TRIALS WITHOUT CURATED EVENT"
        )

        for item in comparison[
            "processed_fall_without_event"
        ][:50]:
            print(
                item
            )

    raw = (
        raw_lineage_audit(
            events
        )
    )

    print()
    print(
        "RAW SENSOR LINEAGE"
    )

    print(
        "Sensor files missing:",
        raw[
            "sensor_missing_count"
        ],
    )

    print(
        "Sensor files ambiguous:",
        raw[
            "sensor_ambiguous_count"
        ],
    )

    print(
        "Annotation files missing:",
        raw[
            "annotation_missing_count"
        ],
    )

    print(
        "Annotation files ambiguous:",
        raw[
            "annotation_ambiguous_count"
        ],
    )

    print(
        "Event positions out of range:",
        raw[
            "position_out_of_range_count"
        ],
    )

    print(
        "KFall events missing FrameCounter:",
        raw[
            "kfall_framecounter_missing_count"
        ],
    )

    print(
        "KFall FrameCounter exact matches:",
        raw[
            "kfall_framecounter_exact_match_count"
        ],
    )

    print(
        "KFall FrameCounter mismatches:",
        raw[
            "kfall_framecounter_mismatch_count"
        ],
    )

    print(
        "UniVR oriented events with FrameCounter field:",
        raw[
            "univr_oriented_framecounter_present_event_count"
        ],
    )

    print(
        "Timestamp fields by dataset:",
        raw[
            "timestamp_field_present_event_counts"
        ],
    )

    rates = [
        float(
            item[
                "sampling_rate_hz"
            ]
        )
        for item
        in events[
            "records"
        ].values()
    ]

    sampling_summary = {
        "count":
            len(rates),

        "min_hz":
            min(rates),

        "median_hz":
            median(rates),

        "max_hz":
            max(rates),

        "all_within_0_01_hz_of_100":
            all(
                abs(rate - 100.0)
                <= 0.01
                for rate
                in rates
            ),
    }

    print()
    print(
        "Sampling-rate summary:",
        sampling_summary,
    )

    builders = (
        source_builder_evidence()
    )

    print()
    print(
        "Event-index builder/provenance source files:",
        len(
            builders
        ),
    )

    for item in builders[
        :20
    ]:
        print()
        print(
            item[
                "path"
            ]
        )

        for match in (
            item[
                "matches"
            ][:4]
        ):
            print(
                match[
                    "context"
                ]
            )

    # Dataset-specific timing qualification.
    kfall_event_count = (
        events[
            "dataset_counts"
        ].get(
            "KFALL",
            0,
        )
    )

    kfall_hard_sequence_qualified = (
        raw[
            "sensor_missing_count"
        ]
        == 0
        and raw[
            "position_out_of_range_count"
        ]
        == 0
        and raw[
            "kfall_framecounter_missing_count"
        ]
        == 0
        and raw[
            "kfall_framecounter_mismatch_count"
        ]
        == 0
        and raw[
            "kfall_framecounter_exact_match_count"
        ]
        == kfall_event_count
    )

    univr_timing_status = (
        "PENDING_ORIGINAL_TIMESTAMP_SYNCHRONIZATION_AUDIT"
    )

    manifest = {
        "schema":
            "crosslayer_phase3l_curated_event_sensor_time_lineage_v1",

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

        "phase3k_interpretation": {
            "workbook_audit_valid":
                True,

            "naive_direct_workbook_trial_join_authoritative":
                False,

            "reason":
                (
                    "Workbook rows are annotation records and can encode "
                    "multiple trial instances. The curated expanded "
                    "event index is audited separately in Phase 3L."
                ),
        },

        "event_index_choice":
            index_choice,

        "curated_event_index": {
            key:
                value
            for key, value
            in events.items()
            if key != "records"
        },

        "processed_inventory": {
            key:
                value
            for key, value
            in processed.items()
            if key
            not in {
                "all_trials",
                "fall_trials",
            }
        },

        "event_processed_comparison":
            comparison,

        "raw_lineage": {
            key:
                value
            for key, value
            in raw.items()
            if key
            != "local_event_records"
        },

        "sampling_rate_summary":
            sampling_summary,

        "source_builder_evidence":
            builders,

        "timing_qualification": {
            "kfall_frame_to_sensor_sample":
                (
                    "QUALIFIED"
                    if kfall_hard_sequence_qualified
                    else "NOT_YET_QUALIFIED"
                ),

            "kfall_hard_sequence_qualified":
                kfall_hard_sequence_qualified,

            "univr_frame_to_sensor_sample":
                univr_timing_status,

            "physical_lead_time_fully_frozen":
                False,
        },

        "scientific_boundary": {
            "fivefold_membership_changed":
                False,

            "model_trained":
                False,

            "predictions_executed":
                False,

            "outer_test_outcomes_opened":
                False,

            "onfield_outcomes_opened":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,

            "physical_lead_time_fully_frozen":
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
        "KFALL_SENSOR_TIME_STATUS=",
        manifest[
            "timing_qualification"
        ][
            "kfall_frame_to_sensor_sample"
        ],
        sep="",
    )

    print(
        "UNIVR_SENSOR_TIME_STATUS=",
        manifest[
            "timing_qualification"
        ][
            "univr_frame_to_sensor_sample"
        ],
        sep="",
    )

    print(
        "PHASE_3L_CURATED_EVENT_SENSOR_TIME_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
