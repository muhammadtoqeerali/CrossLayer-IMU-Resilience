from __future__ import annotations

import csv
import hashlib
import io
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

EVENT_INDEX = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "HR_LR_Fallings/risk_data_generated/"
    "FALL_EVENT_INDEX_COMBINED_LABELED.csv"
)

UNIVR_ORIENTED = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/sensors_data"
)

UNIVR_ORIGINAL_CANDIDATES = [
    Path(
        "/mnt/hdd16T/protechto/"
        "UniVrFallOriginalDataset"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "uniVr-dataset/UniVrFall_Dataset"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "uniVr-dataset/UniVR_Protechto_dataset"
    ),
]

RISK_REPO = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "HR_LR_Fallings"
)

OUT = (
    ROOT
    / "manifests/"
      "phase_3m_canonical_event_binding_univr_time_v1.json"
)

WINDOW_MS = 300
DEADLINE_MS = 150
OVERLAP = 0.50


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def as_int(value: Any) -> int:
    if pd.isna(value):
        raise ValueError(
            "missing integer value"
        )

    value = float(value)
    rounded = int(round(value))

    if abs(value - rounded) > 1e-6:
        raise ValueError(
            f"non-integral value {value}"
        )

    return rounded


def source_subject_number(
    value: Any,
) -> int:
    text = str(value).strip()

    match = re.search(
        r"(\d+)$",
        text,
    )

    if not match:
        raise ValueError(
            f"Cannot parse source subject: {value!r}"
        )

    return int(
        match.group(1)
    )


def canonical_processed_subject(
    dataset: str,
    source_subject: int,
) -> int:
    dataset = dataset.upper()

    if dataset == "UNIVR":
        return int(
            source_subject
        )

    if dataset == "KFALL":
        return int(
            source_subject
        ) + 100

    raise ValueError(
        f"Unknown dataset {dataset}"
    )


def expected_event_namespace_subject(
    dataset: str,
    source_subject: int,
) -> int:
    dataset = dataset.upper()

    if dataset == "UNIVR":
        return int(
            source_subject
        ) + 1000

    if dataset == "KFALL":
        return int(
            source_subject
        ) + 100

    raise ValueError(
        f"Unknown dataset {dataset}"
    )


def key(
    dataset: str,
    processed_subject: int,
    task: int,
    trial: int,
) -> tuple[str, int, int, int]:

    return (
        str(dataset).upper(),
        int(processed_subject),
        int(task),
        int(trial),
    )


def processed_inventory():
    all_trials = {}
    fall_trials = {}

    counts = Counter()
    fall_counts = Counter()

    for label_path in sorted(
        PRIMARY.rglob(
            "labels.npy"
        )
    ):
        rel = (
            label_path.parent
            .relative_to(
                PRIMARY
            )
        )

        if len(rel.parts) != 3:
            raise RuntimeError(
                f"Unexpected processed path {rel}"
            )

        subject = int(
            rel.parts[0]
        )

        task = int(
            rel.parts[1]
        )

        trial = int(
            rel.parts[2]
        )

        dataset = (
            "UNIVR"
            if subject < 100
            else "KFALL"
        )

        labels = np.load(
            label_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        values, n = np.unique(
            labels,
            return_counts=True,
        )

        label_counts = {
            str(v):
                int(c)
            for v, c
            in zip(
                values.tolist(),
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

        k = key(
            dataset,
            subject,
            task,
            trial,
        )

        row = {
            "dataset":
                dataset,

            "processed_subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "processed_path":
                rel.as_posix(),

            "window_count":
                int(
                    labels.shape[0]
                ),

            "label_counts":
                label_counts,

            "contains_falling":
                contains_falling,
        }

        if k in all_trials:
            raise RuntimeError(
                f"Duplicate processed trial {k}"
            )

        all_trials[k] = row
        counts[dataset] += 1

        if contains_falling:
            fall_trials[k] = row
            fall_counts[dataset] += 1

    return {
        "all_trials":
            all_trials,

        "fall_trials":
            fall_trials,

        "all_count":
            len(all_trials),

        "fall_count":
            len(fall_trials),

        "dataset_counts":
            dict(counts),

        "dataset_fall_counts":
            dict(fall_counts),
    }


def load_events():
    if not EVENT_INDEX.is_file():
        raise RuntimeError(
            f"Missing event index: {EVENT_INDEX}"
        )

    frame = pd.read_csv(
        EVENT_INDEX
    )

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
            f"Missing event-index columns: {missing}"
        )

    records = {}
    namespace_failures = []
    position_conventions = Counter()
    dataset_counts = Counter()

    for row_index, row in frame.iterrows():
        dataset = str(
            row["dataset_id"]
        ).upper()

        source_subject = (
            source_subject_number(
                row[
                    "source_subject_id"
                ]
            )
        )

        processed_subject = (
            canonical_processed_subject(
                dataset,
                source_subject,
            )
        )

        namespace_subject = (
            expected_event_namespace_subject(
                dataset,
                source_subject,
            )
        )

        subject_text = str(
            row["subject_id"]
        )

        subject_match = re.search(
            r"(\d+)$",
            subject_text,
        )

        if not subject_match:
            namespace_failures.append(
                {
                    "row":
                        int(row_index)
                        + 2,

                    "subject_id":
                        subject_text,

                    "reason":
                        "UNPARSEABLE_SUBJECT_ID",
                }
            )
            continue

        actual_namespace_subject = int(
            subject_match.group(1)
        )

        if (
            actual_namespace_subject
            != namespace_subject
        ):
            namespace_failures.append(
                {
                    "event_id":
                        str(
                            row[
                                "event_id"
                            ]
                        ),

                    "dataset":
                        dataset,

                    "source_subject":
                        source_subject,

                    "actual_namespace_subject":
                        actual_namespace_subject,

                    "expected_namespace_subject":
                        namespace_subject,
                }
            )

        task = as_int(
            row["task_id"]
        )

        trial = as_int(
            row["trial_id"]
        )

        onset_frame = as_int(
            row[
                "fall_start_frame"
            ]
        )

        impact_frame = as_int(
            row[
                "impact_frame"
            ]
        )

        onset_pos = as_int(
            row[
                "fall_start_position"
            ]
        )

        impact_pos = as_int(
            row[
                "impact_position"
            ]
        )

        onset_offset = (
            onset_pos
            - onset_frame
        )

        impact_offset = (
            impact_pos
            - impact_frame
        )

        position_conventions[
            (
                dataset,
                onset_offset,
                impact_offset,
            )
        ] += 1

        fs = float(
            row[
                "sampling_rate_hz"
            ]
        )

        window_samples = int(
            round(
                WINDOW_MS
                * fs
                / 1000.0
            )
        )

        deadline_samples = int(
            round(
                DEADLINE_MS
                * fs
                / 1000.0
            )
        )

        available = (
            impact_pos
            - onset_pos
            - deadline_samples
        )

        eligible = (
            available
            >= window_samples
        )

        k = key(
            dataset,
            processed_subject,
            task,
            trial,
        )

        if k in records:
            raise RuntimeError(
                f"Duplicate canonical event key {k}"
            )

        records[k] = {
            "event_id":
                str(
                    row[
                        "event_id"
                    ]
                ),

            "dataset":
                dataset,

            "event_namespace_subject":
                actual_namespace_subject,

            "source_subject":
                source_subject,

            "processed_subject":
                processed_subject,

            "task":
                task,

            "trial":
                trial,

            "sensor_file":
                str(
                    row[
                        "sensor_file"
                    ]
                ),

            "annotation_file":
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
                onset_pos,

            "impact_position":
                impact_pos,

            "sampling_rate_hz":
                fs,

            "fall_duration_ms":
                float(
                    row[
                        "fall_duration_ms"
                    ]
                ),

            "window_samples":
                window_samples,

            "deadline_samples":
                deadline_samples,

            "post_onset_predeadline_available_samples":
                available,

            "post_onset_predeadline_available_ms":
                (
                    available
                    * 1000.0
                    / fs
                ),

            "eligible_for_complete_post_onset_predeadline_window":
                eligible,
        }

        dataset_counts[
            dataset
        ] += 1

    return {
        "records":
            records,

        "event_count":
            len(records),

        "dataset_counts":
            dict(
                dataset_counts
            ),

        "namespace_failure_count":
            len(
                namespace_failures
            ),

        "namespace_failures":
            namespace_failures,

        "position_conventions":
            [
                {
                    "dataset":
                        dataset,

                    "position_minus_frame_onset":
                        onset_offset,

                    "position_minus_frame_impact":
                        impact_offset,

                    "count":
                        count,
                }
                for (
                    dataset,
                    onset_offset,
                    impact_offset,
                ), count
                in sorted(
                    position_conventions.items()
                )
            ],
    }


def reconcile(
    processed,
    events,
):
    all_processed = set(
        processed[
            "all_trials"
        ]
    )

    processed_fall = set(
        processed[
            "fall_trials"
        ]
    )

    event_keys = set(
        events[
            "records"
        ]
    )

    eligible = {
        k
        for k, row
        in events[
            "records"
        ].items()
        if row[
            "eligible_for_complete_post_onset_predeadline_window"
        ]
    }

    ineligible = (
        event_keys
        - eligible
    )

    missing_processed = (
        event_keys
        - all_processed
    )

    eligible_missing_fall = (
        eligible
        - processed_fall
    )

    fall_without_eligible = (
        processed_fall
        - eligible
    )

    ineligible_but_fall_positive = (
        ineligible
        & processed_fall
    )

    def details(keys):
        return [
            {
                "key":
                    list(k),

                "event":
                    events[
                        "records"
                    ].get(
                        k
                    ),

                "processed":
                    processed[
                        "all_trials"
                    ].get(
                        k
                    ),
            }
            for k
            in sorted(keys)
        ]

    by_dataset = {}

    for dataset in (
        "UNIVR",
        "KFALL",
    ):
        d_events = {
            k
            for k
            in event_keys
            if k[0] == dataset
        }

        d_eligible = {
            k
            for k
            in eligible
            if k[0] == dataset
        }

        d_fall = {
            k
            for k
            in processed_fall
            if k[0] == dataset
        }

        by_dataset[
            dataset
        ] = {
            "annotated_events":
                len(
                    d_events
                ),

            "protocol_eligible_events":
                len(
                    d_eligible
                ),

            "processed_fall_positive_trials":
                len(
                    d_fall
                ),

            "eligible_intersection":
                len(
                    d_eligible
                    & d_fall
                ),

            "eligible_missing_processed_fall":
                len(
                    d_eligible
                    - d_fall
                ),

            "processed_fall_without_eligible_event":
                len(
                    d_fall
                    - d_eligible
                ),
        }

    return {
        "annotated_event_count":
            len(
                event_keys
            ),

        "protocol_eligible_event_count":
            len(
                eligible
            ),

        "protocol_ineligible_event_count":
            len(
                ineligible
            ),

        "processed_fall_positive_trial_count":
            len(
                processed_fall
            ),

        "event_missing_processed_trial_count":
            len(
                missing_processed
            ),

        "eligible_event_processed_fall_intersection":
            len(
                eligible
                & processed_fall
            ),

        "eligible_event_missing_processed_fall_count":
            len(
                eligible_missing_fall
            ),

        "processed_fall_without_eligible_event_count":
            len(
                fall_without_eligible
            ),

        "ineligible_event_processed_as_fall_positive_count":
            len(
                ineligible_but_fall_positive
            ),

        "ineligible_events":
            details(
                ineligible
            ),

        "event_missing_processed_trials":
            details(
                missing_processed
            ),

        "eligible_event_missing_processed_fall":
            details(
                eligible_missing_fall
            ),

        "processed_fall_without_eligible_event":
            details(
                fall_without_eligible
            ),

        "by_dataset":
            by_dataset,
    }


def build_name_index(
    root: Path,
):
    index = defaultdict(list)

    if not root.is_dir():
        return {}

    for path in root.rglob(
        "*.csv"
    ):
        index[
            path.name.lower()
        ].append(
            path
        )

    return dict(index)


def detect_delimited_table(
    path: Path,
) -> dict[str, Any]:
    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    lines = text.splitlines()

    candidates = [
        ",",
        ";",
        "\t",
    ]

    best = None

    expected_tokens = {
        "timestamp",
        "timestamps",
        "timestampms",
        "framecounter",
        "accx",
        "accy",
        "accz",
        "gyrx",
        "gyry",
        "gyrz",
        "eulerx",
        "eulery",
        "eulerz",
    }

    for line_index, line in enumerate(
        lines[:80]
    ):
        if not line.strip():
            continue

        for delimiter in candidates:
            try:
                fields = next(
                    csv.reader(
                        [line],
                        delimiter=delimiter,
                    )
                )
            except Exception:
                continue

            normalized = [
                re.sub(
                    r"[^a-z0-9]+",
                    "",
                    str(value).lower(),
                )
                for value
                in fields
            ]

            score = sum(
                1
                for value
                in normalized
                if value
                in expected_tokens
                or "timestamp"
                in value
                or "framecounter"
                in value
            )

            if (
                best is None
                or score
                > best[
                    "score"
                ]
            ):
                best = {
                    "header_line_zero_based":
                        line_index,

                    "delimiter":
                        delimiter,

                    "score":
                        score,

                    "fields":
                        fields,
                }

    if (
        best is None
        or best[
            "score"
        ] < 2
    ):
        return {
            "status":
                "HEADER_NOT_DETECTED",

            "first_lines":
                lines[:12],
        }

    try:
        frame = pd.read_csv(
            path,
            skiprows=best[
                "header_line_zero_based"
            ],
            sep=best[
                "delimiter"
            ],
            engine="python",
        )

    except Exception as exc:
        return {
            "status":
                "PARSE_FAILED",

            "header":
                best,

            "error":
                repr(exc),
        }

    return {
        "status":
            "PARSED",

        "header":
            best,

        "frame":
            frame,
    }


def choose_column(
    columns,
    terms,
):
    normalized = {
        re.sub(
            r"[^a-z0-9]+",
            "",
            str(column).lower(),
        ):
            column
        for column
        in columns
    }

    for exact in terms:
        if exact in normalized:
            return normalized[
                exact
            ]

    for norm, column in (
        normalized.items()
    ):
        if any(
            term in norm
            for term
            in terms
        ):
            return column

    return None


def numeric_series(
    series,
):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


def infer_timestamp_seconds(
    values,
):
    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size < 3:
        return None

    diffs = np.diff(
        finite
    )

    positive = diffs[
        diffs > 0
    ]

    if positive.size == 0:
        return None

    med = float(
        np.median(
            positive
        )
    )

    # Typical UniVR rates are ~0.01 s or ~10 ms.
    if med > 1.0:
        scale = 0.001
        unit = "milliseconds"
    else:
        scale = 1.0
        unit = "seconds"

    return {
        "seconds":
            values
            * scale,

        "detected_unit":
            unit,

        "median_positive_delta_seconds":
            med
            * scale,
    }


def audit_univr_original(
    events,
):
    roots = [
        path
        for path
        in UNIVR_ORIGINAL_CANDIDATES
        if path.is_dir()
    ]

    root_summaries = []

    root_indexes = {}

    for root in roots:
        index = build_name_index(
            root
        )

        root_indexes[
            str(root)
        ] = index

        root_summaries.append(
            {
                "root":
                    str(
                        root.resolve()
                    ),

                "csv_count":
                    sum(
                        len(v)
                        for v
                        in index.values()
                    ),

                "unique_basenames":
                    len(index),
            }
        )

    uni_events = [
        row
        for row
        in events[
            "records"
        ].values()
        if row[
            "dataset"
        ]
        == "UNIVR"
    ]

    resolved = []
    missing = []
    ambiguous = []
    parse_failures = []
    out_of_range = []
    timestamp_missing = []
    timestamp_unusable = []
    framecounter_missing = []

    timestamp_duration_errors = []
    timestamp_delta_medians = []
    timestamp_units = Counter()
    framecounter_offsets = Counter()

    source_root_counts = Counter()

    examples = []

    for event in uni_events:
        basename = Path(
            event[
                "sensor_file"
            ]
        ).name.lower()

        candidates = []

        for root in roots:
            found = (
                root_indexes[
                    str(root)
                ].get(
                    basename,
                    [],
                )
            )

            for item in found:
                candidates.append(
                    item
                )

        # Deduplicate exact filesystem objects.
        unique = {}

        for path in candidates:
            try:
                real = str(
                    path.resolve()
                )
            except Exception:
                real = str(path)

            unique[
                real
            ] = path

        candidates = list(
            unique.values()
        )

        if len(candidates) == 0:
            missing.append(
                event[
                    "event_id"
                ]
            )
            continue

        if len(candidates) > 1:
            # Multiple dataset copies can legitimately contain the same
            # basename. Prefer the explicit protechto original root if it
            # has exactly one candidate.
            preferred = [
                p
                for p
                in candidates
                if str(p).startswith(
                    str(
                        UNIVR_ORIGINAL_CANDIDATES[
                            0
                        ]
                    )
                )
            ]

            if len(preferred) == 1:
                path = preferred[0]
            else:
                ambiguous.append(
                    {
                        "event_id":
                            event[
                                "event_id"
                            ],

                        "candidates":
                            [
                                str(p)
                                for p
                                in candidates
                            ],
                    }
                )
                continue
        else:
            path = candidates[0]

        source_root_counts[
            next(
                (
                    str(root)
                    for root
                    in roots
                    if str(path).startswith(
                        str(root)
                    )
                ),
                "UNKNOWN",
            )
        ] += 1

        parsed = detect_delimited_table(
            path
        )

        if parsed[
            "status"
        ] != "PARSED":
            parse_failures.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "path":
                        str(path),

                    "parse":
                        parsed,
                }
            )
            continue

        frame = parsed[
            "frame"
        ]

        onset = event[
            "fall_start_position"
        ]

        impact = event[
            "impact_position"
        ]

        if (
            onset < 0
            or impact < 0
            or onset >= len(frame)
            or impact >= len(frame)
        ):
            out_of_range.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "path":
                        str(path),

                    "rows":
                        int(
                            len(frame)
                        ),

                    "onset_position":
                        onset,

                    "impact_position":
                        impact,
                }
            )
            continue

        timestamp_col = choose_column(
            frame.columns,
            [
                "timestamps",
                "timestamp",
                "timestampms",
                "time",
            ],
        )

        framecounter_col = choose_column(
            frame.columns,
            [
                "framecounter",
            ],
        )

        timestamp_result = None

        if timestamp_col is None:
            timestamp_missing.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "path":
                        str(path),

                    "columns":
                        [
                            str(c)
                            for c
                            in frame.columns
                        ],
                }
            )

        else:
            values = numeric_series(
                frame[
                    timestamp_col
                ]
            )

            timestamp_result = (
                infer_timestamp_seconds(
                    values
                )
            )

            if timestamp_result is None:
                timestamp_unusable.append(
                    {
                        "event_id":
                            event[
                                "event_id"
                            ],

                        "path":
                            str(path),

                        "column":
                            str(
                                timestamp_col
                            ),
                    }
                )

            else:
                seconds = (
                    timestamp_result[
                        "seconds"
                    ]
                )

                timestamp_units[
                    timestamp_result[
                        "detected_unit"
                    ]
                ] += 1

                timestamp_delta_medians.append(
                    timestamp_result[
                        "median_positive_delta_seconds"
                    ]
                )

                if (
                    not np.isfinite(
                        seconds[
                            onset
                        ]
                    )
                    or not np.isfinite(
                        seconds[
                            impact
                        ]
                    )
                ):
                    timestamp_unusable.append(
                        {
                            "event_id":
                                event[
                                    "event_id"
                                ],

                            "reason":
                                "ONSET_OR_IMPACT_TIMESTAMP_NONFINITE",
                        }
                    )

                else:
                    observed_ms = (
                        seconds[
                            impact
                        ]
                        - seconds[
                            onset
                        ]
                    ) * 1000.0

                    expected_ms = (
                        event[
                            "fall_duration_ms"
                        ]
                    )

                    error_ms = (
                        observed_ms
                        - expected_ms
                    )

                    timestamp_duration_errors.append(
                        {
                            "event_id":
                                event[
                                    "event_id"
                                ],

                            "observed_ms":
                                float(
                                    observed_ms
                                ),

                            "expected_ms":
                                float(
                                    expected_ms
                                ),

                            "error_ms":
                                float(
                                    error_ms
                                ),
                        }
                    )

        if framecounter_col is None:
            framecounter_missing.append(
                event[
                    "event_id"
                ]
            )

        else:
            fc = numeric_series(
                frame[
                    framecounter_col
                ]
            )

            if (
                np.isfinite(
                    fc[
                        onset
                    ]
                )
                and np.isfinite(
                    fc[
                        impact
                    ]
                )
            ):
                onset_offset = int(
                    round(
                        fc[
                            onset
                        ]
                    )
                ) - event[
                    "fall_start_frame"
                ]

                impact_offset = int(
                    round(
                        fc[
                            impact
                        ]
                    )
                ) - event[
                    "impact_frame"
                ]

                framecounter_offsets[
                    (
                        onset_offset,
                        impact_offset,
                    )
                ] += 1

        resolved.append(
            {
                "event_id":
                    event[
                        "event_id"
                    ],

                "path":
                    str(
                        path.resolve()
                    ),

                "rows":
                    int(
                        len(frame)
                    ),

                "timestamp_column":
                    (
                        str(
                            timestamp_col
                        )
                        if timestamp_col
                        is not None
                        else None
                    ),

                "framecounter_column":
                    (
                        str(
                            framecounter_col
                        )
                        if framecounter_col
                        is not None
                        else None
                    ),

                "header":
                    parsed[
                        "header"
                    ],
            }
        )

        if len(examples) < 12:
            examples.append(
                {
                    "event":
                        event,

                    "path":
                        str(
                            path.resolve()
                        ),

                    "columns":
                        [
                            str(c)
                            for c
                            in frame.columns
                        ],

                    "header":
                        parsed[
                            "header"
                        ],
                }
            )

    errors = [
        abs(
            item[
                "error_ms"
            ]
        )
        for item
        in timestamp_duration_errors
        if math.isfinite(
            item[
                "error_ms"
            ]
        )
    ]

    if errors:
        error_summary = {
            "count":
                len(errors),

            "max_abs_error_ms":
                max(errors),

            "median_abs_error_ms":
                median(errors),

            "within_1ms_count":
                sum(
                    value <= 1.0
                    for value
                    in errors
                ),

            "within_11ms_count":
                sum(
                    value <= 11.0
                    for value
                    in errors
                ),
        }
    else:
        error_summary = {
            "count":
                0,

            "max_abs_error_ms":
                None,

            "median_abs_error_ms":
                None,

            "within_1ms_count":
                0,

            "within_11ms_count":
                0,
        }

    sample_time_qualified = (
        len(uni_events)
        == 573
        and len(missing) == 0
        and len(ambiguous) == 0
        and len(parse_failures) == 0
        and len(out_of_range) == 0
        and len(timestamp_missing) == 0
        and len(timestamp_unusable) == 0
        and len(
            timestamp_duration_errors
        )
        == 573
        and error_summary[
            "max_abs_error_ms"
        ]
        is not None
        and error_summary[
            "max_abs_error_ms"
        ]
        <= 11.0
    )

    return {
        "candidate_roots":
            root_summaries,

        "event_count":
            len(
                uni_events
            ),

        "resolved_count":
            len(
                resolved
            ),

        "missing_count":
            len(
                missing
            ),

        "missing":
            missing,

        "ambiguous_count":
            len(
                ambiguous
            ),

        "ambiguous":
            ambiguous,

        "parse_failure_count":
            len(
                parse_failures
            ),

        "parse_failures":
            parse_failures[:30],

        "position_out_of_range_count":
            len(
                out_of_range
            ),

        "position_out_of_range":
            out_of_range[:30],

        "timestamp_missing_count":
            len(
                timestamp_missing
            ),

        "timestamp_missing":
            timestamp_missing[:30],

        "timestamp_unusable_count":
            len(
                timestamp_unusable
            ),

        "timestamp_unusable":
            timestamp_unusable[:30],

        "framecounter_missing_count":
            len(
                framecounter_missing
            ),

        "framecounter_offsets":
            [
                {
                    "framecounter_minus_annotation_onset":
                        k[0],

                    "framecounter_minus_annotation_impact":
                        k[1],

                    "count":
                        v,
                }
                for k, v
                in sorted(
                    framecounter_offsets.items()
                )
            ],

        "timestamp_unit_counts":
            dict(
                timestamp_units
            ),

        "timestamp_median_delta_seconds_summary":
            {
                "count":
                    len(
                        timestamp_delta_medians
                    ),

                "min":
                    (
                        min(
                            timestamp_delta_medians
                        )
                        if timestamp_delta_medians
                        else None
                    ),

                "median":
                    (
                        median(
                            timestamp_delta_medians
                        )
                        if timestamp_delta_medians
                        else None
                    ),

                "max":
                    (
                        max(
                            timestamp_delta_medians
                        )
                        if timestamp_delta_medians
                        else None
                    ),
            },

        "timestamp_duration_error_summary":
            error_summary,

        "timestamp_duration_error_examples":
            sorted(
                timestamp_duration_errors,
                key=lambda item:
                    abs(
                        item[
                            "error_ms"
                        ]
                    ),
                reverse=True,
            )[:30],

        "source_root_counts":
            dict(
                source_root_counts
            ),

        "examples":
            examples,

        "original_sample_time_correspondence_qualified":
            sample_time_qualified,
    }


def source_convention_evidence():
    if not RISK_REPO.is_dir():
        return {
            "repo_exists":
                False,

            "files":
                [],
        }

    terms = re.compile(
        r"1000|100\s*\+|"
        r"UNIVR|KFALL|"
        r"fall_start_position|"
        r"impact_position|"
        r"FrameCounter|"
        r"Fall_onset|"
        r"onset_frame|"
        r"impact_frame",
        re.IGNORECASE,
    )

    interesting = []

    preferred = [
        RISK_REPO
        / "risk_analysis/"
          "build_event_index.py",

        RISK_REPO
        / "risk_analysis/"
          "dataset.py",

        RISK_REPO
        / "cascade_pipeline/"
          "index_builder.py",

        RISK_REPO
        / "dataloaders/"
          "MasterFixedFoldDataloader.py",
    ]

    for path in preferred:
        if not path.is_file():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        lines = text.splitlines()

        matches = []

        for i, line in enumerate(
            lines,
            start=1,
        ):
            if not terms.search(
                line
            ):
                continue

            start = max(
                1,
                i - 6,
            )

            end = min(
                len(lines),
                i + 8,
            )

            matches.append(
                {
                    "match_line":
                        i,

                    "context":
                        "\n".join(
                            f"{n}: {lines[n - 1]}"
                            for n
                            in range(
                                start,
                                end + 1,
                            )
                        ),
                }
            )

        interesting.append(
            {
                "path":
                    str(
                        path.resolve()
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "matches":
                    matches[:50],
            }
        )

    return {
        "repo_exists":
            True,

        "files":
            interesting,
    }


def main():
    processed = (
        processed_inventory()
    )

    events = (
        load_events()
    )

    reconciliation = reconcile(
        processed,
        events,
    )

    print(
        "Processed trials:",
        processed[
            "all_count"
        ],
    )

    print(
        "Processed fall-positive trials:",
        processed[
            "fall_count"
        ],
    )

    print()
    print(
        "Curated annotated events:",
        events[
            "event_count"
        ],
    )

    print(
        "Event namespace failures:",
        events[
            "namespace_failure_count"
        ],
    )

    print()
    print(
        "POSITION CONVENTIONS"
    )

    for item in events[
        "position_conventions"
    ]:
        print(item)

    print()
    print(
        "CANONICAL RECONCILIATION"
    )

    for name in [
        "annotated_event_count",
        "protocol_eligible_event_count",
        "protocol_ineligible_event_count",
        "processed_fall_positive_trial_count",
        "event_missing_processed_trial_count",
        "eligible_event_processed_fall_intersection",
        "eligible_event_missing_processed_fall_count",
        "processed_fall_without_eligible_event_count",
        "ineligible_event_processed_as_fall_positive_count",
    ]:
        print(
            name,
            "=",
            reconciliation[
                name
            ],
        )

    print(
        "by_dataset =",
        reconciliation[
            "by_dataset"
        ],
    )

    print()
    print(
        "PROTOCOL-INELIGIBLE EVENTS"
    )

    for item in (
        reconciliation[
            "ineligible_events"
        ]
    ):
        print(item)

    if (
        reconciliation[
            "eligible_event_missing_processed_fall_count"
        ]
        or reconciliation[
            "processed_fall_without_eligible_event_count"
        ]
        or reconciliation[
            "event_missing_processed_trial_count"
        ]
    ):
        print()
        print(
            "RECONCILIATION DISCREPANCIES"
        )

        for name in [
            "event_missing_processed_trials",
            "eligible_event_missing_processed_fall",
            "processed_fall_without_eligible_event",
        ]:
            print()
            print(name)

            for item in (
                reconciliation[
                    name
                ][:50]
            ):
                print(item)

    uni = audit_univr_original(
        events
    )

    print()
    print(
        "UNIVR ORIGINAL DATA AUDIT"
    )

    print(
        "candidate roots:",
        uni[
            "candidate_roots"
        ],
    )

    for name in [
        "event_count",
        "resolved_count",
        "missing_count",
        "ambiguous_count",
        "parse_failure_count",
        "position_out_of_range_count",
        "timestamp_missing_count",
        "timestamp_unusable_count",
        "framecounter_missing_count",
    ]:
        print(
            name,
            "=",
            uni[
                name
            ],
        )

    print(
        "framecounter_offsets =",
        uni[
            "framecounter_offsets"
        ],
    )

    print(
        "timestamp_unit_counts =",
        uni[
            "timestamp_unit_counts"
        ],
    )

    print(
        "timestamp delta summary =",
        uni[
            "timestamp_median_delta_seconds_summary"
        ],
    )

    print(
        "timestamp duration error summary =",
        uni[
            "timestamp_duration_error_summary"
        ],
    )

    print(
        "source_root_counts =",
        uni[
            "source_root_counts"
        ],
    )

    print(
        "UNIVR_ORIGINAL_SAMPLE_TIME_CORRESPONDENCE=",
        (
            "QUALIFIED"
            if uni[
                "original_sample_time_correspondence_qualified"
            ]
            else "NOT_YET_QUALIFIED"
        ),
        sep="",
    )

    if (
        uni[
            "timestamp_missing_count"
        ]
        or uni[
            "parse_failure_count"
        ]
        or uni[
            "timestamp_unusable_count"
        ]
        or uni[
            "position_out_of_range_count"
        ]
    ):
        print()
        print(
            "UNIVR ORIGINAL AUDIT EXCEPTIONS"
        )

        for name in [
            "timestamp_missing",
            "parse_failures",
            "timestamp_unusable",
            "position_out_of_range",
        ]:
            print()
            print(name)

            for item in (
                uni[
                    name
                ][:20]
            ):
                print(item)

    source = (
        source_convention_evidence()
    )

    print()
    print(
        "SOURCE CONVENTION EVIDENCE"
    )

    for item in source[
        "files"
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
            ][:15]
        ):
            print(
                match[
                    "context"
                ]
            )

    canonical_reconciliation_pass = (
        reconciliation[
            "annotated_event_count"
        ]
        == 2919
        and reconciliation[
            "protocol_eligible_event_count"
        ]
        == 2918
        and reconciliation[
            "protocol_ineligible_event_count"
        ]
        == 1
        and reconciliation[
            "processed_fall_positive_trial_count"
        ]
        == 2918
        and reconciliation[
            "event_missing_processed_trial_count"
        ]
        == 0
        and reconciliation[
            "eligible_event_processed_fall_intersection"
        ]
        == 2918
        and reconciliation[
            "eligible_event_missing_processed_fall_count"
        ]
        == 0
        and reconciliation[
            "processed_fall_without_eligible_event_count"
        ]
        == 0
        and reconciliation[
            "ineligible_event_processed_as_fall_positive_count"
        ]
        == 0
    )

    ineligible_ids = [
        item[
            "event"
        ][
            "event_id"
        ]
        for item
        in reconciliation[
            "ineligible_events"
        ]
        if item[
            "event"
        ]
        is not None
    ]

    sole_expected_ineligible = (
        ineligible_ids
        == [
            "KFALL_106_T27_R05"
        ]
    )

    position_convention_map = {
        (
            item[
                "dataset"
            ],
            item[
                "position_minus_frame_onset"
            ],
            item[
                "position_minus_frame_impact"
            ]
        ):
            item[
                "count"
            ]
        for item
        in events[
            "position_conventions"
        ]
    }

    position_conventions_pass = (
        position_convention_map.get(
            (
                "KFALL",
                -1,
                -1,
            )
        )
        == 2346
        and position_convention_map.get(
            (
                "UNIVR",
                0,
                0,
            )
        )
        == 573
        and sum(
            position_convention_map.values()
        )
        == 2919
    )

    manifest = {
        "schema":
            "crosslayer_phase3m_canonical_event_binding_univr_time_v1",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),

        "status":
            (
                "PASS"
                if (
                    canonical_reconciliation_pass
                    and sole_expected_ineligible
                    and position_conventions_pass
                )
                else "FAIL"
            ),

        "audit_mode":
            "NO_MODEL_EXECUTION",

        "protocol": {
            "window_ms":
                WINDOW_MS,

            "deadline_ms_before_impact":
                DEADLINE_MS,

            "overlap":
                OVERLAP,
        },

        "canonical_subject_mapping": {
            "UNIVR": {
                "source_subject":
                    "SAxx",

                "event_namespace_subject":
                    "1000 + xx",

                "processed_storage_subject":
                    "xx",
            },

            "KFALL": {
                "source_subject":
                    "SAxx",

                "event_namespace_subject":
                    "100 + xx",

                "processed_storage_subject":
                    "100 + xx",
            },
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

        "curated_events": {
            key:
                value
            for key, value
            in events.items()
            if key
            != "records"
        },

        "canonical_reconciliation":
            reconciliation,

        "canonical_reconciliation_pass":
            canonical_reconciliation_pass,

        "sole_expected_protocol_ineligible_event":
            sole_expected_ineligible,

        "dataset_specific_position_conventions_pass":
            position_conventions_pass,

        "univr_original_time_audit":
            uni,

        "source_convention_evidence":
            source,

        "timing_status": {
            "kfall_framecounter":
                "QUALIFIED_IN_PHASE_3L",

            "univr_event_position_convention":
                (
                    "QUALIFIED_DATASET_SPECIFIC_CONVENTION"
                    if position_conventions_pass
                    else "NOT_YET_QUALIFIED"
                ),

            "univr_original_sample_time_correspondence":
                (
                    "QUALIFIED"
                    if uni[
                        "original_sample_time_correspondence_qualified"
                    ]
                    else "NOT_YET_QUALIFIED"
                ),

            "univr_video_imu_annotation_synchronization":
                (
                    "PENDING_DOCUMENTED_PROVENANCE_CONFIRMATION"
                ),

            "physical_lead_time_protocol_fully_frozen":
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

            "physical_lead_time_protocol_fully_frozen":
                False,
        },
    }

    OUT.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if manifest[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Phase-3M canonical event reconciliation failed"
        )

    print()
    print(
        "PHASE_3M_CANONICAL_EVENT_BINDING=PASS"
    )

    print(
        "PHASE_3M_PROTOCOL_ELIGIBILITY_RECONCILIATION=PASS"
    )

    print(
        "PHASE_3M_DATASET_SPECIFIC_POSITION_CONVENTIONS=PASS"
    )


if __name__ == "__main__":
    main()
