from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

HOME_TOQEER = Path.home() / "toqeer"

COMBINED = Path(
    "/mnt/hdd16T/protechto/data/back/"
    "UniVrFall_KFall/segments/"
    "400ms_50ov_npseg_filt_binary"
)

SPLIT_MANIFEST = (
    ROOT
    / "manifests/"
      "phase_3b_historical_split_recovery_v1.json"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3c_label_raw_timing_semantics_v1.json"
)

EXPECTED_LABELS = {
    "Activity",
    "Falling",
}

AUGMENTATION_ONLY = {
    "999",
    "1000",
}

RAW_ROOTS = {
    "UNIVRFALL_ORIENTED": Path(
        "/mnt/hdd16T/protechto/"
        "UniVrFall_oriented/sensors_data"
    ),

    "UNIVRFALL_ORIGINAL": Path(
        "/mnt/hdd16T/protechto/"
        "UniVrFallOriginalDataset"
    ),

    "KFALL_ORIENTED": Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/"
        "KFall_oriented/sensors_data"
    ),

    "KFALL_ORIGINAL": Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/"
        "KFall/sensors_data"
    ),

    "ONFIELD_RAW": Path(
        "/mnt/hdd16T/protechto/"
        "ThirdPartyDatasets/"
        "OnFieldRecordings"
    ),
}

TRIAL_RE = re.compile(
    r"^S(?P<subject>\d+)"
    r"T(?P<task>\d+)"
    r"R(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)

LINEAGE_FILENAMES = {
    "reliability_trial_inventory_candidate_v1.json",
    "reliability_lineage_v2.json",
    "reliability_eval_registry_v1.json",
}

LINEAGE_SEARCH_ROOTS = [
    HOME_TOQEER / "IMU_Reliability",
    HOME_TOQEER / "RC-RGD-IMU_publish",
    HOME_TOQEER / "RC-RGD-IMU",
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def numeric_sort(
    values,
) -> list[str]:
    def key(value: str):
        text = str(value)

        if text.isdigit():
            return (
                0,
                int(text),
            )

        return (
            1,
            text,
        )

    return sorted(
        [
            str(value)
            for value
            in values
        ],
        key=key,
    )


def partition_map() -> dict[str, str]:
    split = json.loads(
        SPLIT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    result = {}

    for subject in split[
        "subjects"
    ]["train"]:
        result[str(subject)] = "development"

    for subject in split[
        "subjects"
    ]["validation"]:
        result[str(subject)] = "calibration"

    for subject in split[
        "subjects"
    ]["test"]:
        result[str(subject)] = "held_out_confirmation"

    for subject in split[
        "subjects"
    ]["augmentation_only"]:
        result[str(subject)] = "augmentation_only"

    return result


def locate_lineage_files() -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {
        filename: []
        for filename
        in LINEAGE_FILENAMES
    }

    seen = set()

    for root in LINEAGE_SEARCH_ROOTS:
        if not root.is_dir():
            continue

        for filename in LINEAGE_FILENAMES:
            try:
                paths = root.rglob(
                    filename
                )
            except Exception:
                continue

            for path in paths:
                if not path.is_file():
                    continue

                resolved = path.resolve()

                key = (
                    filename,
                    str(resolved),
                )

                if key in seen:
                    continue

                seen.add(key)

                item: dict[str, Any] = {
                    "path":
                        str(resolved),

                    "sha256":
                        sha256_file(
                            resolved
                        ),

                    "bytes":
                        resolved.stat().st_size,
                }

                try:
                    data = json.loads(
                        resolved.read_text(
                            encoding="utf-8"
                        )
                    )

                    if isinstance(
                        data,
                        dict,
                    ):
                        item[
                            "top_level_keys"
                        ] = sorted(
                            data.keys()
                        )

                        if (
                            filename
                            == "reliability_trial_inventory_candidate_v1.json"
                        ):
                            item[
                                "trial_count"
                            ] = len(
                                data.get(
                                    "trials",
                                    [],
                                )
                            )

                            augmentation = data.get(
                                "historical_training_augmentation",
                                {},
                            )

                            item[
                                "augmentation_trial_count"
                            ] = int(
                                augmentation.get(
                                    "trial_count",
                                    0,
                                )
                            )

                except Exception as exc:
                    item[
                        "json_error"
                    ] = repr(
                        exc
                    )

                found[
                    filename
                ].append(
                    item
                )

    return found


def load_preferred_trial_inventory(
    lineage_files: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    candidates = lineage_files[
        "reliability_trial_inventory_candidate_v1.json"
    ]

    if not candidates:
        return None, None

    preferred = sorted(
        candidates,
        key=lambda item: (
            0
            if "/IMU_Reliability/"
            in item["path"]
            else 1,
            item["path"],
        ),
    )[0]

    path = Path(
        preferred["path"]
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return (
        data,
        preferred,
    )


def audit_labels() -> dict[str, Any]:
    if not COMBINED.is_dir():
        raise RuntimeError(
            f"Historical protected tree missing: {COMBINED}"
        )

    partitions = partition_map()

    global_labels = Counter()
    partition_labels: dict[str, Counter] = defaultdict(
        Counter
    )
    partition_trials = Counter()
    partition_windows = Counter()

    trial_label_structure: dict[
        str,
        Counter,
    ] = defaultdict(
        Counter
    )

    per_subject: dict[
        str,
        dict[str, Any],
    ] = {}

    unknown_subjects = set()

    shape_errors = []
    length_errors = []
    path_errors = []
    unknown_label_examples = []

    empty_trials = []

    trial_count = 0

    for segment_path in sorted(
        COMBINED.rglob(
            "segments.npy"
        )
    ):
        trial_dir = segment_path.parent
        label_path = (
            trial_dir
            / "labels.npy"
        )

        if not label_path.is_file():
            raise RuntimeError(
                f"labels.npy missing beside {segment_path}"
            )

        relative = trial_dir.relative_to(
            COMBINED
        )

        if len(
            relative.parts
        ) != 3:
            path_errors.append(
                relative.as_posix()
            )
            continue

        subject, task, trial = (
            relative.parts
        )

        if not (
            subject.isdigit()
            and task.isdigit()
            and trial.isdigit()
        ):
            path_errors.append(
                relative.as_posix()
            )
            continue

        partition = partitions.get(
            subject
        )

        if partition is None:
            unknown_subjects.add(
                subject
            )
            continue

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

        trial_count += 1

        if segments.size == 0:
            n_windows = 0

            if segments.shape != (
                0,
            ):
                shape_errors.append(
                    {
                        "trial":
                            relative.as_posix(),

                        "segment_shape":
                            list(
                                segments.shape
                            ),
                    }
                )

        else:
            if (
                segments.ndim != 3
                or tuple(
                    segments.shape[1:]
                )
                != (
                    40,
                    9,
                )
            ):
                shape_errors.append(
                    {
                        "trial":
                            relative.as_posix(),

                        "segment_shape":
                            list(
                                segments.shape
                            ),
                    }
                )

            n_windows = int(
                segments.shape[0]
            )

        if labels.ndim != 1:
            shape_errors.append(
                {
                    "trial":
                        relative.as_posix(),

                    "label_shape":
                        list(
                            labels.shape
                        ),
                }
            )

        if int(
            labels.shape[0]
        ) != n_windows:
            length_errors.append(
                {
                    "trial":
                        relative.as_posix(),

                    "windows":
                        n_windows,

                    "labels":
                        int(
                            labels.shape[0]
                        ),
                }
            )

        if n_windows == 0:
            empty_trials.append(
                relative.as_posix()
            )

        unique, counts = np.unique(
            labels,
            return_counts=True,
        )

        trial_counts = {
            str(label):
                int(count)
            for label, count
            in zip(
                unique.tolist(),
                counts.tolist(),
            )
        }

        for label, count in (
            trial_counts.items()
        ):
            global_labels[
                label
            ] += count

            partition_labels[
                partition
            ][
                label
            ] += count

            if (
                label
                not in EXPECTED_LABELS
                and len(
                    unknown_label_examples
                ) < 50
            ):
                unknown_label_examples.append(
                    {
                        "trial":
                            relative.as_posix(),

                        "label":
                            label,
                    }
                )

        label_set = set(
            trial_counts
        )

        if not label_set:
            trial_type = "empty"

        elif label_set == {
            "Activity"
        }:
            trial_type = "activity_only"

        elif label_set == {
            "Falling"
        }:
            trial_type = "falling_only"

        elif label_set == {
            "Activity",
            "Falling",
        }:
            trial_type = "mixed"

        else:
            trial_type = "unexpected_labels"

        trial_label_structure[
            partition
        ][
            trial_type
        ] += 1

        partition_trials[
            partition
        ] += 1

        partition_windows[
            partition
        ] += n_windows

        subject_record = per_subject.setdefault(
            subject,
            {
                "partition":
                    partition,

                "trials":
                    0,

                "windows":
                    0,

                "label_counts":
                    Counter(),
            },
        )

        subject_record[
            "trials"
        ] += 1

        subject_record[
            "windows"
        ] += n_windows

        for label, count in (
            trial_counts.items()
        ):
            subject_record[
                "label_counts"
            ][
                label
            ] += count

    if path_errors:
        raise RuntimeError(
            "Unexpected protected-trial path layout"
        )

    if unknown_subjects:
        raise RuntimeError(
            "Protected tree has subjects outside recovered split: "
            f"{numeric_sort(unknown_subjects)}"
        )

    if shape_errors:
        raise RuntimeError(
            "Protected segment/label shape errors found: "
            f"{shape_errors[:5]}"
        )

    if length_errors:
        raise RuntimeError(
            "Segment/label length mismatch: "
            f"{length_errors[:5]}"
        )

    observed_label_set = set(
        global_labels
    )

    if (
        observed_label_set
        != EXPECTED_LABELS
    ):
        raise RuntimeError(
            "Historical task labels changed. "
            f"Observed={sorted(observed_label_set)}"
        )

    serializable_subjects = {}

    for subject in numeric_sort(
        per_subject
    ):
        item = per_subject[
            subject
        ]

        serializable_subjects[
            subject
        ] = {
            "partition":
                item[
                    "partition"
                ],

            "trials":
                int(
                    item[
                        "trials"
                    ]
                ),

            "windows":
                int(
                    item[
                        "windows"
                    ]
                ),

            "label_counts":
                dict(
                    sorted(
                        item[
                            "label_counts"
                        ].items()
                    )
                ),
        }

    return {
        "protected_root":
            str(
                COMBINED
            ),

        "trial_count":
            trial_count,

        "window_count":
            int(
                sum(
                    partition_windows.values()
                )
            ),

        "observed_labels":
            sorted(
                observed_label_set
            ),

        "global_label_counts":
            dict(
                sorted(
                    global_labels.items()
                )
            ),

        "partition_label_counts": {
            partition:
                dict(
                    sorted(
                        counts.items()
                    )
                )
            for partition, counts
            in sorted(
                partition_labels.items()
            )
        },

        "partition_trial_counts":
            {
                key:
                    int(value)
                for key, value
                in sorted(
                    partition_trials.items()
                )
            },

        "partition_window_counts":
            {
                key:
                    int(value)
                for key, value
                in sorted(
                    partition_windows.items()
                )
            },

        "trial_label_structure": {
            partition:
                {
                    key:
                        int(value)
                    for key, value
                    in sorted(
                        counts.items()
                    )
                }
            for partition, counts
            in sorted(
                trial_label_structure.items()
            )
        },

        "empty_trials":
            empty_trials,

        "empty_trial_count":
            len(
                empty_trials
            ),

        "per_subject":
            serializable_subjects,

        "unknown_label_examples":
            unknown_label_examples,

        "label_semantics": {
            "Activity":
                "historical task class label",

            "Falling":
                "historical task class label",

            "physical_event_time_encoded":
                False,

            "fall_onset_timestamp_encoded":
                False,

            "impact_timestamp_encoded":
                False,

            "note":
                (
                    "labels.npy is a window-level task label artifact. "
                    "No physical event timestamp is inferred from it."
                ),
        },
    }


def build_raw_index(
    root: Path,
) -> dict[
    tuple[int, int, int],
    list[Path],
]:
    result: dict[
        tuple[int, int, int],
        list[Path],
    ] = defaultdict(
        list
    )

    if not root.is_dir():
        return {}

    for path in sorted(
        root.rglob(
            "*.csv"
        )
    ):
        match = TRIAL_RE.match(
            path.name
        )

        if match is None:
            continue

        key = (
            int(
                match.group(
                    "subject"
                )
            ),
            int(
                match.group(
                    "task"
                )
            ),
            int(
                match.group(
                    "trial"
                )
            ),
        )

        result[
            key
        ].append(
            path
        )

    return dict(
        result
    )


def detect_csv_header(
    path: Path,
) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(
            8192
        )

    if not sample:
        return []

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t",
        )

        delimiter = dialect.delimiter

    except Exception:
        first_line = sample.splitlines()[
            0
        ]

        delimiter = (
            ";"
            if first_line.count(";")
            > first_line.count(",")
            else ","
        )

    first_line = sample.splitlines()[
        0
    ]

    reader = csv.reader(
        [
            first_line,
        ],
        delimiter=delimiter,
    )

    return [
        value.strip()
        for value
        in next(
            reader,
            [],
        )
    ]


def header_audit(
    root_name: str,
    index,
) -> dict[str, Any]:
    signatures = Counter()

    timestamp_present = 0
    frame_counter_present = 0

    files = 0

    examples_missing_timestamp = []
    examples_missing_counter = []

    for paths in index.values():
        for path in paths:
            files += 1

            header = detect_csv_header(
                path
            )

            normalized = {
                value.strip().lower():
                    value
                for value
                in header
            }

            signature = "|".join(
                header
            )

            signatures[
                signature
            ] += 1

            time_keys = {
                "timestamp(s)",
                "timestamp",
                "time[ms]",
                "time(ms)",
                "time",
            }

            has_time = bool(
                set(
                    normalized
                )
                & time_keys
            )

            has_counter = (
                "framecounter"
                in normalized
            )

            if has_time:
                timestamp_present += 1

            elif (
                len(
                    examples_missing_timestamp
                )
                < 10
            ):
                examples_missing_timestamp.append(
                    str(
                        path
                    )
                )

            if has_counter:
                frame_counter_present += 1

            elif (
                len(
                    examples_missing_counter
                )
                < 10
            ):
                examples_missing_counter.append(
                    str(
                        path
                    )
                )

    return {
        "root":
            root_name,

        "indexed_trial_keys":
            len(
                index
            ),

        "csv_files":
            files,

        "timestamp_field_present_files":
            timestamp_present,

        "frame_counter_present_files":
            frame_counter_present,

        "header_signature_count":
            len(
                signatures
            ),

        "header_signatures": [
            {
                "header":
                    signature,

                "count":
                    int(
                        count
                    ),
            }
            for signature, count
            in signatures.most_common(
                20
            )
        ],

        "missing_timestamp_examples":
            examples_missing_timestamp,

        "missing_counter_examples":
            examples_missing_counter,
    }


def audit_raw_roots() -> dict[str, Any]:
    indexes = {}

    audits = {}

    for name, root in RAW_ROOTS.items():
        if (
            name
            == "ONFIELD_RAW"
        ):
            audits[
                name
            ] = {
                "path":
                    str(
                        root
                    ),

                "exists":
                    root.is_dir(),

                "exact_trial_pairing_role":
                    False,

                "reason":
                    (
                        "Historical OnField trial numbering depended on "
                        "processing order; exact trial-to-file mapping is "
                        "not assumed."
                    ),
            }

            continue

        index = build_raw_index(
            root
        )

        indexes[
            name
        ] = index

        item = header_audit(
            name,
            index,
        )

        item[
            "path"
        ] = str(
            root
        )

        item[
            "exists"
        ] = root.is_dir()

        audits[
            name
        ] = item

    return {
        "indexes":
            indexes,

        "audits":
            audits,
    }


def summarize_trial_inventory(
    data: dict[str, Any] | None,
) -> dict[str, Any]:
    if data is None:
        return {
            "available":
                False,
        }

    records = data.get(
        "trials",
        [],
    )

    source_counts = Counter()

    source_status_counts = Counter()

    raw_pairing_counts = Counter()

    split_source_counts = Counter()

    original_paths = {
        "UNIVRFALL":
            set(),

        "KFALL":
            set(),
    }

    exact_original_pair_counts = Counter()

    missing_original_paths = []

    for record in records:
        dataset = str(
            record.get(
                "source_dataset",
                "UNKNOWN",
            )
        )

        split = str(
            record.get(
                "split",
                "UNKNOWN",
            )
        )

        source_status = str(
            record.get(
                "processed_source_status",
                "UNKNOWN",
            )
        )

        raw = record.get(
            "raw_pairing",
            {},
        )

        raw_status = str(
            raw.get(
                "status",
                "UNKNOWN",
            )
        )

        source_counts[
            dataset
        ] += 1

        source_status_counts[
            source_status
        ] += 1

        raw_pairing_counts[
            (
                dataset,
                raw_status,
            )
        ] += 1

        split_source_counts[
            (
                split,
                dataset,
            )
        ] += 1

        if (
            dataset
            in original_paths
            and raw_status
            == "EXACT_FILENAME_PAIR"
        ):
            paths = raw.get(
                "acquisition_original_raw",
                [],
            )

            if len(
                paths
            ) == 1:
                path = Path(
                    paths[0]
                )

                original_paths[
                    dataset
                ].add(
                    str(
                        path
                    )
                )

                exact_original_pair_counts[
                    dataset
                ] += 1

                if (
                    not path.is_file()
                    and len(
                        missing_original_paths
                    )
                    < 50
                ):
                    missing_original_paths.append(
                        str(
                            path
                        )
                    )

    augmentation = data.get(
        "historical_training_augmentation",
        {},
    )

    return {
        "available":
            True,

        "record_count":
            len(
                records
            ),

        "source_dataset_counts":
            dict(
                sorted(
                    source_counts.items()
                )
            ),

        "processed_source_status_counts":
            dict(
                sorted(
                    source_status_counts.items()
                )
            ),

        "raw_pairing_counts": [
            {
                "dataset":
                    dataset,

                "status":
                    status,

                "trial_count":
                    int(
                        count
                    ),
            }
            for (
                dataset,
                status,
            ), count
            in sorted(
                raw_pairing_counts.items()
            )
        ],

        "split_source_counts": [
            {
                "split":
                    split,

                "dataset":
                    dataset,

                "trial_count":
                    int(
                        count
                    ),
            }
            for (
                split,
                dataset,
            ), count
            in sorted(
                split_source_counts.items()
            )
        ],

        "exact_original_pair_counts":
            {
                key:
                    int(
                        value
                    )
                for key, value
                in sorted(
                    exact_original_pair_counts.items()
                )
            },

        "unique_original_raw_paths":
            {
                key:
                    len(
                        value
                    )
                for key, value
                in sorted(
                    original_paths.items()
                )
            },

        "missing_original_raw_path_count":
            len(
                missing_original_paths
            ),

        "missing_original_raw_examples":
            missing_original_paths,

        "historical_training_augmentation":
            {
                "trial_count":
                    int(
                        augmentation.get(
                            "trial_count",
                            0,
                        )
                    ),

                "total_windows":
                    int(
                        augmentation.get(
                            "total_windows",
                            0,
                        )
                    ),

                "excluded_from_reliability_split":
                    bool(
                        augmentation.get(
                            "excluded_from_reliability_split",
                            False,
                        )
                    ),
            },
    }


def main() -> int:
    labels = audit_labels()

    print(
        "Protected trials:",
        labels[
            "trial_count"
        ],
    )

    print(
        "Protected windows:",
        labels[
            "window_count"
        ],
    )

    print(
        "Observed labels:",
        labels[
            "observed_labels"
        ],
    )

    print(
        "Global label counts:",
        labels[
            "global_label_counts"
        ],
    )

    print(
        "Empty trials:",
        labels[
            "empty_trial_count"
        ],
    )

    lineage_files = locate_lineage_files()

    print()
    print("Recovered historical lineage files:")

    for filename in sorted(
        lineage_files
    ):
        print(
            " ",
            filename,
            "=",
            len(
                lineage_files[
                    filename
                ]
            ),
        )

        for item in lineage_files[
            filename
        ]:
            print(
                "    ",
                item["path"],
            )

            print(
                "      sha256:",
                item["sha256"],
            )

    (
        trial_inventory,
        trial_inventory_source,
    ) = load_preferred_trial_inventory(
        lineage_files
    )

    trial_lineage = (
        summarize_trial_inventory(
            trial_inventory
        )
    )

    print()
    print(
        "Trial inventory available:",
        trial_lineage[
            "available"
        ],
    )

    if trial_lineage[
        "available"
    ]:
        print(
            "Trial inventory records:",
            trial_lineage[
                "record_count"
            ],
        )

        print(
            "Source dataset counts:",
            trial_lineage[
                "source_dataset_counts"
            ],
        )

        print(
            "Exact original raw pairs:",
            trial_lineage[
                "exact_original_pair_counts"
            ],
        )

        print(
            "Missing original raw paths:",
            trial_lineage[
                "missing_original_raw_path_count"
            ],
        )

    raw = audit_raw_roots()

    print()
    print("Raw acquisition roots:")

    for name, item in raw[
        "audits"
    ].items():
        print()
        print(
            " ",
            name,
        )

        print(
            "    exists:",
            item[
                "exists"
            ],
        )

        if (
            name
            != "ONFIELD_RAW"
        ):
            print(
                "    indexed trial keys:",
                item[
                    "indexed_trial_keys"
                ],
            )

            print(
                "    CSV files:",
                item[
                    "csv_files"
                ],
            )

            print(
                "    timestamp field files:",
                item[
                    "timestamp_field_present_files"
                ],
            )

            print(
                "    frame counter files:",
                item[
                    "frame_counter_present_files"
                ],
            )

    # Strong raw-header invariants supported by the recovered provenance.
    kfall_original = raw[
        "audits"
    ][
        "KFALL_ORIGINAL"
    ]

    if (
        kfall_original[
            "csv_files"
        ]
        <= 0
    ):
        raise RuntimeError(
            "No KFall original raw trials indexed"
        )

    if (
        kfall_original[
            "timestamp_field_present_files"
        ]
        != kfall_original[
            "csv_files"
        ]
    ):
        raise RuntimeError(
            "KFall original raw files do not all expose timestamps"
        )

    if (
        kfall_original[
            "frame_counter_present_files"
        ]
        != kfall_original[
            "csv_files"
        ]
    ):
        raise RuntimeError(
            "KFall original raw files do not all expose FrameCounter"
        )

    univr_original = raw[
        "audits"
    ][
        "UNIVRFALL_ORIGINAL"
    ]

    if (
        univr_original[
            "csv_files"
        ]
        <= 0
    ):
        raise RuntimeError(
            "No UniVR original raw trials indexed"
        )

    if (
        univr_original[
            "timestamp_field_present_files"
        ]
        != univr_original[
            "csv_files"
        ]
    ):
        raise RuntimeError(
            "UniVR original raw files do not all expose a time field"
        )

    # We intentionally do NOT require UniVR original FrameCounter.
    # Historical provenance says that acquisition provenance is absent.
    timing_semantics = {
        "protected_window": {
            "stored_shape":
                [40, 9],

            "nominal_sampling_hz":
                100,

            "nominal_window_duration_ms":
                400,

            "tree_name":
                "400ms_50ov_npseg_filt_binary",

            "overlap_from_directory_name":
                "50ov",

            "overlap_freeze_status":
                "NOT_YET_FROZEN_FROM_PREPROCESSOR_IMPLEMENTATION",
        },

        "processed_artifact_metadata": {
            "timestamp_retained":
                False,

            "frame_counter_retained":
                False,

            "fifo_status_retained":
                False,

            "physical_impact_timestamp_retained":
                False,

            "fall_onset_timestamp_retained":
                False,
        },

        "kfall": {
            "original_timestamp_field_present":
                True,

            "original_frame_counter_present":
                True,

            "frame_counter_provenance":
                "QUALIFIED_SEQUENCE_EVIDENCE",

            "timestamp_role":
                "RAW_ACQUISITION_TIMING_EVIDENCE",

            "physical_impact_time_semantics":
                "NOT_ESTABLISHED_BY_PHASE3C",
        },

        "univr": {
            "original_time_field_present":
                True,

            "original_frame_counter_required":
                False,

            "oriented_frame_counter_hard_evidence":
                False,

            "timestamp_role":
                (
                    "RAW_LIKE_TIME_EVIDENCE_WITH_DUPLICATE/"
                    "VARIABLE_DELTAS_ALLOWED"
                ),

            "physical_impact_time_semantics":
                "NOT_ESTABLISHED_BY_PHASE3C",
        },

        "onfield": {
            "primary_evaluation_role":
                False,

            "historical_role":
                "TRAINING_AUGMENTATION_ONLY",

            "exact_raw_trial_mapping":
                False,
        },

        "preimpact_lead_time": {
            "status":
                "NOT_YET_QUALIFIED",

            "reason":
                (
                    "Window labels do not encode physical onset/impact time. "
                    "A defensible event-time anchor must be recovered or "
                    "defined separately before reporting pre-impact lead time."
                ),
        },
    }

    manifest = {
        "schema":
            "crosslayer_phase3c_label_raw_timing_semantics_v1",

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

        "task_label_audit":
            labels,

        "historical_lineage_files":
            lineage_files,

        "preferred_trial_inventory_source":
            trial_inventory_source,

        "trial_lineage_summary":
            trial_lineage,

        "raw_acquisition_audit":
            {
                name:
                    item
                for name, item
                in raw[
                    "audits"
                ].items()
            },

        "timing_semantics":
            timing_semantics,

        "phase3c_conclusions": {
            "task_label_set_verified":
                True,

            "window_labels_are_physical_event_timestamps":
                False,

            "kfall_raw_counter_qualified_as_sequence_evidence":
                True,

            "univr_oriented_counter_qualified_as_hard_acquisition_evidence":
                False,

            "onfield_eligible_for_primary_evaluation":
                False,

            "preimpact_timing_frozen":
                False,
        },

        "scientific_boundary": {
            "subject_split_changed":
                False,

            "labels_modified":
                False,

            "windows_regenerated":
                False,

            "model_predictions_opened":
                False,

            "held_out_task_outcomes_used":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,

            "detector_thresholds_selected":
                False,
        },

        "next_questions": [
            (
                "Recover the exact preprocessing/window stride "
                "implementation rather than relying only on the 50ov name."
            ),
            (
                "Determine whether a defensible physical fall-onset, "
                "impact or pre-impact event anchor survives for KFall."
            ),
            (
                "Determine the strongest defensible event-time semantics "
                "for UniVR without promoting derived metadata."
            ),
            (
                "Freeze whether the paper reports true physical lead time "
                "or a narrower window-relative decision timing metric."
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
        "Task label set verified:",
        True,
    )

    print(
        "KFall raw timing + FrameCounter:",
        True,
    )

    print(
        "UniVR original time field:",
        True,
    )

    print(
        "UniVR oriented counter hard-qualified:",
        False,
    )

    print(
        "Pre-impact timing frozen:",
        False,
    )

    print()
    print(
        "PHASE_3C_LABEL_RAW_TIMING_AUDIT=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
