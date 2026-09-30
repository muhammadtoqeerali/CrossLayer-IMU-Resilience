from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
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

SPLIT_FILENAME = (
    "date2025_cnn400_inferred_split_v1.json"
)

DIRECT_CANDIDATES = [
    (
        HOME_TOQEER
        / "IMU_Reliability"
        / "data"
        / "manifests"
        / SPLIT_FILENAME
    ),
    (
        HOME_TOQEER
        / "RC-RGD-IMU_publish"
        / "data"
        / "manifests"
        / SPLIT_FILENAME
    ),
    (
        HOME_TOQEER
        / "RC-RGD-IMU"
        / "data"
        / "manifests"
        / SPLIT_FILENAME
    ),
]

SEARCH_ROOTS = [
    HOME_TOQEER / "IMU_Reliability",
    HOME_TOQEER / "RC-RGD-IMU_publish",
    HOME_TOQEER / "RC-RGD-IMU",
]

AUGMENTATION_ONLY = {
    "999",
    "1000",
}

EXPECTED = {
    "train": {
        "trial_count": 4628,
        "window_count": 510479,
    },
    "validation": {
        "trial_count": 447,
        "window_count": 89868,
    },
    "test": {
        "trial_count": 1116,
        "window_count": 366507,
    },
    "augmentation_only": {
        "trial_count": 2,
        "window_count": 220472,
    },
    "total": {
        "trial_count": 6193,
        "window_count": 1187326,
    },
}

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_3b_historical_split_recovery_v1.json"
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


def find_split_manifests() -> list[Path]:
    found = []

    for path in DIRECT_CANDIDATES:
        if path.is_file():
            found.append(
                path.resolve()
            )

    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue

        try:
            matches = root.rglob(
                SPLIT_FILENAME
            )
        except Exception:
            continue

        for path in matches:
            if path.is_file():
                found.append(
                    path.resolve()
                )

    unique = []

    seen = set()

    for path in found:
        value = str(path)

        if value in seen:
            continue

        seen.add(value)
        unique.append(path)

    return sorted(
        unique,
        key=lambda value: str(value),
    )


def normalize_subjects(
    values,
) -> list[str]:
    result = []

    for value in values:
        text = str(value).strip()

        if not text:
            raise RuntimeError(
                "Blank subject identifier"
            )

        result.append(text)

    if len(result) != len(set(result)):
        raise RuntimeError(
            "Duplicate subject identifier within split"
        )

    return result


def numeric_sort(
    values,
) -> list[str]:
    def key(value: str):
        if value.isdigit():
            return (0, int(value))

        return (1, value)

    return sorted(
        values,
        key=key,
    )


def parse_split(
    path: Path,
) -> dict[str, Any]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    required = (
        "train_subjects",
        "validation_subjects",
        "test_subjects",
    )

    for key in required:
        if key not in data:
            raise RuntimeError(
                f"Missing split key {key} in {path}"
            )

    train = normalize_subjects(
        data["train_subjects"]
    )

    validation = normalize_subjects(
        data["validation_subjects"]
    )

    test = normalize_subjects(
        data["test_subjects"]
    )

    train_set = set(train)
    validation_set = set(validation)
    test_set = set(test)

    if train_set & validation_set:
        raise RuntimeError(
            "Train/validation subject overlap"
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test subject overlap"
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation/test subject overlap"
        )

    if (
        AUGMENTATION_ONLY
        & (
            train_set
            | validation_set
            | test_set
        )
    ):
        raise RuntimeError(
            "Augmentation-only subjects appear in protected split"
        )

    return {
        "path":
            str(path),

        "sha256":
            sha256_file(path),

        "train_subjects":
            numeric_sort(train),

        "validation_subjects":
            numeric_sort(validation),

        "test_subjects":
            numeric_sort(test),

        "train_count":
            len(train),

        "validation_count":
            len(validation),

        "test_count":
            len(test),

        "union_count":
            len(
                train_set
                | validation_set
                | test_set
            ),

        "raw":
            data,
    }


def window_count(
    path: Path,
) -> int:
    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if array.size == 0:
        return 0

    if array.ndim != 3:
        raise RuntimeError(
            "Unexpected non-empty segment shape "
            f"{array.shape} at {path}"
        )

    if tuple(
        array.shape[1:]
    ) != (
        40,
        9,
    ):
        raise RuntimeError(
            "Unexpected protected-window shape "
            f"{array.shape} at {path}"
        )

    return int(
        array.shape[0]
    )


def inventory_combined() -> dict[str, Any]:
    if not COMBINED.is_dir():
        raise RuntimeError(
            f"Combined tree missing: {COMBINED}"
        )

    records = []

    layout_errors = []

    for seg in sorted(
        COMBINED.rglob(
            "segments.npy"
        )
    ):
        trial_dir = seg.parent

        labels = (
            trial_dir
            / "labels.npy"
        )

        if not labels.is_file():
            raise RuntimeError(
                f"Missing labels.npy beside {seg}"
            )

        relative = (
            trial_dir
            .relative_to(
                COMBINED
            )
        )

        if len(
            relative.parts
        ) != 3:
            layout_errors.append(
                relative.as_posix()
            )
            continue

        subject, task, trial = (
            relative.parts
        )

        n_windows = window_count(
            seg
        )

        labels_array = np.load(
            labels,
            mmap_mode="r",
            allow_pickle=False,
        )

        if int(
            labels_array.shape[0]
        ) != n_windows:
            raise RuntimeError(
                "Segment/label length mismatch at "
                f"{relative}"
            )

        records.append(
            {
                "subject":
                    subject,

                "task":
                    task,

                "trial":
                    trial,

                "windows":
                    n_windows,

                "relative_path":
                    relative.as_posix(),
            }
        )

    if layout_errors:
        raise RuntimeError(
            "Unexpected combined layout examples: "
            + ", ".join(
                layout_errors[:10]
            )
        )

    return {
        "records":
            records,

        "trial_count":
            len(records),

        "window_count":
            int(
                sum(
                    item["windows"]
                    for item
                    in records
                )
            ),

        "subjects":
            numeric_sort(
                {
                    item["subject"]
                    for item
                    in records
                }
            ),
    }


def assign_partition(
    subject: str,
    split: dict[str, Any],
) -> str:
    if subject in AUGMENTATION_ONLY:
        return "augmentation_only"

    if subject in set(
        split["train_subjects"]
    ):
        return "train"

    if subject in set(
        split["validation_subjects"]
    ):
        return "validation"

    if subject in set(
        split["test_subjects"]
    ):
        return "test"

    return "unassigned"


def verify_counts(
    split: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    trial_counts = Counter()
    window_counts = Counter()

    per_subject = {}

    unassigned = []

    for record in inventory[
        "records"
    ]:
        partition = assign_partition(
            record["subject"],
            split,
        )

        trial_counts[
            partition
        ] += 1

        window_counts[
            partition
        ] += int(
            record["windows"]
        )

        subject_record = per_subject.setdefault(
            record["subject"],
            {
                "partition":
                    partition,

                "trials":
                    0,

                "windows":
                    0,
            },
        )

        if (
            subject_record[
                "partition"
            ]
            != partition
        ):
            raise RuntimeError(
                "Subject assigned to multiple partitions"
            )

        subject_record[
            "trials"
        ] += 1

        subject_record[
            "windows"
        ] += int(
            record["windows"]
        )

        if partition == "unassigned":
            unassigned.append(
                record[
                    "relative_path"
                ]
            )

    result = {}

    for partition in (
        "train",
        "validation",
        "test",
        "augmentation_only",
    ):
        observed_trials = int(
            trial_counts[
                partition
            ]
        )

        observed_windows = int(
            window_counts[
                partition
            ]
        )

        expected = EXPECTED[
            partition
        ]

        result[
            partition
        ] = {
            "trial_count":
                observed_trials,

            "window_count":
                observed_windows,

            "expected_trial_count":
                expected[
                    "trial_count"
                ],

            "expected_window_count":
                expected[
                    "window_count"
                ],

            "trial_count_match":
                (
                    observed_trials
                    == expected[
                        "trial_count"
                    ]
                ),

            "window_count_match":
                (
                    observed_windows
                    == expected[
                        "window_count"
                    ]
                ),
        }

    result[
        "unassigned"
    ] = {
        "trial_count":
            int(
                trial_counts[
                    "unassigned"
                ]
            ),

        "window_count":
            int(
                window_counts[
                    "unassigned"
                ]
            ),

        "examples":
            unassigned[:50],
    }

    total_trials = int(
        sum(
            trial_counts.values()
        )
    )

    total_windows = int(
        sum(
            window_counts.values()
        )
    )

    result[
        "total"
    ] = {
        "trial_count":
            total_trials,

        "window_count":
            total_windows,

        "expected_trial_count":
            EXPECTED[
                "total"
            ][
                "trial_count"
            ],

        "expected_window_count":
            EXPECTED[
                "total"
            ][
                "window_count"
            ],

        "trial_count_match":
            (
                total_trials
                == EXPECTED[
                    "total"
                ][
                    "trial_count"
                ]
            ),

        "window_count_match":
            (
                total_windows
                == EXPECTED[
                    "total"
                ][
                    "window_count"
                ]
            ),
    }

    return {
        "partitions":
            result,

        "per_subject":
            {
                subject:
                    per_subject[
                        subject
                    ]
                for subject
                in numeric_sort(
                    per_subject
                )
            },
    }


def main() -> int:
    candidates = find_split_manifests()

    print(
        "Candidate split manifests:",
        len(candidates),
    )

    if not candidates:
        raise RuntimeError(
            "Historical split manifest was not located"
        )

    parsed = []

    for path in candidates:
        item = parse_split(
            path
        )

        parsed.append(
            item
        )

        print()
        print(
            "SPLIT MANIFEST:",
            path,
        )

        print(
            "  SHA256:",
            item["sha256"],
        )

        print(
            "  train subjects:",
            item["train_count"],
        )

        print(
            "  validation subjects:",
            item["validation_count"],
        )

        print(
            "  test subjects:",
            item["test_count"],
        )

        print(
            "  union subjects:",
            item["union_count"],
        )

    canonical = parsed[0]

    signatures = {
        json.dumps(
            {
                "train":
                    item[
                        "train_subjects"
                    ],

                "validation":
                    item[
                        "validation_subjects"
                    ],

                "test":
                    item[
                        "test_subjects"
                    ],
            },
            sort_keys=True,
        )
        for item in parsed
    }

    split_content_agreement = (
        len(signatures) == 1
    )

    if not split_content_agreement:
        raise RuntimeError(
            "Multiple historical split manifests disagree"
        )

    inventory = inventory_combined()

    verification = verify_counts(
        canonical,
        inventory,
    )

    partitions = verification[
        "partitions"
    ]

    required_partitions = (
        "train",
        "validation",
        "test",
        "augmentation_only",
        "total",
    )

    for partition in required_partitions:
        item = partitions[
            partition
        ]

        if not item[
            "trial_count_match"
        ]:
            raise RuntimeError(
                f"{partition} historical trial count mismatch"
            )

        if not item[
            "window_count_match"
        ]:
            raise RuntimeError(
                f"{partition} historical window count mismatch"
            )

    if (
        partitions[
            "unassigned"
        ][
            "trial_count"
        ]
        != 0
    ):
        raise RuntimeError(
            "Historical combined tree contains unassigned trials"
        )

    split_union = (
        set(
            canonical[
                "train_subjects"
            ]
        )
        | set(
            canonical[
                "validation_subjects"
            ]
        )
        | set(
            canonical[
                "test_subjects"
            ]
        )
    )

    combined_subjects = set(
        inventory[
            "subjects"
        ]
    )

    expected_combined_subjects = (
        split_union
        | AUGMENTATION_ONLY
    )

    if (
        combined_subjects
        != expected_combined_subjects
    ):
        raise RuntimeError(
            "Historical combined subject set does not equal "
            "split subjects plus augmentation-only subjects"
        )

    leakage = {
        "train_validation_overlap":
            numeric_sort(
                set(
                    canonical[
                        "train_subjects"
                    ]
                )
                & set(
                    canonical[
                        "validation_subjects"
                    ]
                )
            ),

        "train_test_overlap":
            numeric_sort(
                set(
                    canonical[
                        "train_subjects"
                    ]
                )
                & set(
                    canonical[
                        "test_subjects"
                    ]
                )
            ),

        "validation_test_overlap":
            numeric_sort(
                set(
                    canonical[
                        "validation_subjects"
                    ]
                )
                & set(
                    canonical[
                        "test_subjects"
                    ]
                )
            ),

        "augmentation_in_protected_split":
            numeric_sort(
                AUGMENTATION_ONLY
                & split_union
            ),
    }

    if any(
        leakage.values()
    ):
        raise RuntimeError(
            "Subject leakage detected"
        )

    manifest = {
        "schema":
            "crosslayer_phase3b_historical_split_recovery_v1",

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

        "split_role":
            "VERIFIED_HISTORICAL_CANDIDATE",

        "reason_not_new_random_split":
            (
                "The protected task model is already trained. "
                "A new subject split could place historically "
                "seen subjects into nominal held-out evaluation."
            ),

        "candidate_manifests": [
            {
                "path":
                    item["path"],

                "sha256":
                    item["sha256"],
            }
            for item
            in parsed
        ],

        "candidate_manifest_count":
            len(parsed),

        "candidate_content_agreement":
            split_content_agreement,

        "canonical_source_path":
            canonical["path"],

        "canonical_source_sha256":
            canonical["sha256"],

        "subjects": {
            "train":
                canonical[
                    "train_subjects"
                ],

            "validation":
                canonical[
                    "validation_subjects"
                ],

            "test":
                canonical[
                    "test_subjects"
                ],

            "augmentation_only":
                numeric_sort(
                    AUGMENTATION_ONLY
                ),
        },

        "subject_counts": {
            "train":
                canonical[
                    "train_count"
                ],

            "validation":
                canonical[
                    "validation_count"
                ],

            "test":
                canonical[
                    "test_count"
                ],

            "augmentation_only":
                len(
                    AUGMENTATION_ONLY
                ),
        },

        "historical_combined_tree": {
            "path":
                str(
                    COMBINED
                ),

            "trial_count":
                inventory[
                    "trial_count"
                ],

            "window_count":
                inventory[
                    "window_count"
                ],

            "subject_count":
                len(
                    inventory[
                        "subjects"
                    ]
                ),
        },

        "partition_verification":
            verification,

        "leakage_audit":
            leakage,

        "proposed_crosslayer_roles": {
            "historical_train":
                "development",

            "historical_validation":
                "calibration",

            "historical_test":
                "held_out_confirmation",

            "historical_999_1000":
                "augmentation_only_not_evaluation",
        },

        "freeze_status":
            "NOT_YET_PHASE3_FROZEN",

        "scientific_boundary": {
            "new_random_split_generated":
                False,

            "subject_membership_changed":
                False,

            "task_predictions_opened":
                False,

            "final_test_outcomes_used":
                False,

            "quantization_calibration_performed":
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
        "Historical combined trials:",
        inventory[
            "trial_count"
        ],
    )

    print(
        "Historical combined windows:",
        inventory[
            "window_count"
        ],
    )

    print()

    for partition in (
        "train",
        "validation",
        "test",
        "augmentation_only",
    ):
        item = partitions[
            partition
        ]

        print(
            f"{partition}: "
            f"trials={item['trial_count']} "
            f"windows={item['window_count']} "
            f"match=True"
        )

    print()
    print(
        "Unassigned trials:",
        partitions[
            "unassigned"
        ][
            "trial_count"
        ],
    )

    print(
        "Subject leakage:",
        any(
            leakage.values()
        ),
    )

    print()
    print(
        "PHASE_3B_HISTORICAL_SPLIT_RECOVERY=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
