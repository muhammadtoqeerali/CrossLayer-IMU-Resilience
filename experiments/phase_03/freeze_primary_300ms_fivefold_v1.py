from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

PHASE3H = (
    ROOT
    / "manifests/"
      "phase_3h_primary_300ms_root_binding_v1.json"
)

PHASE3I = (
    ROOT
    / "manifests/"
      "phase_3i_onfield_external_role_v1.json"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3j_primary_300ms_fivefold_v1.json"
)

CONFIG_OUTPUT = (
    ROOT
    / "configs/datasets/"
      "primary_300ms_fivefold_v1.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def dataset_content_digest(
    root: Path,
) -> dict[str, Any]:

    files = sorted(
        [
            *root.rglob("segments.npy"),
            *root.rglob("labels.npy"),
        ],
        key=lambda path:
            path.relative_to(
                root
            ).as_posix(),
    )

    digest = hashlib.sha256()

    total_bytes = 0

    for index, path in enumerate(
        files,
        start=1,
    ):
        relative = (
            path.relative_to(
                root
            ).as_posix()
        )

        size = int(
            path.stat().st_size
        )

        file_sha = sha256_file(
            path
        )

        total_bytes += size

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

        digest.update(
            str(size).encode(
                "ascii"
            )
        )

        digest.update(b"\0")

        digest.update(
            file_sha.encode(
                "ascii"
            )
        )

        digest.update(b"\n")

        if (
            index % 2000
            == 0
        ):
            print(
                "  hashed files:",
                index,
                "/",
                len(files),
            )

    return {
        "algorithm":
            (
                "SHA256("
                "relative_path + NUL + size + NUL + "
                "file_sha256 + newline"
                ")"
            ),

        "file_count":
            len(files),

        "total_bytes":
            total_bytes,

        "digest":
            digest.hexdigest(),
    }


def canonical_subject(
    storage_id: str,
) -> dict[str, str]:

    value = int(
        storage_id
    )

    if 9 <= value <= 37:
        return {
            "storage_id":
                storage_id,

            "dataset":
                "UNIVR",

            "dataset_subject_id":
                f"{value:02d}",

            "canonical_id":
                f"UNIVR_{value:02d}",
        }

    kfall_subject = (
        value
        - 100
    )

    valid_kfall = {
        6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 35,
        36, 37, 38,
    }

    if (
        kfall_subject
        in valid_kfall
    ):
        return {
            "storage_id":
                storage_id,

            "dataset":
                "KFALL",

            "dataset_subject_id":
                f"{kfall_subject:02d}",

            "canonical_id":
                f"KFALL_{kfall_subject:02d}",
        }

    raise RuntimeError(
        "Unresolved primary subject: "
        f"{storage_id}"
    )


def convert_subjects(
    storage_ids,
) -> list[dict[str, str]]:
    return [
        canonical_subject(
            str(value)
        )
        for value
        in storage_ids
    ]


def ids_only(
    records,
    key,
) -> list[str]:
    return [
        item[key]
        for item
        in records
    ]


def dataset_counts(
    records,
) -> dict[str, int]:
    counts = Counter(
        item[
            "dataset"
        ]
        for item
        in records
    )

    return dict(
        sorted(
            counts.items()
        )
    )


def main() -> None:

    p3h = json.loads(
        PHASE3H.read_text(
            encoding="utf-8"
        )
    )

    p3i = json.loads(
        PHASE3I.read_text(
            encoding="utf-8"
        )
    )

    assert p3h["status"] == "PASS"
    assert p3i["status"] == "PASS"

    if not DATA_ROOT.is_dir():
        raise RuntimeError(
            f"Primary dataset root missing: {DATA_ROOT}"
        )

    # Historical loader semantics:
    # sorted(os.listdir(root_directory))
    storage_subjects = sorted(
        [
            child.name
            for child
            in DATA_ROOT.iterdir()
            if (
                child.is_dir()
                and child.name.isdigit()
            )
        ]
    )

    if len(
        storage_subjects
    ) != 61:
        raise RuntimeError(
            "Expected exactly 61 primary subjects, got "
            f"{len(storage_subjects)}"
        )

    subject_records = (
        convert_subjects(
            storage_subjects
        )
    )

    if len(
        {
            item[
                "canonical_id"
            ]
            for item
            in subject_records
        }
    ) != 61:
        raise RuntimeError(
            "Canonical identity collision"
        )

    if (
        dataset_counts(
            subject_records
        )
        != {
            "KFALL": 32,
            "UNIVR": 29,
        }
    ):
        raise RuntimeError(
            "Unexpected primary dataset composition"
        )

    subjects = np.array(
        storage_subjects,
        dtype=object,
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    folds = []

    test_occurrences = Counter()

    all_storage = set(
        storage_subjects
    )

    for fold_number, (
        outer_train_idx,
        outer_test_idx,
    ) in enumerate(
        kfold.split(
            subjects
        ),
        start=1,
    ):

        train_idx, validation_idx = (
            train_test_split(
                outer_train_idx,
                test_size=0.2,
                random_state=42,
            )
        )

        train_storage = (
            subjects[
                train_idx
            ].tolist()
        )

        validation_storage = (
            subjects[
                validation_idx
            ].tolist()
        )

        test_storage = (
            subjects[
                outer_test_idx
            ].tolist()
        )

        train_set = set(
            train_storage
        )

        validation_set = set(
            validation_storage
        )

        test_set = set(
            test_storage
        )

        if (
            train_set
            & validation_set
        ):
            raise RuntimeError(
                f"Fold {fold_number}: train/validation overlap"
            )

        if (
            train_set
            & test_set
        ):
            raise RuntimeError(
                f"Fold {fold_number}: train/test overlap"
            )

        if (
            validation_set
            & test_set
        ):
            raise RuntimeError(
                f"Fold {fold_number}: validation/test overlap"
            )

        if (
            train_set
            | validation_set
            | test_set
        ) != all_storage:
            raise RuntimeError(
                f"Fold {fold_number}: incomplete population"
            )

        for subject in (
            test_storage
        ):
            test_occurrences[
                subject
            ] += 1

        train_records = (
            convert_subjects(
                train_storage
            )
        )

        validation_records = (
            convert_subjects(
                validation_storage
            )
        )

        test_records = (
            convert_subjects(
                test_storage
            )
        )

        fold = {
            "fold":
                fold_number,

            "train": {
                "subject_count":
                    len(
                        train_records
                    ),

                "storage_ids":
                    ids_only(
                        train_records,
                        "storage_id",
                    ),

                "canonical_ids":
                    ids_only(
                        train_records,
                        "canonical_id",
                    ),

                "dataset_counts":
                    dataset_counts(
                        train_records
                    ),
            },

            "validation": {
                "subject_count":
                    len(
                        validation_records
                    ),

                "storage_ids":
                    ids_only(
                        validation_records,
                        "storage_id",
                    ),

                "canonical_ids":
                    ids_only(
                        validation_records,
                        "canonical_id",
                    ),

                "dataset_counts":
                    dataset_counts(
                        validation_records
                    ),
            },

            "outer_test": {
                "subject_count":
                    len(
                        test_records
                    ),

                "storage_ids":
                    ids_only(
                        test_records,
                        "storage_id",
                    ),

                "canonical_ids":
                    ids_only(
                        test_records,
                        "canonical_id",
                    ),

                "dataset_counts":
                    dataset_counts(
                        test_records
                    ),
            },
        }

        folds.append(
            fold
        )

    if set(
        test_occurrences
    ) != all_storage:
        raise RuntimeError(
            "Outer-test coverage does not include all 61 subjects"
        )

    if not all(
        count == 1
        for count
        in test_occurrences.values()
    ):
        raise RuntimeError(
            "Every subject must appear in outer test exactly once"
        )

    expected_counts = [
        (
            fold[
                "train"
            ][
                "subject_count"
            ],
            fold[
                "validation"
            ][
                "subject_count"
            ],
            fold[
                "outer_test"
            ][
                "subject_count"
            ],
        )
        for fold
        in folds
    ]

    print(
        "Fold sizes:",
        expected_counts,
    )

    # Expected sklearn geometry for 61 subjects:
    # Fold 1: outer-test 13, outer-train 48 -> validation 10 -> train 38
    # Folds 2-5: outer-test 12, outer-train 49 -> validation 10 -> train 39
    if expected_counts != [
        (38, 10, 13),
        (39, 10, 12),
        (39, 10, 12),
        (39, 10, 12),
        (39, 10, 12),
    ]:
        raise RuntimeError(
            "Unexpected 61-subject fold geometry: "
            f"{expected_counts}"
        )

    print()
    print(
        "Computing deterministic primary-dataset content digest..."
    )

    dataset_digest = (
        dataset_content_digest(
            DATA_ROOT
        )
    )

    print(
        "Dataset file count:",
        dataset_digest[
            "file_count"
        ],
    )

    print(
        "Dataset bytes:",
        dataset_digest[
            "total_bytes"
        ],
    )

    print(
        "Dataset content digest:",
        dataset_digest[
            "digest"
        ],
    )

    retained_onfield = (
        p3i[
            "prospective_onfield_policy"
        ][
            "retained_storage_ids"
        ]
    )

    rejected_onfield = (
        p3i[
            "rejected_onfield_cases"
        ][
            "storage_ids"
        ]
    )

    if retained_onfield != [
        str(value)
        for value
        in range(
            1001,
            1011,
        )
    ]:
        raise RuntimeError(
            "Unexpected retained OnField IDs"
        )

    if rejected_onfield != [
        "999",
        "1000",
    ]:
        raise RuntimeError(
            "Unexpected rejected OnField IDs"
        )

    split_policy = {
        "status":
            "FROZEN",

        "population":
            "UNIVR_PLUS_KFALL_61",

        "subject_count":
            61,

        "fold_count":
            5,

        "subject_order":
            "LEXICOGRAPHIC_STORAGE_DIRECTORY_NAME",

        "outer_split": {
            "algorithm":
                "sklearn.model_selection.KFold",

            "n_splits":
                5,

            "shuffle":
                True,

            "random_state":
                42,
        },

        "inner_validation_split": {
            "algorithm":
                "sklearn.model_selection.train_test_split",

            "source":
                "outer_non_test_indices_only",

            "test_size":
                0.2,

            "random_state":
                42,
        },

        "folds":
            folds,

        "coverage": {
            "every_subject_outer_test_exactly_once":
                True,

            "subject_leakage":
                False,
        },
    }

    role_policy = {
        "training": {
            "model_fit_allowed":
                True,

            "future_int8_calibration_source_allowed":
                True,

            "fault_parameter_development_allowed":
                True,

            "outer_test_information_allowed":
                False,
        },

        "validation": {
            "model_selection_allowed":
                True,

            "decision_threshold_selection_allowed":
                True,

            "monitor_protection_calibration_allowed":
                True,

            "supervisor_recovery_calibration_allowed":
                True,

            "int8_calibration_source_allowed":
                False,
        },

        "outer_test": {
            "model_fit_allowed":
                False,

            "model_selection_allowed":
                False,

            "int8_calibration_allowed":
                False,

            "fault_parameter_selection_allowed":
                False,

            "threshold_tuning_allowed":
                False,

            "protection_tuning_allowed":
                False,

            "evaluation_only":
                True,
        },

        "onfield_1001_1010": {
            "activity_only":
                True,

            "training_allowed":
                False,

            "validation_allowed":
                False,

            "int8_calibration_allowed":
                False,

            "fault_parameter_selection_allowed":
                False,

            "threshold_tuning_allowed":
                False,

            "external_evaluation_only":
                True,
        },

        "onfield_999_1000": {
            "use_anywhere":
                False,

            "historical_provenance_only":
                True,
        },
    }

    manifest = {
        "schema":
            "crosslayer_phase3j_primary_300ms_fivefold_v1",

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

        "freeze_status":
            "FIVEFOLD_MEMBERSHIP_FROZEN",

        "audit_mode":
            "NO_MODEL_EXECUTION",

        "dataset": {
            "root":
                str(
                    DATA_ROOT
                ),

            "window_ms":
                300,

            "sampling_hz":
                100,

            "samples_per_window":
                30,

            "overlap_percent":
                50,

            "stride_samples":
                15,

            "stride_ms":
                150,

            "subject_count":
                61,

            "dataset_counts": {
                "UNIVR":
                    29,

                "KFALL":
                    32,
            },

            "content_digest":
                dataset_digest,
        },

        "subject_identity": {
            "canonical_subjects":
                subject_records,

            "canonical_count":
                len(
                    subject_records
                ),

            "storage_ids_are_scientific_cross_dataset_ids":
                False,
        },

        "split_policy":
            split_policy,

        "partition_role_policy":
            role_policy,

        "external_onfield": {
            "retained_storage_ids":
                retained_onfield,

            "retained_count":
                10,

            "activity_only":
                True,

            "rejected_storage_ids":
                rejected_onfield,
        },

        "scientific_boundary": {
            "model_trained":
                False,

            "model_prediction_executed":
                False,

            "outer_test_opened":
                False,

            "onfield_predictions_opened":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,

            "thresholds_selected":
                False,
        },

        "remaining_before_full_phase3_freeze": [
            (
                "Map UniVRFall fall-onset and impact annotations "
                "to primary processed trial identities."
            ),
            (
                "Map KFall fall-onset and impact annotations "
                "to primary processed trial identities."
            ),
            (
                "Freeze causal window timestamp convention."
            ),
            (
                "Freeze physical lead-time equation and eligible events."
            ),
            (
                "Freeze deterministic training-only INT8 calibration "
                "sample selection."
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

    CONFIG_OUTPUT.write_text(
        json.dumps(
            {
                "schema":
                    "crosslayer_primary_300ms_fivefold_v1",

                "status":
                    "FROZEN",

                "dataset_root":
                    str(
                        DATA_ROOT
                    ),

                "dataset_content_sha256":
                    dataset_digest[
                        "digest"
                    ],

                "window_ms":
                    300,

                "sampling_hz":
                    100,

                "overlap_percent":
                    50,

                "stride_ms":
                    150,

                "subjects":
                    subject_records,

                "folds":
                    folds,

                "partition_role_policy":
                    role_policy,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "FIVEFOLD_SUBJECT_COVERAGE=PASS"
    )

    print(
        "FIVEFOLD_SUBJECT_LEAKAGE=PASS"
    )

    print(
        "FIVEFOLD_MEMBERSHIP_FREEZE=PASS"
    )

    for fold in folds:
        print()
        print(
            "FOLD",
            fold["fold"],
        )

        for role in (
            "train",
            "validation",
            "outer_test",
        ):
            item = fold[role]

            print(
                " ",
                role,
                "n=",
                item[
                    "subject_count"
                ],
                "datasets=",
                item[
                    "dataset_counts"
                ],
            )

            print(
                "   ",
                item[
                    "canonical_ids"
                ],
            )

    print()
    print(
        "PHASE_3J_PRIMARY_FIVEFOLD_FREEZE=PASS"
    )


if __name__ == "__main__":
    main()
