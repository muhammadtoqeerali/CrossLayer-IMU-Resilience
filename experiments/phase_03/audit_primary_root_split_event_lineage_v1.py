from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

PHASE3B = (
    ROOT
    / "manifests"
    / "phase_3b_historical_split_recovery_v1.json"
)

PHASE3F = (
    ROOT
    / "manifests"
    / "phase_3f_exact_fivefold_and_simulator_lineage_v1.json"
)

PRIMARY_ROOT = Path(
    "/mnt/hdd16T/protechto/data/back/"
    "UniVrFall_KFall/segments/"
    "400ms_50ov_npseg_filt_binary"
)

PRIMARY_CHECKPOINT = Path(
    "/mnt/hdd16T/protechto/checkpoints/"
    "CNN/400ms/2025-02-25_12_24_47/"
    "best-checkpoint.ckpt"
)

PRIMARY_CHECKPOINT_EXPECTED_SHA256 = (
    "ee7c0079bfb8555bff45c3077cc24eaa"
    "4373c57729045d92a831a1d7a3ea9bb1"
)

RUN_ID = "2025-02-25_12_24_47"

SEARCH_ROOTS = [
    Path("/mnt/hdd16T/protechto"),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "HR_LR_Fallings"
    ),
]

RISK_ROOTS = [
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "HR_LR_Fallings/risk_data_generated"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori/risk_data_generated"
    ),
]

SUBJECT_FOLD_FILENAMES = [
    "SUBJECT_FOLDS_COMBINED_CERTAIN_V1.json",
]

EVENT_FILENAMES = [
    "FALL_EVENT_INDEX_COMBINED_LABELED.csv",
    "FALL_EVENT_INDEX_COMBINED_UNLABELED.csv",
    "FALL_EVENT_INDEX_KFALL_UNLABELED.csv",
    "FALL_EVENT_INDEX_UNIVR_UNLABELED.csv",
]

TEXT_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".md",
    ".toml",
    ".sh",
}

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_3g_primary_root_split_event_lineage_v1.json"
)


# ------------------------------------------------------------------
# Historical verified split from Phase 3B
# ------------------------------------------------------------------

HISTORICAL_TRAIN = {
    "9", "14", "15", "16", "18", "19", "20", "21",
    "22", "23", "25", "26", "27", "28", "32", "34",
    "35", "37",
    "106", "107", "110", "111", "113", "114", "115",
    "116", "117", "118", "119", "120", "122", "123",
    "125", "127", "128", "129", "130", "131", "133",
    "135", "136", "138",
    "1002", "1003", "1004", "1008", "1009",
}

HISTORICAL_VALIDATION = {
    "24",
    "29",
    "30",
    "109",
    "126",
    "1007",
}

HISTORICAL_TEST = {
    "17",
    "31",
    "33",
    "36",
    "108",
    "112",
    "121",
    "124",
    "132",
    "137",
    "1001",
    "1005",
    "1006",
    "1010",
}

AUGMENTATION_ONLY = {
    "999",
    "1000",
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


def read_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def simple_checkpoint_metadata(
    path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }

    if not path.is_file():
        return result

    result["bytes"] = int(
        path.stat().st_size
    )

    result["sha256"] = sha256_file(
        path
    )

    result[
        "expected_sha256"
    ] = PRIMARY_CHECKPOINT_EXPECTED_SHA256

    result[
        "sha256_verified"
    ] = (
        result["sha256"]
        == PRIMARY_CHECKPOINT_EXPECTED_SHA256
    )

    try:
        import torch

        for candidate in [
            Path(
                "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
                "Protechto-master"
            ),
            Path(
                "/mnt/hdd16T/protechto"
            ),
        ]:
            if candidate.is_dir():
                sys.path.insert(
                    0,
                    str(candidate),
                )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        result[
            "top_level_type"
        ] = type(
            checkpoint
        ).__name__

        if isinstance(
            checkpoint,
            dict,
        ):
            result[
                "top_level_keys"
            ] = sorted(
                str(key)
                for key
                in checkpoint
            )

            hparams = checkpoint.get(
                "hyper_parameters"
            )

            if isinstance(
                hparams,
                dict,
            ):
                simple = {}

                for key, value in hparams.items():
                    if isinstance(
                        value,
                        (
                            str,
                            int,
                            float,
                            bool,
                            type(None),
                        ),
                    ):
                        simple[
                            str(key)
                        ] = value

                    elif isinstance(
                        value,
                        (
                            list,
                            tuple,
                        ),
                    ) and all(
                        isinstance(
                            item,
                            (
                                str,
                                int,
                                float,
                                bool,
                                type(None),
                            ),
                        )
                        for item
                        in value
                    ):
                        simple[
                            str(key)
                        ] = list(value)

                result[
                    "simple_hyper_parameters"
                ] = simple

        result[
            "load_status"
        ] = "PASS"

    except Exception as exc:
        result[
            "load_status"
        ] = "FAILED_NONBLOCKING"

        result[
            "load_error"
        ] = repr(exc)

    return result


def reconstruct_primary_subject_set() -> dict[str, Any]:
    if not PRIMARY_ROOT.is_dir():
        raise RuntimeError(
            f"Primary candidate root missing: {PRIMARY_ROOT}"
        )

    raw = sorted(
        [
            child.name
            for child
            in PRIMARY_ROOT.iterdir()
            if (
                child.is_dir()
                and child.name.isdigit()
            )
        ]
    )

    real = [
        value
        for value
        in raw
        if value not in AUGMENTATION_ONLY
    ]

    historical_real = (
        HISTORICAL_TRAIN
        | HISTORICAL_VALIDATION
        | HISTORICAL_TEST
    )

    return {
        "path":
            str(PRIMARY_ROOT),

        "raw_subjects":
            raw,

        "raw_subject_count":
            len(raw),

        "augmentation_subjects_present":
            sorted(
                set(raw)
                & AUGMENTATION_ONLY
            ),

        "real_subjects":
            real,

        "real_subject_count":
            len(real),

        "historical_subject_count":
            len(
                historical_real
            ),

        "subject_set_equals_historical":
            (
                set(real)
                == historical_real
            ),

        "historical_missing_from_root":
            sorted(
                historical_real
                - set(real)
            ),

        "root_extra_real_subjects":
            sorted(
                set(real)
                - historical_real
            ),
    }


def compare_phase3f_folds() -> dict[str, Any]:
    data = read_json(
        PHASE3F
    )

    candidates = data[
        "candidate_root_reconstructions"
    ]

    target = None

    for item in candidates:
        if (
            Path(
                item["path"]
            )
            == PRIMARY_ROOT
        ):
            target = item
            break

    if target is None:
        raise RuntimeError(
            "Primary root not present in Phase-3F evidence"
        )

    historical_outer_train = (
        HISTORICAL_TRAIN
        | HISTORICAL_VALIDATION
    )

    comparisons = []

    for fold in target["folds"]:
        train = set(
            str(value)
            for value
            in fold["train_subjects"]
        )

        validation = set(
            str(value)
            for value
            in fold[
                "validation_subjects"
            ]
        )

        test = set(
            str(value)
            for value
            in fold["test_subjects"]
        )

        outer_non_test = (
            train
            | validation
        )

        comparisons.append(
            {
                "fold":
                    int(
                        fold["fold"]
                    ),

                "test_exact_historical_match":
                    (
                        test
                        == HISTORICAL_TEST
                    ),

                "test_intersection_count":
                    len(
                        test
                        & HISTORICAL_TEST
                    ),

                "outer_non_test_exact_historical_match":
                    (
                        outer_non_test
                        == historical_outer_train
                    ),

                "historical_validation_equals_fold_validation":
                    (
                        validation
                        == HISTORICAL_VALIDATION
                    ),

                "historical_train_equals_fold_train":
                    (
                        train
                        == HISTORICAL_TRAIN
                    ),

                "fold_train_count":
                    len(train),

                "fold_validation_count":
                    len(validation),

                "fold_test_count":
                    len(test),

                "fold_validation_subjects":
                    sorted(
                        validation
                    ),

                "historical_validation_subjects":
                    sorted(
                        HISTORICAL_VALIDATION
                    ),

                "validation_intersection":
                    sorted(
                        validation
                        & HISTORICAL_VALIDATION
                    ),

                "test_subjects":
                    sorted(test),
            }
        )

    exact_test_matches = [
        item["fold"]
        for item
        in comparisons
        if item[
            "test_exact_historical_match"
        ]
    ]

    exact_outer_matches = [
        item["fold"]
        for item
        in comparisons
        if item[
            "outer_non_test_exact_historical_match"
        ]
    ]

    return {
        "root":
            str(PRIMARY_ROOT),

        "comparisons":
            comparisons,

        "exact_historical_test_match_folds":
            exact_test_matches,

        "exact_historical_outer_partition_match_folds":
            exact_outer_matches,

        "historical_split_counts": {
            "train":
                len(
                    HISTORICAL_TRAIN
                ),

            "validation":
                len(
                    HISTORICAL_VALIDATION
                ),

            "test":
                len(
                    HISTORICAL_TEST
                ),
        },

        "fivefold_fold1_counts": {
            "train":
                target[
                    "folds"
                ][0][
                    "train_subject_count"
                ],

            "validation":
                target[
                    "folds"
                ][0][
                    "validation_subject_count"
                ],

            "test":
                target[
                    "folds"
                ][0][
                    "test_subject_count"
                ],
        },
    }


def targeted_run_references() -> list[dict[str, Any]]:
    records = []

    seen = set()

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "data",
        "datasets",
        "checkpoints",
        "artifacts",
        "results",
        "outputs",
    }

    for root in SEARCH_ROOTS:
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
                not in skip_dirs
            ]

            for filename in files:
                path = (
                    Path(current)
                    / filename
                )

                if (
                    path.suffix.lower()
                    not in TEXT_EXTENSIONS
                ):
                    continue

                try:
                    size = (
                        path.stat().st_size
                    )
                except Exception:
                    continue

                if size > 2_000_000:
                    continue

                try:
                    text = path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception:
                    continue

                if RUN_ID not in text:
                    continue

                resolved = str(
                    path.resolve()
                )

                if resolved in seen:
                    continue

                seen.add(resolved)

                lines = text.splitlines()

                excerpts = []

                for number, line in enumerate(
                    lines,
                    start=1,
                ):
                    if RUN_ID not in line:
                        continue

                    start = max(
                        1,
                        number - 8,
                    )

                    end = min(
                        len(lines),
                        number + 12,
                    )

                    excerpts.append(
                        {
                            "match_line":
                                number,

                            "context":
                                "\n".join(
                                    (
                                        f"{index}: "
                                        f"{lines[index - 1]}"
                                    )
                                    for index
                                    in range(
                                        start,
                                        end + 1,
                                    )
                                ),
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

                        "excerpts":
                            excerpts[:20],
                    }
                )

    return records


def locate_risk_assets() -> dict[str, list[str]]:
    found = {
        "subject_fold_json":
            [],

        "event_csv":
            [],
    }

    for root in RISK_ROOTS:
        if not root.is_dir():
            continue

        for filename in (
            SUBJECT_FOLD_FILENAMES
        ):
            for path in root.rglob(
                filename
            ):
                found[
                    "subject_fold_json"
                ].append(
                    str(
                        path.resolve()
                    )
                )

        for filename in (
            EVENT_FILENAMES
        ):
            for path in root.rglob(
                filename
            ):
                found[
                    "event_csv"
                ].append(
                    str(
                        path.resolve()
                    )
                )

    for key in found:
        found[key] = sorted(
            set(
                found[key]
            )
        )

    return found


def collect_strings(
    obj: Any,
) -> list[str]:
    result = []

    if isinstance(
        obj,
        dict,
    ):
        for key, value in (
            obj.items()
        ):
            result.append(
                str(key)
            )

            result.extend(
                collect_strings(
                    value
                )
            )

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            result.extend(
                collect_strings(
                    value
                )
            )

    elif isinstance(
        obj,
        (
            str,
            int,
            float,
        ),
    ):
        result.append(
            str(obj)
        )

    return result


def audit_subject_fold_json(
    path: Path,
) -> dict[str, Any]:
    result = {
        "path":
            str(path),

        "sha256":
            sha256_file(
                path
            ),
    }

    try:
        data = read_json(
            path
        )

        strings = collect_strings(
            data
        )

        canonical = sorted(
            {
                value
                for value
                in strings
                if re.fullmatch(
                    r"(?:KFALL|UNIVR)[_-]?\d+",
                    value,
                    flags=re.IGNORECASE,
                )
            }
        )

        kfall = [
            value
            for value
            in canonical
            if value.upper().startswith(
                "KFALL"
            )
        ]

        univr = [
            value
            for value
            in canonical
            if value.upper().startswith(
                "UNIVR"
            )
        ]

        result.update(
            {
                "parse_status":
                    "PASS",

                "top_level_type":
                    type(
                        data
                    ).__name__,

                "top_level_keys":
                    (
                        sorted(
                            str(key)
                            for key
                            in data
                        )
                        if isinstance(
                            data,
                            dict,
                        )
                        else None
                    ),

                "canonical_subject_id_count":
                    len(
                        canonical
                    ),

                "canonical_kfall_subject_count":
                    len(
                        kfall
                    ),

                "canonical_univr_subject_count":
                    len(
                        univr
                    ),

                "canonical_subject_ids":
                    canonical,
            }
        )

    except Exception as exc:
        result[
            "parse_status"
        ] = "FAILED"

        result[
            "parse_error"
        ] = repr(exc)

    return result


def detect_dataset(
    row: dict[str, str],
) -> str:
    text = " ".join(
        str(value)
        for value
        in row.values()
    ).upper()

    if "KFALL" in text:
        return "KFALL"

    if "UNIVR" in text:
        return "UNIVR"

    return "UNRESOLVED"


def audit_event_csv(
    path: Path,
) -> dict[str, Any]:
    result = {
        "path":
            str(path),

        "sha256":
            sha256_file(
                path
            ),
    }

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            rows = list(
                reader
            )

            columns = (
                reader.fieldnames
                or []
            )

        normalized_columns = {
            column:
                re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    column.lower(),
                ).strip("_")
            for column
            in columns
        }

        dataset_counts = Counter(
            detect_dataset(
                row
            )
            for row
            in rows
        )

        exact_row_keys = [
            tuple(
                row.get(
                    column,
                    ""
                )
                for column
                in columns
            )
            for row
            in rows
        ]

        duplicate_exact_rows = (
            len(
                exact_row_keys
            )
            - len(
                set(
                    exact_row_keys
                )
            )
        )

        id_candidates = [
            column
            for column
            in columns
            if any(
                token
                in normalized_columns[
                    column
                ]
                for token
                in (
                    "event",
                    "trial",
                    "record",
                    "file",
                    "sample",
                    "subject",
                )
            )
        ]

        duplicate_candidate_counts = {}

        for column in id_candidates:
            values = [
                row.get(
                    column,
                    ""
                ).strip()
                for row
                in rows
            ]

            populated = [
                value
                for value
                in values
                if value
            ]

            duplicate_candidate_counts[
                column
            ] = (
                len(
                    populated
                )
                - len(
                    set(
                        populated
                    )
                )
            )

        timing_columns = [
            column
            for column
            in columns
            if any(
                token
                in normalized_columns[
                    column
                ]
                for token
                in (
                    "onset",
                    "impact",
                    "frame",
                    "time",
                    "timestamp",
                )
            )
        ]

        missing_by_timing_column = {}

        for column in timing_columns:
            missing_by_timing_column[
                column
            ] = sum(
                1
                for row
                in rows
                if not (
                    row.get(
                        column,
                        ""
                    )
                    or ""
                ).strip()
            )

        result.update(
            {
                "parse_status":
                    "PASS",

                "row_count":
                    len(
                        rows
                    ),

                "columns":
                    columns,

                "dataset_counts":
                    dict(
                        dataset_counts
                    ),

                "exact_duplicate_row_count":
                    duplicate_exact_rows,

                "candidate_identifier_columns":
                    id_candidates,

                "duplicate_counts_by_candidate_identifier":
                    duplicate_candidate_counts,

                "timing_columns":
                    timing_columns,

                "missing_counts_by_timing_column":
                    missing_by_timing_column,

                "first_rows":
                    rows[:5],
            }
        )

    except Exception as exc:
        result[
            "parse_status"
        ] = "FAILED"

        result[
            "parse_error"
        ] = repr(exc)

    return result


def main() -> None:
    phase3b = read_json(
        PHASE3B
    )

    phase3f = read_json(
        PHASE3F
    )

    assert (
        phase3b["status"]
        == "PASS"
    )

    assert (
        phase3f["status"]
        == "PASS"
    )

    if (
        len(
            HISTORICAL_TRAIN
        )
        != 47
    ):
        raise RuntimeError(
            "Historical train subject count mismatch"
        )

    if (
        len(
            HISTORICAL_VALIDATION
        )
        != 6
    ):
        raise RuntimeError(
            "Historical validation subject count mismatch"
        )

    if (
        len(
            HISTORICAL_TEST
        )
        != 14
    ):
        raise RuntimeError(
            "Historical test subject count mismatch"
        )

    if (
        HISTORICAL_TRAIN
        & HISTORICAL_VALIDATION
    ):
        raise RuntimeError(
            "Historical train/validation overlap"
        )

    if (
        HISTORICAL_TRAIN
        & HISTORICAL_TEST
    ):
        raise RuntimeError(
            "Historical train/test overlap"
        )

    if (
        HISTORICAL_VALIDATION
        & HISTORICAL_TEST
    ):
        raise RuntimeError(
            "Historical validation/test overlap"
        )

    root_binding = (
        reconstruct_primary_subject_set()
    )

    fold_binding = (
        compare_phase3f_folds()
    )

    checkpoint = (
        simple_checkpoint_metadata(
            PRIMARY_CHECKPOINT
        )
    )

    if not checkpoint[
        "exists"
    ]:
        raise RuntimeError(
            "Primary historical checkpoint missing"
        )

    if not checkpoint[
        "sha256_verified"
    ]:
        raise RuntimeError(
            "Primary historical checkpoint SHA changed"
        )

    references = (
        targeted_run_references()
    )

    assets = (
        locate_risk_assets()
    )

    fold_assets = [
        audit_subject_fold_json(
            Path(path)
        )
        for path
        in assets[
            "subject_fold_json"
        ]
    ]

    event_assets = [
        audit_event_csv(
            Path(path)
        )
        for path
        in assets[
            "event_csv"
        ]
    ]

    if not root_binding[
        "subject_set_equals_historical"
    ]:
        raise RuntimeError(
            "67-subject candidate root does not equal "
            "historical population"
        )

    if (
        fold_binding[
            "exact_historical_test_match_folds"
        ]
        != [1]
    ):
        raise RuntimeError(
            "Expected unique historical outer-test match at fold 1"
        )

    if (
        fold_binding[
            "exact_historical_outer_partition_match_folds"
        ]
        != [1]
    ):
        raise RuntimeError(
            "Historical outer non-test partition did not match fold 1"
        )

    fold1 = (
        fold_binding[
            "comparisons"
        ][0]
    )

    if fold1[
        "historical_validation_equals_fold_validation"
    ]:
        raise RuntimeError(
            "Historical validation unexpectedly equals "
            "prospective five-fold validation"
        )

    manifest = {
        "schema":
            "crosslayer_phase3g_primary_root_split_event_lineage_v1",

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

        "primary_baseline": {
            "identifier":
                "DATE2025_CNN_400MS_RECONSTRUCTED",

            "checkpoint":
                checkpoint,

            "window_ms":
                400,
        },

        "historical_split": {
            "train_subjects":
                sorted(
                    HISTORICAL_TRAIN
                ),

            "validation_subjects":
                sorted(
                    HISTORICAL_VALIDATION
                ),

            "test_subjects":
                sorted(
                    HISTORICAL_TEST
                ),

            "augmentation_only_subjects":
                sorted(
                    AUGMENTATION_ONLY
                ),

            "train_count":
                len(
                    HISTORICAL_TRAIN
                ),

            "validation_count":
                len(
                    HISTORICAL_VALIDATION
                ),

            "test_count":
                len(
                    HISTORICAL_TEST
                ),
        },

        "primary_root_binding":
            root_binding,

        "fivefold_binding":
            fold_binding,

        "historical_run_references":
            references,

        "risk_asset_locations":
            assets,

        "subject_fold_assets":
            fold_assets,

        "event_index_assets":
            event_assets,

        "protocol_interpretation": {
            "authoritative_real_subject_root_candidate":
                str(
                    PRIMARY_ROOT
                ),

            "authoritative_root_subject_population_status":
                "EXACT_MATCH",

            "historical_outer_test_status":
                "EXACT_MATCH_TO_RECONSTRUCTED_FOLD_1",

            "historical_outer_non_test_status":
                "EXACT_MATCH_TO_RECONSTRUCTED_FOLD_1",

            "historical_inner_validation_status":
                "DISTINCT_FROM_CURRENT_FIVEFOLD_INNER_VALIDATION",

            "historical_checkpoint_split_role":
                "FROZEN_BASELINE_PROVENANCE",

            "prospective_fivefold_role":
                "NEW_FOLD_SPECIFIC_REPLICATION_PROTOCOL",

            "reuse_single_checkpoint_across_folds":
                False,

            "event_timing_status":
                "NOT_YET_FROZEN",
        },

        "scientific_boundary": {
            "model_retrained":
                False,

            "checkpoint_modified":
                False,

            "split_selected_using_performance":
                False,

            "held_out_predictions_opened":
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

    print(
        "Primary root:",
        root_binding[
            "path"
        ],
    )

    print(
        "Raw subjects:",
        root_binding[
            "raw_subject_count"
        ],
    )

    print(
        "Real subjects:",
        root_binding[
            "real_subject_count"
        ],
    )

    print(
        "Subject set exact historical match:",
        root_binding[
            "subject_set_equals_historical"
        ],
    )

    print(
        "Exact historical test-match folds:",
        fold_binding[
            "exact_historical_test_match_folds"
        ],
    )

    print(
        "Exact historical outer-partition folds:",
        fold_binding[
            "exact_historical_outer_partition_match_folds"
        ],
    )

    print(
        "Historical inner validation equals fold-1 validation:",
        fold1[
            "historical_validation_equals_fold_validation"
        ],
    )

    print(
        "Historical counts:",
        fold_binding[
            "historical_split_counts"
        ],
    )

    print(
        "Prospective fold-1 counts:",
        fold_binding[
            "fivefold_fold1_counts"
        ],
    )

    print(
        "Primary checkpoint SHA verified:",
        checkpoint[
            "sha256_verified"
        ],
    )

    print(
        "Historical run references:",
        len(
            references
        ),
    )

    print(
        "Subject-fold assets:",
        len(
            fold_assets
        ),
    )

    print(
        "Event-index assets:",
        len(
            event_assets
        ),
    )

    print()
    print(
        "PHASE_3G_PRIMARY_ROOT_SPLIT_EVENT_LINEAGE=PASS"
    )


if __name__ == "__main__":
    main()
