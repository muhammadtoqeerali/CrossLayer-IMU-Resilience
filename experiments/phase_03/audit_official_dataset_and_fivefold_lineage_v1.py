from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
TOQEER = HOME / "toqeer"

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_3c_official_dataset_and_fivefold_lineage_v1.json"
)

HISTORICAL_SPLIT = (
    ROOT
    / "manifests"
    / "phase_3b_historical_split_recovery_v1.json"
)

# Candidate UniVR trees.
UNIVR_CANDIDATES = [
    (
        "legacy_original",
        Path(
            "/mnt/hdd16T/protechto/"
            "UniVrFallOriginalDataset"
        ),
    ),
    (
        "historical_oriented",
        Path(
            "/mnt/hdd16T/protechto/"
            "UniVrFall_oriented"
        ),
    ),
    (
        "user_univr_dataset",
        TOQEER / "uniVr-dataset",
    ),
    (
        "user_UniVrFall_Dataset",
        TOQEER / "UniVrFall_Dataset",
    ),
]

KFALL_ROOT = Path(
    "/mnt/hdd16T/protechto/"
    "ThirdPartyDatasets/KFall"
)

SEARCH_REPOS = [
    TOQEER / "IMU_Reliability",
    TOQEER / "RC-RGD-IMU_publish",
    TOQEER / "Protechto-master",
    TOQEER / "Protechto_master",
    TOQEER / "Protechto-repo",
    TOQEER / "Protechto",
    TOQEER / "fall_project_code",
    TOQEER / "HR_LR_Fallings",
]

FOLD_FILENAME_RE = re.compile(
    r"(fold|cross.?val|split|kfold|cv)",
    re.IGNORECASE,
)

FOLD_CONTENT_TERMS = (
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "StratifiedGroupKFold",
    "n_splits=5",
    "n_splits = 5",
    "5-fold",
    "5 fold",
    "five-fold",
    "train_subject",
    "validation_subject",
    "test_subject",
    "fold_",
)

TRIAL_RE = re.compile(
    r"^S(?P<subject>\d+)T(?P<task>\d+)R(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)

LABEL_RE = re.compile(
    r"^SA?(?P<subject>\d+)_label\.xlsx$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sample_text(path: Path, limit: int = 32768) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )[:limit]
    except Exception:
        return ""


def robust_csv_header(path: Path) -> dict[str, Any]:
    """
    Search the first several physical lines for known sensor-column names
    instead of assuming line 1 is the header.
    """
    try:
        text = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )
    except Exception as exc:
        return {
            "readable": False,
            "error": repr(exc),
        }

    lines = text.splitlines()[:25]

    known = (
        "timestamp",
        "framecounter",
        "accx",
        "accy",
        "accz",
        "gyrx",
        "gyry",
        "gyrz",
    )

    candidates = []

    for index, line in enumerate(lines):
        for delimiter in (",", ";", "\t"):
            values = [
                value.strip()
                for value in line.split(delimiter)
            ]

            normalized = [
                re.sub(
                    r"[^a-z0-9]+",
                    "",
                    value.lower(),
                )
                for value in values
            ]

            score = sum(
                any(
                    token in value
                    for value in normalized
                )
                for token in known
            )

            if score:
                candidates.append(
                    {
                        "line_index":
                            index,

                        "delimiter":
                            delimiter,

                        "columns":
                            values,

                        "normalized":
                            normalized,

                        "score":
                            score,
                    }
                )

    if not candidates:
        return {
            "readable":
                True,

            "header_found":
                False,

            "first_lines":
                lines[:5],
        }

    best = max(
        candidates,
        key=lambda item: (
            item["score"],
            len(item["columns"]),
        ),
    )

    normalized = set(
        best["normalized"]
    )

    has_timestamp = any(
        (
            "timestamp"
            in value
            or value.startswith("time")
        )
        for value in normalized
    )

    has_counter = any(
        "framecounter" in value
        for value in normalized
    )

    return {
        "readable":
            True,

        "header_found":
            True,

        "header_line_index":
            best["line_index"],

        "delimiter":
            best["delimiter"],

        "columns":
            best["columns"],

        "has_timestamp":
            has_timestamp,

        "has_frame_counter":
            has_counter,
    }


def inspect_csv_tree(root: Path) -> dict[str, Any]:
    result = {
        "path":
            str(root),

        "exists":
            root.is_dir(),
    }

    if not root.is_dir():
        return result

    csvs = sorted(
        root.rglob("*.csv")
    )

    trial_files = []
    subjects = set()

    timestamp_count = 0
    counter_count = 0
    header_found_count = 0

    header_examples = []
    no_header_examples = []

    for path in csvs:
        match = TRIAL_RE.match(
            path.name
        )

        if match:
            subjects.add(
                int(
                    match.group("subject")
                )
            )

            trial_files.append(
                path
            )

        info = robust_csv_header(
            path
        )

        if info.get(
            "header_found"
        ):
            header_found_count += 1

            if info.get(
                "has_timestamp"
            ):
                timestamp_count += 1

            if info.get(
                "has_frame_counter"
            ):
                counter_count += 1

            if len(
                header_examples
            ) < 10:
                header_examples.append(
                    {
                        "file":
                            str(path),

                        "header_line_index":
                            info[
                                "header_line_index"
                            ],

                        "columns":
                            info[
                                "columns"
                            ],
                    }
                )

        elif len(
            no_header_examples
        ) < 10:
            no_header_examples.append(
                {
                    "file":
                        str(path),

                    "first_lines":
                        info.get(
                            "first_lines",
                            [],
                        ),
                }
            )

    return {
        **result,

        "csv_count":
            len(csvs),

        "trial_filename_count":
            len(trial_files),

        "subjects":
            sorted(
                subjects
            ),

        "subject_count":
            len(subjects),

        "header_found_count":
            header_found_count,

        "timestamp_header_count":
            timestamp_count,

        "frame_counter_header_count":
            counter_count,

        "header_examples":
            header_examples,

        "unparsed_header_examples":
            no_header_examples,
    }


def inspect_excel_annotations(
    root: Path,
) -> dict[str, Any]:
    files = sorted(
        root.rglob("*.xlsx")
    ) if root.is_dir() else []

    result = {
        "root":
            str(root),

        "xlsx_count":
            len(files),

        "subjects":
            [],

        "subject_count":
            0,

        "annotation_rows":
            0,

        "rows_with_onset":
            0,

        "rows_with_impact":
            0,

        "rows_with_both":
            0,

        "column_signatures":
            {},

        "examples":
            [],
    }

    if not files:
        return result

    try:
        import pandas as pd
    except Exception as exc:
        result[
            "pandas_error"
        ] = repr(exc)

        return result

    subjects = set()
    signatures = Counter()

    for path in files:
        match = LABEL_RE.match(
            path.name
        )

        if match:
            subjects.add(
                int(
                    match.group(
                        "subject"
                    )
                )
            )

        try:
            frame = pd.read_excel(
                path
            )
        except Exception:
            continue

        columns = [
            str(column)
            for column
            in frame.columns
        ]

        signatures[
            "|".join(columns)
        ] += 1

        normalized = {
            re.sub(
                r"[^a-z0-9]+",
                "",
                str(column).lower(),
            ):
                column
            for column
            in frame.columns
        }

        onset_column = None
        impact_column = None

        for key, original in (
            normalized.items()
        ):
            if (
                "fallonset" in key
                or "startfall" in key
                or "onsetframe" in key
            ):
                onset_column = original

            if (
                "fallimpact" in key
                or "endfall" in key
                or "impactframe" in key
            ):
                impact_column = original

        result[
            "annotation_rows"
        ] += int(
            len(frame)
        )

        if onset_column is not None:
            onset = frame[
                onset_column
            ].notna()

            result[
                "rows_with_onset"
            ] += int(
                onset.sum()
            )
        else:
            onset = None

        if impact_column is not None:
            impact = frame[
                impact_column
            ].notna()

            result[
                "rows_with_impact"
            ] += int(
                impact.sum()
            )
        else:
            impact = None

        if (
            onset is not None
            and impact is not None
        ):
            result[
                "rows_with_both"
            ] += int(
                (
                    onset
                    & impact
                ).sum()
            )

        if len(
            result["examples"]
        ) < 10:
            result[
                "examples"
            ].append(
                {
                    "file":
                        str(path),

                    "rows":
                        int(
                            len(frame)
                        ),

                    "columns":
                        columns,

                    "onset_column":
                        (
                            str(onset_column)
                            if onset_column
                            is not None
                            else None
                        ),

                    "impact_column":
                        (
                            str(impact_column)
                            if impact_column
                            is not None
                            else None
                        ),
                }
            )

    result[
        "subjects"
    ] = sorted(
        subjects
    )

    result[
        "subject_count"
    ] = len(
        subjects
    )

    result[
        "column_signatures"
    ] = dict(
        signatures
    )

    return result


def find_fivefold_sources() -> list[dict[str, Any]]:
    results = []

    allowed = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".csv",
    }

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "data",
        "datasets",
        "artifacts",
        "checkpoints",
        "results",
        "outputs",
        "runs",
    }

    seen = set()

    for repo in SEARCH_REPOS:
        if not repo.is_dir():
            continue

        for current, dirs, files in os.walk(
            repo
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
                    not in allowed
                ):
                    continue

                try:
                    relative = path.relative_to(
                        repo
                    ).as_posix()

                    size = path.stat().st_size

                except Exception:
                    continue

                if size > 2_000_000:
                    continue

                text = sample_text(
                    path,
                    limit=200_000,
                )

                name_hit = bool(
                    FOLD_FILENAME_RE.search(
                        filename
                    )
                )

                content_hits = [
                    term
                    for term
                    in FOLD_CONTENT_TERMS
                    if term
                    in text
                ]

                if not (
                    name_hit
                    or content_hits
                ):
                    continue

                key = (
                    str(repo),
                    relative,
                )

                if key in seen:
                    continue

                seen.add(key)

                line_hits = []

                for number, line in enumerate(
                    text.splitlines(),
                    start=1,
                ):
                    if any(
                        term in line
                        for term
                        in FOLD_CONTENT_TERMS
                    ):
                        line_hits.append(
                            {
                                "line":
                                    number,

                                "text":
                                    line.strip()[:500],
                            }
                        )

                    if len(
                        line_hits
                    ) >= 30:
                        break

                results.append(
                    {
                        "repository":
                            str(repo),

                        "relative_path":
                            relative,

                        "sha256":
                            sha256_file(
                                path
                            ),

                        "filename_match":
                            name_hit,

                        "content_terms":
                            content_hits,

                        "line_hits":
                            line_hits,
                    }
                )

    return results


def audit_explicit_fold_json(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audits = []

    for item in sources:
        if not item[
            "relative_path"
        ].lower().endswith(
            ".json"
        ):
            continue

        path = (
            Path(
                item["repository"]
            )
            / item[
                "relative_path"
            ]
        )

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        text = json.dumps(
            data
        ).lower()

        if "fold" not in text:
            continue

        audit = {
            "path":
                str(path),

            "sha256":
                item["sha256"],

            "top_level_type":
                type(
                    data
                ).__name__,

            "top_level_keys":
                (
                    sorted(
                        data.keys()
                    )
                    if isinstance(
                        data,
                        dict,
                    )
                    else None
                ),
        }

        # Generic recursive extraction of train/val/test subject lists.
        fold_records = []

        def recurse(
            node,
            location: str,
        ):
            if isinstance(
                node,
                dict,
            ):
                lowered = {
                    str(key).lower():
                        value
                    for key, value
                    in node.items()
                }

                train_key = next(
                    (
                        key
                        for key
                        in lowered
                        if "train" in key
                        and "subject" in key
                    ),
                    None,
                )

                val_key = next(
                    (
                        key
                        for key
                        in lowered
                        if (
                            "val" in key
                            or "validation"
                            in key
                        )
                        and "subject"
                        in key
                    ),
                    None,
                )

                test_key = next(
                    (
                        key
                        for key
                        in lowered
                        if "test" in key
                        and "subject" in key
                    ),
                    None,
                )

                if (
                    train_key
                    and val_key
                    and test_key
                    and isinstance(
                        lowered[
                            train_key
                        ],
                        list,
                    )
                    and isinstance(
                        lowered[
                            val_key
                        ],
                        list,
                    )
                    and isinstance(
                        lowered[
                            test_key
                        ],
                        list,
                    )
                ):
                    train = {
                        str(value)
                        for value
                        in lowered[
                            train_key
                        ]
                    }

                    validation = {
                        str(value)
                        for value
                        in lowered[
                            val_key
                        ]
                    }

                    test = {
                        str(value)
                        for value
                        in lowered[
                            test_key
                        ]
                    }

                    fold_records.append(
                        {
                            "location":
                                location,

                            "train_count":
                                len(train),

                            "validation_count":
                                len(
                                    validation
                                ),

                            "test_count":
                                len(test),

                            "train_validation_overlap":
                                sorted(
                                    train
                                    & validation
                                ),

                            "train_test_overlap":
                                sorted(
                                    train
                                    & test
                                ),

                            "validation_test_overlap":
                                sorted(
                                    validation
                                    & test
                                ),

                            "train_subjects":
                                sorted(
                                    train
                                ),

                            "validation_subjects":
                                sorted(
                                    validation
                                ),

                            "test_subjects":
                                sorted(
                                    test
                                ),
                        }
                    )

                for key, value in (
                    node.items()
                ):
                    recurse(
                        value,
                        (
                            f"{location}."
                            f"{key}"
                        ),
                    )

            elif isinstance(
                node,
                list,
            ):
                for index, value in enumerate(
                    node
                ):
                    recurse(
                        value,
                        (
                            f"{location}"
                            f"[{index}]"
                        ),
                    )

        recurse(
            data,
            "$",
        )

        audit[
            "detected_fold_records"
        ] = fold_records

        audits.append(
            audit
        )

    return audits


def main() -> int:
    historical = json.loads(
        HISTORICAL_SPLIT.read_text(
            encoding="utf-8"
        )
    )

    if historical[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Phase-3B historical split did not pass"
        )

    print(
        "Historical protected split remains:"
    )

    print(
        "  train subjects:",
        len(
            historical[
                "subjects"
            ]["train"]
        ),
    )

    print(
        "  validation subjects:",
        len(
            historical[
                "subjects"
            ]["validation"]
        ),
    )

    print(
        "  test subjects:",
        len(
            historical[
                "subjects"
            ]["test"]
        ),
    )

    print()

    univr_trees = []

    for name, root in (
        UNIVR_CANDIDATES
    ):
        print(
            "AUDIT UNIVR TREE:",
            name,
            root,
        )

        csv_audit = inspect_csv_tree(
            root
        )

        xlsx_audit = (
            inspect_excel_annotations(
                root
            )
        )

        record = {
            "name":
                name,

            "root":
                str(root),

            "csv":
                csv_audit,

            "xlsx":
                xlsx_audit,
        }

        univr_trees.append(
            record
        )

        print(
            "  exists:",
            root.is_dir(),
        )

        if root.is_dir():
            print(
                "  CSVs:",
                csv_audit.get(
                    "csv_count",
                    0,
                ),
            )

            print(
                "  subjects from trial filenames:",
                csv_audit.get(
                    "subject_count",
                    0,
                ),
            )

            print(
                "  timestamp headers:",
                csv_audit.get(
                    "timestamp_header_count",
                    0,
                ),
            )

            print(
                "  FrameCounter headers:",
                csv_audit.get(
                    "frame_counter_header_count",
                    0,
                ),
            )

            print(
                "  annotation XLSX:",
                xlsx_audit.get(
                    "xlsx_count",
                    0,
                ),
            )

            print(
                "  annotation subjects:",
                xlsx_audit.get(
                    "subject_count",
                    0,
                ),
            )

            print(
                "  rows with onset+impact:",
                xlsx_audit.get(
                    "rows_with_both",
                    0,
                ),
            )

        print()

    print(
        "AUDIT KFALL:",
        KFALL_ROOT,
    )

    kfall_csv = inspect_csv_tree(
        KFALL_ROOT
    )

    kfall_xlsx = (
        inspect_excel_annotations(
            KFALL_ROOT
        )
    )

    print(
        "  exists:",
        KFALL_ROOT.is_dir(),
    )

    print(
        "  CSVs:",
        kfall_csv.get(
            "csv_count",
            0,
        ),
    )

    print(
        "  subjects:",
        kfall_csv.get(
            "subject_count",
            0,
        ),
    )

    print(
        "  timestamp headers:",
        kfall_csv.get(
            "timestamp_header_count",
            0,
        ),
    )

    print(
        "  FrameCounter headers:",
        kfall_csv.get(
            "frame_counter_header_count",
            0,
        ),
    )

    print(
        "  annotation XLSX:",
        kfall_xlsx.get(
            "xlsx_count",
            0,
        ),
    )

    print(
        "  rows with onset+impact:",
        kfall_xlsx.get(
            "rows_with_both",
            0,
        ),
    )

    print()
    print(
        "SEARCHING FOR 5-FOLD IMPLEMENTATION..."
    )

    fold_sources = (
        find_fivefold_sources()
    )

    fold_json_audits = (
        audit_explicit_fold_json(
            fold_sources
        )
    )

    print(
        "Candidate fold-related source files:",
        len(
            fold_sources
        ),
    )

    for item in fold_sources[
        :100
    ]:
        print(
            " ",
            item["repository"],
            "::",
            item[
                "relative_path"
            ],
        )

        for hit in item[
            "line_hits"
        ][:5]:
            print(
                f"    L{hit['line']}: "
                f"{hit['text']}"
            )

    print()
    print(
        "JSON files with detectable train/validation/test subject records:",
        sum(
            len(
                item[
                    "detected_fold_records"
                ]
            )
            for item
            in fold_json_audits
        ),
    )

    # Determine whether any explicit detected fold record leaks subjects.
    detected_records = []

    for item in fold_json_audits:
        for fold in item[
            "detected_fold_records"
        ]:
            record = {
                "source":
                    item[
                        "path"
                    ],

                **fold,
            }

            detected_records.append(
                record
            )

            if (
                fold[
                    "train_validation_overlap"
                ]
                or fold[
                    "train_test_overlap"
                ]
                or fold[
                    "validation_test_overlap"
                ]
            ):
                raise RuntimeError(
                    "Subject overlap detected in explicit five-fold record: "
                    f"{record}"
                )

    # Public specification recorded from official sources.
    public_spec = {
        "univrfall": {
            "laboratory_participants":
                29,

            "construction_workers":
                10,

            "total_participants":
                39,

            "sampling_hz":
                100,

            "fall_types":
                21,

            "fall_events":
                573,

            "raw_sensor_columns":
                11,

            "raw_sensor_expected_fields": [
                "TimeStamp(s)",
                "FrameCounter",
                "AccX",
                "AccY",
                "AccZ",
                "GyrX",
                "GyrY",
                "GyrZ",
                "EulerX",
                "EulerY",
                "EulerZ",
            ],

            "annotation_expected_fields": [
                "Task Code",
                "Description",
                "Trial ID",
                "Fall onset frame",
                "Fall impact frame",
            ],

            "subject_independent_fivefold_documented":
                True,

            "source":
                "Zenodo 10.5281/zenodo.18346755",
        },

        "kfall": {
            "participants":
                32,

            "sampling_hz":
                100,

            "motion_files":
                5075,

            "adl_motion_files":
                2729,

            "fall_motion_files":
                2346,

            "adl_types":
                21,

            "fall_types":
                15,

            "fall_onset_and_impact_annotations":
                True,

            "source":
                "KFall original dataset documentation",
        },
    }

    manifest = {
        "schema":
            "crosslayer_phase3c_official_dataset_and_fivefold_lineage_v1",

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

        "public_dataset_specification":
            public_spec,

        "local_univr_candidate_trees":
            univr_trees,

        "local_kfall": {
            "root":
                str(
                    KFALL_ROOT
                ),

            "csv":
                kfall_csv,

            "xlsx":
                kfall_xlsx,
        },

        "fivefold_source_candidates":
            fold_sources,

        "explicit_fold_json_audits":
            fold_json_audits,

        "detected_explicit_fold_records":
            detected_records,

        "historical_checkpoint_split": {
            "status":
                "RETAINS_PROVENANCE_ROLE",

            "train_subject_count":
                len(
                    historical[
                        "subjects"
                    ]["train"]
                ),

            "validation_subject_count":
                len(
                    historical[
                        "subjects"
                    ]["validation"]
                ),

            "test_subject_count":
                len(
                    historical[
                        "subjects"
                    ]["test"]
                ),

            "reason":
                (
                    "The frozen historical checkpoint was trained under "
                    "this lineage. It cannot be post-hoc evaluated as if "
                    "trained under a different fold."
                ),
        },

        "fivefold_role_candidate": {
            "status":
                "AUDIT_BEFORE_FREEZE",

            "intended_role":
                (
                    "subject-independent repeated training/evaluation "
                    "with independent model fitting inside each fold"
                ),

            "single_frozen_checkpoint_may_be_reused_across_all_folds":
                False,

            "reason":
                (
                    "Cross-validation requires fold-specific training. "
                    "A single historical checkpoint cannot become five "
                    "independent trained models after the fact."
                ),
        },

        "timing_correction": {
            "univr_public_release_has_event_anchors":
                True,

            "kfall_has_event_anchors":
                True,

            "univr_event_anchor":
                "fall_onset_frame_and_fall_impact_frame",

            "kfall_event_anchor":
                "fall_onset_frame_and_fall_impact_frame",

            "physical_preimpact_lead_time_status":
                (
                    "POTENTIALLY_QUALIFIABLE_AFTER_LOCAL "
                    "ANNOTATION_TO_PROTECTED_TRIAL_MAPPING"
                ),

            "previous_blanket_not_qualified_statement":
                "REVISED",
        },

        "scientific_boundary": {
            "historical_split_changed":
                False,

            "fivefold_membership_changed":
                False,

            "new_fold_generated":
                False,

            "model_predictions_opened":
                False,

            "held_out_results_used":
                False,

            "int8_calibration_performed":
                False,

            "faults_injected":
                False,
        },

        "next_actions": [
            (
                "Identify the authoritative local five-fold implementation "
                "and freeze its exact subject memberships."
            ),
            (
                "Map UniVR annotation onset/impact frames to protected "
                "historical trials."
            ),
            (
                "Map KFall onset/impact frames to protected historical trials."
            ),
            (
                "Verify exact historical 400-ms, 50-percent-overlap "
                "windowing implementation."
            ),
            (
                "Only then freeze physical pre-impact lead-time semantics."
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
        "PHASE_3C_REPAIRED_AUDIT=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
