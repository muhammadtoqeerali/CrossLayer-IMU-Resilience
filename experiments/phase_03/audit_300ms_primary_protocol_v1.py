from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOQEER = Path.home() / "toqeer"

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3_300ms_primary_protocol_audit_v1.json"
)

PROJECT_CANDIDATES = [
    TOQEER / "Protechto-master",
    TOQEER / "Protechto_master",
    TOQEER / "Protechto-master_ori",
    TOQEER / "HR_LR_Fallings",
]

RAW_ORIENTED_ROOTS = {
    "UNIVR": [
        Path(
            "/mnt/hdd16T/protechto/"
            "UniVrFall_oriented"
        ),
        Path(
            "/home/shared/data/protechto/"
            "UniVrFall_oriented"
        ),
    ],

    "KFALL": [
        Path(
            "/mnt/hdd16T/protechto/data/"
            "KFall_oriented"
        ),
        Path(
            "/mnt/hdd16T/protechto/"
            "ThirdPartyDatasets/"
            "KFall_oriented"
        ),
        Path(
            "/home/shared/data/protechto/"
            "KFall_oriented"
        ),
    ],

    "ONFIELD": [
        Path(
            "/mnt/hdd16T/protechto/"
            "ThirdPartyDatasets/"
            "OnFieldRecordings"
        ),
        Path(
            "/home/shared/data/protechto/"
            "OnFieldRecordings"
        ),
    ],
}

SEGMENT_SEARCH_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto"
    ),
    TOQEER,
]

CHECKPOINT_SEARCH_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto/checkpoints"
    ),
]

WINDOWING_MODULE_NAMES = (
    "preprocessing/windowing.py",
    "preprocessing/helper.py",
)

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".toml",
    ".sh",
}

CKPT_RE = re.compile(
    r"best-checkpoint-fold_(?P<fold>[1-5])\.ckpt$"
)


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


def read_text(
    path: Path,
) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def inspect_windowing_sources() -> list[dict[str, Any]]:
    records = []

    seen = set()

    for project in PROJECT_CANDIDATES:
        if not project.is_dir():
            continue

        for relative in WINDOWING_MODULE_NAMES:
            path = project / relative

            if not path.is_file():
                continue

            resolved = str(
                path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(resolved)

            text = read_text(path)

            relevant = []

            lines = text.splitlines()

            terms = (
                "argparse",
                "add_argument",
                "overlap",
                "window",
                "step",
                "stride",
                "window_size",
                "samples_per_window",
                "50ov",
                "95",
            )

            for number, line in enumerate(
                lines,
                start=1,
            ):
                if any(
                    term.lower()
                    in line.lower()
                    for term
                    in terms
                ):
                    relevant.append(
                        {
                            "line":
                                number,

                            "text":
                                line.strip()[:700],
                        }
                    )

            records.append(
                {
                    "path":
                        resolved,

                    "sha256":
                        sha256_file(path),

                    "relevant_lines":
                        relevant[:200],
                }
            )

    return records


def discover_300ms_roots() -> list[dict[str, Any]]:
    records = []

    seen = set()

    for base in SEGMENT_SEARCH_ROOTS:
        if not base.is_dir():
            continue

        try:
            paths = base.rglob(
                "300ms_50ov_npseg_filt_binary"
            )
        except Exception:
            continue

        for path in paths:
            if not path.is_dir():
                continue

            resolved = str(
                path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(resolved)

            subjects = sorted(
                [
                    child.name
                    for child
                    in path.iterdir()
                    if (
                        child.is_dir()
                        and child.name.isdigit()
                    )
                ]
            )

            records.append(
                {
                    "path":
                        resolved,

                    "subject_count":
                        len(subjects),

                    "subjects":
                        subjects,

                    "contains_999":
                        "999" in subjects,

                    "contains_1000":
                        "1000" in subjects,
                }
            )

    return sorted(
        records,
        key=lambda item: item["path"],
    )


def discover_raw_subjects() -> dict[str, Any]:
    result = {}

    trial_pattern = re.compile(
        r"^S(?P<subject>\d+)"
        r"T\d+R\d+\.csv$",
        re.IGNORECASE,
    )

    for dataset, candidates in (
        RAW_ORIENTED_ROOTS.items()
    ):
        records = []

        for root in candidates:
            if not root.exists():
                continue

            subjects = set()

            for path in root.rglob(
                "*.csv"
            ):
                match = trial_pattern.match(
                    path.name
                )

                if match:
                    subjects.add(
                        match.group(
                            "subject"
                        )
                    )

            records.append(
                {
                    "path":
                        str(
                            root.resolve()
                        ),

                    "subject_count":
                        len(subjects),

                    "subjects":
                        sorted(
                            subjects,
                            key=lambda value:
                                int(value),
                        ),
                }
            )

        result[
            dataset
        ] = records

    return result


def discover_300ms_checkpoints() -> dict[str, Any]:
    files = []

    for base in CHECKPOINT_SEARCH_ROOTS:
        if not base.is_dir():
            continue

        for path in base.rglob(
            "*.ckpt"
        ):
            if "/300ms/" not in str(
                path
            ):
                continue

            match = CKPT_RE.match(
                path.name
            )

            if not match:
                continue

            parts = path.parts

            architecture = None

            try:
                index = parts.index(
                    "checkpoints"
                )

                architecture = (
                    parts[
                        index + 1
                    ]
                )

            except Exception:
                pass

            files.append(
                {
                    "path":
                        str(
                            path.resolve()
                        ),

                    "architecture":
                        architecture,

                    "run_dir":
                        str(
                            path.parent.resolve()
                        ),

                    "fold":
                        int(
                            match.group(
                                "fold"
                            )
                        ),

                    "bytes":
                        int(
                            path.stat().st_size
                        ),

                    "sha256":
                        sha256_file(
                            path
                        ),
                }
            )

    grouped = defaultdict(
        list
    )

    for item in files:
        grouped[
            (
                item["architecture"],
                item["run_dir"],
            )
        ].append(
            item
        )

    runs = []

    for (
        architecture,
        run_dir,
    ), values in grouped.items():

        folds = {
            item["fold"]
            for item
            in values
        }

        runs.append(
            {
                "architecture":
                    architecture,

                "run_dir":
                    run_dir,

                "fold_count":
                    len(folds),

                "complete_fivefold":
                    (
                        folds
                        == {
                            1,
                            2,
                            3,
                            4,
                            5,
                        }
                    ),

                "checkpoints":
                    sorted(
                        values,
                        key=lambda value:
                            value["fold"],
                    ),
            }
        )

    return {
        "file_count":
            len(files),

        "run_count":
            len(runs),

        "complete_fivefold_run_count":
            sum(
                bool(
                    run[
                        "complete_fivefold"
                    ]
                )
                for run
                in runs
            ),

        "architecture_counts":
            dict(
                sorted(
                    {
                        architecture:
                            sum(
                                run[
                                    "architecture"
                                ]
                                == architecture
                                for run
                                in runs
                            )
                        for architecture
                        in {
                            run[
                                "architecture"
                            ]
                            for run
                            in runs
                        }
                    }.items()
                )
            ),

        "runs":
            sorted(
                runs,
                key=lambda item: (
                    str(
                        item[
                            "architecture"
                        ]
                    ),
                    item[
                        "run_dir"
                    ],
                ),
            ),
    }


def find_training_commands() -> list[dict[str, Any]]:
    records = []

    terms = (
        "300ms",
        "python -m train",
        "KFoldDataloader",
        "k-fold",
        "windowing",
        "-w 300",
        "-o 95",
    )

    for project in PROJECT_CANDIDATES:
        if not project.is_dir():
            continue

        for current, dirs, files in os.walk(
            project
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
                    "results",
                    "outputs",
                }
            ]

            for filename in files:
                path = (
                    Path(current)
                    / filename
                )

                if (
                    path.suffix.lower()
                    not in TEXT_SUFFIXES
                ):
                    continue

                try:
                    if (
                        path.stat().st_size
                        > 1_500_000
                    ):
                        continue
                except Exception:
                    continue

                text = read_text(
                    path
                )

                hits = [
                    term
                    for term
                    in terms
                    if term.lower()
                    in text.lower()
                ]

                if not hits:
                    continue

                lines = []

                for number, line in enumerate(
                    text.splitlines(),
                    start=1,
                ):
                    if any(
                        term.lower()
                        in line.lower()
                        for term
                        in hits
                    ):
                        lines.append(
                            {
                                "line":
                                    number,

                                "text":
                                    line.strip()[
                                        :800
                                    ],
                            }
                        )

                    if len(
                        lines
                    ) >= 80:
                        break

                records.append(
                    {
                        "path":
                            str(
                                path.resolve()
                            ),

                        "sha256":
                            sha256_file(
                                path
                            ),

                        "terms":
                            hits,

                        "lines":
                            lines,
                    }
                )

    return records


def main() -> None:
    windowing = (
        inspect_windowing_sources()
    )

    roots = (
        discover_300ms_roots()
    )

    raw_subjects = (
        discover_raw_subjects()
    )

    checkpoints = (
        discover_300ms_checkpoints()
    )

    commands = (
        find_training_commands()
    )

    print()
    print(
        "WINDOWING SOURCES:",
        len(windowing),
    )

    for item in windowing:
        print()
        print(
            item["path"]
        )

        for line in (
            item[
                "relevant_lines"
            ][:80]
        ):
            print(
                f"  L{line['line']}: "
                f"{line['text']}"
            )

    print()
    print(
        "300-ms PROCESSED ROOTS:",
        len(roots),
    )

    for root in roots:
        print()
        print(
            root["path"]
        )

        print(
            "  subjects:",
            root[
                "subject_count"
            ],
        )

        print(
            "  999:",
            root[
                "contains_999"
            ],
        )

        print(
            "  1000:",
            root[
                "contains_1000"
            ],
        )

        print(
            "  subject IDs:",
            root[
                "subjects"
            ],
        )

    print()
    print(
        "ORIENTED RAW SUBJECT POPULATIONS"
    )

    for dataset, values in (
        raw_subjects.items()
    ):
        print()
        print(
            dataset
        )

        for value in values:
            print(
                " ",
                value[
                    "path"
                ],
            )

            print(
                "    subject count:",
                value[
                    "subject_count"
                ],
            )

            print(
                "    subjects:",
                value[
                    "subjects"
                ],
            )

    print()
    print(
        "300-ms CHECKPOINT FILES:",
        checkpoints[
            "file_count"
        ],
    )

    print(
        "300-ms RUNS:",
        checkpoints[
            "run_count"
        ],
    )

    print(
        "COMPLETE 5-FOLD RUNS:",
        checkpoints[
            "complete_fivefold_run_count"
        ],
    )

    print(
        "RUNS BY ARCHITECTURE:",
        checkpoints[
            "architecture_counts"
        ],
    )

    for run in checkpoints[
        "runs"
    ]:
        if not run[
            "complete_fivefold"
        ]:
            continue

        print()
        print(
            "COMPLETE:",
            run[
                "architecture"
            ],
            run[
                "run_dir"
            ],
        )

        for checkpoint in (
            run[
                "checkpoints"
            ]
        ):
            print(
                "  fold",
                checkpoint[
                    "fold"
                ],
                checkpoint[
                    "sha256"
                ],
            )

    print()
    print(
        "HISTORICAL 300-ms COMMAND REFERENCES:",
        len(commands),
    )

    for item in commands[
        :100
    ]:
        print()
        print(
            item[
                "path"
            ],
        )

        for line in (
            item[
                "lines"
            ][:20]
        ):
            print(
                f"  L{line['line']}: "
                f"{line['text']}"
            )

    manifest = {
        "schema":
            "crosslayer_phase3_300ms_primary_protocol_audit_v1",

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

        "proposed_protocol_direction": {
            "primary_window_ms":
                300,

            "primary_population":
                "UNIVR_PLUS_KFALL",

            "expected_real_subject_count":
                61,

            "onfield_role":
                (
                    "AUGMENTATION_OR_EXTERNAL_ROBUSTNESS"
                ),

            "historical_400ms_role":
                "REFERENCE_AND_WINDOW_SENSITIVITY",

            "manual_numeric_collision_renaming_allowed":
                False,

            "canonical_dataset_prefixed_ids_required":
                True,
        },

        "windowing_source_audit":
            windowing,

        "processed_300ms_roots":
            roots,

        "raw_oriented_subject_populations":
            raw_subjects,

        "checkpoint_inventory":
            checkpoints,

        "historical_command_references":
            commands,

        "unresolved_before_freeze": [
            (
                "exact semantic meaning of preprocessing -o 95"
            ),
            (
                "why historical output path contains 50ov"
            ),
            (
                "which complete 300-ms CNN run corresponds to "
                "the intended UniVR+KFall population"
            ),
            (
                "canonical subject mapping for manually renamed "
                "historical combined directories"
            ),
            (
                "exact train/validation/test subjects in each fold"
            ),
            (
                "event onset/impact mapping"
            ),
        ],

        "scientific_boundary": {
            "new_data_generated":
                False,

            "new_fold_generated":
                False,

            "checkpoint_selected_using_performance":
                False,

            "model_retrained":
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

    print()
    print(
        "PHASE_3_300MS_PRIMARY_PROTOCOL_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
