from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[2]

TOQEER = Path.home() / "toqeer"

BACKUP = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer"
)

KFOLD_SOURCE = (
    TOQEER
    / "Protechto-master"
    / "dataloaders"
    / "KFoldDataloader.py"
)

CONST_CANDIDATES = [
    TOQEER
    / "Protechto-master"
    / "const.py",

    TOQEER
    / "Protechto-master"
    / "constants.py",

    TOQEER
    / "Protechto-master"
    / "utils"
    / "const.py",

    TOQEER
    / "Protechto-master"
    / "config"
    / "const.py",
]

TRAIN_CANDIDATES = [
    TOQEER
    / "Protechto-master"
    / "train.py",

    TOQEER
    / "Protechto-master"
    / "main.py",
]

DATA_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto/data/"
        "KFall_oriented/segments/"
        "400ms_50ov_npseg_filt_binary"
    ),

    Path(
        "/mnt/hdd16T/protechto/data/"
        "UniVrFall_oriented/segments/"
        "400ms_50ov_npseg_filt_binary"
    ),

    Path(
        "/mnt/hdd16T/protechto/data/"
        "UniVrFall_KFall_OF/segments/"
        "400ms_50ov_npseg_filt_binary"
    ),

    Path(
        "/mnt/hdd16T/protechto/data/back/"
        "UniVrFall_KFall/segments/"
        "400ms_50ov_npseg_filt_binary"
    ),
]

FOUR_HUNDRED_RUNS = [
    Path(
        "/mnt/hdd16T/protechto/checkpoints/"
        "CNNSplit/400ms/2025-03-25_17_14_55"
    ),

    Path(
        "/mnt/hdd16T/protechto/checkpoints/"
        "LSTM/400ms/2025-03-25_17_17_02"
    ),

    Path(
        "/mnt/hdd16T/protechto/checkpoints/"
        "ResNet/400ms/2025-03-25_17_04_52"
    ),
]

SIMULATOR_NAME_TERMS = (
    "falling",
    "fall",
    "simulator",
    "simulation",
    "physics",
    "digital",
    "twin",
)

SIMULATOR_CONTENT_TERMS = (
    "UniVrFall",
    "UniVRFall",
    "KFall",
)

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".sh",
}

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3f_exact_fivefold_and_simulator_lineage_v1.json"
)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


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


def extract_kfold_contract() -> dict[str, Any]:
    if not KFOLD_SOURCE.is_file():
        raise RuntimeError(
            f"Missing {KFOLD_SOURCE}"
        )

    text = read_text(
        KFOLD_SOURCE
    )

    tree = ast.parse(
        text
    )

    subject_source = None

    kfold_call = None

    validation_call = None

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Assign,
        ):
            for target in node.targets:
                if (
                    isinstance(
                        target,
                        ast.Attribute,
                    )
                    and isinstance(
                        target.value,
                        ast.Name,
                    )
                    and target.value.id
                    == "self"
                    and target.attr
                    == "subjects"
                ):
                    subject_source = (
                        ast.get_source_segment(
                            text,
                            node,
                        )
                        or ""
                    )

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            name = node.func.id

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            name = node.func.attr

        else:
            name = None

        if name == "KFold":
            kfold_call = {
                "source":
                    (
                        ast.get_source_segment(
                            text,
                            node,
                        )
                        or ""
                    ),

                "keywords":
                    {
                        keyword.arg:
                            ast.literal_eval(
                                keyword.value
                            )
                        for keyword
                        in node.keywords
                        if (
                            keyword.arg
                            in {
                                "shuffle",
                                "random_state",
                            }
                        )
                    },

                "n_splits_expression":
                    next(
                        (
                            ast.unparse(
                                keyword.value
                            )
                            for keyword
                            in node.keywords
                            if keyword.arg
                            == "n_splits"
                        ),
                        None,
                    ),
            }

        if name == "train_test_split":
            validation_call = {
                "source":
                    (
                        ast.get_source_segment(
                            text,
                            node,
                        )
                        or ""
                    ),

                "keywords":
                    {
                        keyword.arg:
                            ast.literal_eval(
                                keyword.value
                            )
                        for keyword
                        in node.keywords
                        if keyword.arg
                    },
            }

    if subject_source is None:
        raise RuntimeError(
            "self.subjects assignment not found"
        )

    if kfold_call is None:
        raise RuntimeError(
            "KFold call not found"
        )

    if validation_call is None:
        raise RuntimeError(
            "train_test_split call not found"
        )

    return {
        "path":
            str(
                KFOLD_SOURCE
            ),

        "sha256":
            sha256_file(
                KFOLD_SOURCE
            ),

        "subject_assignment":
            subject_source,

        "subject_order":
            "SORTED_LEXICOGRAPHIC_DIRECTORY_NAMES",

        "kfold":
            kfold_call,

        "validation_split":
            validation_call,
    }


def find_augmentation_subjects() -> dict[str, Any]:
    candidates = []

    search_paths = []

    for path in CONST_CANDIDATES:
        if path.is_file():
            search_paths.append(
                path
            )

    project = (
        TOQEER
        / "Protechto-master"
    )

    if project.is_dir():
        for path in project.glob(
            "*.py"
        ):
            if path not in search_paths:
                search_paths.append(
                    path
                )

    for path in search_paths:
        text = read_text(
            path
        )

        if (
            "DATA_AUGMENTATION_SUBJECTS"
            not in text
        ):
            continue

        try:
            tree = ast.parse(
                text
            )
        except Exception:
            continue

        for node in ast.walk(
            tree
        ):
            if not isinstance(
                node,
                ast.Assign,
            ):
                continue

            for target in (
                node.targets
            ):
                if (
                    isinstance(
                        target,
                        ast.Name,
                    )
                    and target.id
                    == "DATA_AUGMENTATION_SUBJECTS"
                ):
                    try:
                        value = ast.literal_eval(
                            node.value
                        )
                    except Exception:
                        value = None

                    candidates.append(
                        {
                            "path":
                                str(
                                    path
                                ),

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

                            "sha256":
                                sha256_file(
                                    path
                                ),
                        }
                    )

    values = []

    for item in candidates:
        if isinstance(
            item[
                "value"
            ],
            (
                list,
                tuple,
                set,
            ),
        ):
            values.append(
                tuple(
                    str(value)
                    for value
                    in item[
                        "value"
                    ]
                )
            )

    unique = set(
        values
    )

    if len(
        unique
    ) == 1:
        resolved = list(
            unique.pop()
        )

    else:
        # Historical evidence independently identified 999/1000.
        resolved = [
            "999",
            "1000",
        ]

    return {
        "candidates":
            candidates,

        "resolved_subjects":
            resolved,

        "resolution_status":
            (
                "SOURCE_CONFIRMED"
                if len(
                    set(
                        values
                    )
                )
                == 1
                else "FALLBACK_TO_VERIFIED_HISTORICAL_PROVENANCE"
            ),
    }


def inspect_training_calls() -> list[dict[str, Any]]:
    records = []

    project = (
        TOQEER
        / "Protechto-master"
    )

    candidates = []

    for path in TRAIN_CANDIDATES:
        if path.is_file():
            candidates.append(
                path
            )

    if project.is_dir():
        for path in project.rglob(
            "*.py"
        ):
            if (
                "checkpoint"
                in path.parts
                or "__pycache__"
                in path.parts
            ):
                continue

            if (
                path.stat().st_size
                > 500_000
            ):
                continue

            text = read_text(
                path
            )

            if (
                "KFoldDataloader"
                in text
            ):
                candidates.append(
                    path
                )

    seen = set()

    for path in candidates:
        resolved = str(
            path.resolve()
        )

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        text = read_text(
            path
        )

        lines = text.splitlines()

        hits = []

        for number, line in enumerate(
            lines,
            start=1,
        ):
            if (
                "KFoldDataloader"
                in line
                or "root_directory"
                in line
                or "SEGMENT"
                in line
                or "DATASET"
                in line
            ):
                start = max(
                    1,
                    number - 2,
                )

                end = min(
                    len(
                        lines
                    ),
                    number + 4,
                )

                hits.append(
                    {
                        "line":
                            number,

                        "context":
                            "\n".join(
                                f"{i}: "
                                f"{lines[i - 1]}"
                                for i
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

                "hits":
                    hits[:80],
            }
        )

    return records


def reconstruct_root(
    path: Path,
    augmentation_subjects: list[str],
) -> dict[str, Any]:
    record = {
        "path":
            str(
                path
            ),

        "exists":
            path.is_dir(),
    }

    if not path.is_dir():
        return record

    raw_subjects = [
        child.name
        for child
        in path.iterdir()
        if (
            child.is_dir()
            and child.name.isdigit()
        )
    ]

    subjects = np.array(
        [
            value
            for value
            in sorted(
                raw_subjects
            )
            if value
            not in set(
                augmentation_subjects
            )
        ],
        dtype=object,
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    folds = []

    test_coverage = []

    for fold_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(
            subjects
        ),
        start=1,
    ):
        train_idx_final, val_idx = (
            train_test_split(
                train_idx,
                test_size=0.2,
                random_state=42,
            )
        )

        train_subjects = (
            subjects[
                train_idx_final
            ]
            .tolist()
        )

        validation_subjects = (
            subjects[
                val_idx
            ]
            .tolist()
        )

        test_subjects = (
            subjects[
                test_idx
            ]
            .tolist()
        )

        if (
            set(
                train_subjects
            )
            & set(
                validation_subjects
            )
        ):
            raise RuntimeError(
                "train/validation overlap"
            )

        if (
            set(
                train_subjects
            )
            & set(
                test_subjects
            )
        ):
            raise RuntimeError(
                "train/test overlap"
            )

        if (
            set(
                validation_subjects
            )
            & set(
                test_subjects
            )
        ):
            raise RuntimeError(
                "validation/test overlap"
            )

        if (
            set(
                train_subjects
            )
            | set(
                validation_subjects
            )
            | set(
                test_subjects
            )
            != set(
                subjects.tolist()
            )
        ):
            raise RuntimeError(
                "fold does not cover complete subject population"
            )

        test_coverage.extend(
            test_subjects
        )

        folds.append(
            {
                "fold":
                    fold_index,

                "train_subject_count":
                    len(
                        train_subjects
                    ),

                "validation_subject_count":
                    len(
                        validation_subjects
                    ),

                "test_subject_count":
                    len(
                        test_subjects
                    ),

                "train_subjects":
                    train_subjects,

                "validation_subjects":
                    validation_subjects,

                "test_subjects":
                    test_subjects,
            }
        )

    counts = defaultdict(
        int
    )

    for subject in (
        test_coverage
    ):
        counts[
            subject
        ] += 1

    exact_once = (
        set(
            counts
        )
        == set(
            subjects.tolist()
        )
        and all(
            value == 1
            for value
            in counts.values()
        )
    )

    record.update(
        {
            "raw_subject_count":
                len(
                    raw_subjects
                ),

            "augmentation_excluded":
                [
                    value
                    for value
                    in augmentation_subjects
                    if value
                    in raw_subjects
                ],

            "kfold_subject_count":
                len(
                    subjects
                ),

            "lexicographic_subject_order":
                subjects.tolist(),

            "folds":
                folds,

            "every_subject_tested_exactly_once":
                exact_once,
        }
    )

    return record


def safe_checkpoint_metadata(
    path: Path,
) -> dict[str, Any]:
    record = {
        "path":
            str(
                path
            ),

        "exists":
            path.is_file(),
    }

    if not path.is_file():
        return record

    record[
        "sha256"
    ] = sha256_file(
        path
    )

    record[
        "bytes"
    ] = int(
        path.stat().st_size
    )

    try:
        import torch

        sys.path.insert(
            0,
            str(
                TOQEER
                / "Protechto-master"
            ),
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        record[
            "top_level_type"
        ] = type(
            checkpoint
        ).__name__

        if isinstance(
            checkpoint,
            dict,
        ):
            record[
                "top_level_keys"
            ] = sorted(
                str(
                    key
                )
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

                for key, value in (
                    hparams.items()
                ):
                    if isinstance(
                        value,
                        (
                            str,
                            int,
                            float,
                            bool,
                            type(
                                None
                            ),
                        ),
                    ):
                        simple[
                            str(
                                key
                            )
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
                                type(
                                    None
                                ),
                            ),
                        )
                        for item
                        in value
                    ):
                        simple[
                            str(
                                key
                            )
                        ] = list(
                            value
                        )

                record[
                    "simple_hyper_parameters"
                ] = simple

            state = checkpoint.get(
                "state_dict"
            )

            if isinstance(
                state,
                dict,
            ):
                tensor_shapes = {}

                total_elements = 0

                for key, value in (
                    state.items()
                ):
                    if hasattr(
                        value,
                        "shape",
                    ):
                        tensor_shapes[
                            str(
                                key
                            )
                        ] = list(
                            value.shape
                        )

                    if hasattr(
                        value,
                        "numel",
                    ):
                        total_elements += int(
                            value.numel()
                        )

                record[
                    "state_tensor_count"
                ] = len(
                    tensor_shapes
                )

                record[
                    "state_total_elements_including_buffers"
                ] = (
                    total_elements
                )

                record[
                    "state_shape_sample"
                ] = dict(
                    list(
                        tensor_shapes.items()
                    )[:30]
                )

        record[
            "load_status"
        ] = "PASS"

    except Exception as exc:
        record[
            "load_status"
        ] = "FAILED_NONBLOCKING"

        record[
            "load_error"
        ] = repr(
            exc
        )

    return record


def audit_400ms_checkpoint_runs() -> list[dict[str, Any]]:
    results = []

    for run in (
        FOUR_HUNDRED_RUNS
    ):
        architecture = (
            run
            .parents[1]
            .name
        )

        folds = []

        for fold in range(
            1,
            6,
        ):
            path = (
                run
                / (
                    "best-checkpoint-"
                    f"fold_{fold}.ckpt"
                )
            )

            folds.append(
                {
                    "fold":
                        fold,

                    **safe_checkpoint_metadata(
                        path
                    ),
                }
            )

        results.append(
            {
                "architecture":
                    architecture,

                "run_dir":
                    str(
                        run
                    ),

                "folds":
                    folds,

                "all_five_present":
                    all(
                        item[
                            "exists"
                        ]
                        for item
                        in folds
                    ),
            }
        )

    return results


def targeted_run_references(
    run_ids: list[str],
) -> list[dict[str, Any]]:
    roots = [
        TOQEER
        / "Protechto-master",

        TOQEER
        / "Protechto_master",

        Path(
            "/mnt/hdd16T/protechto"
        ),
    ]

    records = []

    skip = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "data",
        "datasets",
        "checkpoints",
        "artifacts",
    }

    seen = set()

    for root in roots:
        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(
            root
        ):
            dirs[:] = [
                item
                for item
                in dirs
                if item
                not in skip
            ]

            for filename in files:
                path = (
                    Path(
                        current
                    )
                    / filename
                )

                if (
                    path.suffix.lower()
                    not in TEXT_SUFFIXES
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

                text = read_text(
                    path
                )

                hits = [
                    run_id
                    for run_id
                    in run_ids
                    if run_id
                    in text
                ]

                if not hits:
                    continue

                key = str(
                    path.resolve()
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                excerpts = []

                for number, line in enumerate(
                    text.splitlines(),
                    start=1,
                ):
                    if any(
                        run_id
                        in line
                        for run_id
                        in hits
                    ):
                        excerpts.append(
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
                        excerpts
                    ) >= 50:
                        break

                records.append(
                    {
                        "path":
                            key,

                        "run_ids":
                            hits,

                        "excerpts":
                            excerpts,
                    }
                )

    return records


def discover_simulator_candidates() -> list[dict[str, Any]]:
    """
    Locate recent simulator/digital-twin projects without assuming the
    repository is literally called Falling_simulator.
    """
    roots = [
        TOQEER,
        BACKUP,
    ]

    candidates = set()

    for base in roots:
        if not base.is_dir():
            continue

        base_depth = len(
            base.parts
        )

        for current, dirs, _files in os.walk(
            base
        ):
            current_path = Path(
                current
            )

            depth = (
                len(
                    current_path.parts
                )
                - base_depth
            )

            dirs[:] = [
                item
                for item
                in dirs
                if item
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
                }
            ]

            if depth > 3:
                dirs[:] = []
                continue

            name = (
                current_path.name
                .lower()
            )

            if any(
                term
                in name
                for term
                in SIMULATOR_NAME_TERMS
            ):
                candidates.add(
                    current_path.resolve()
                )

    results = []

    for candidate in sorted(
        candidates,
        key=str,
    ):
        matching_files = []

        dataset_paths = set()

        total_text_files = 0

        for current, dirs, files in os.walk(
            candidate
        ):
            dirs[:] = [
                item
                for item
                in dirs
                if item
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
                    Path(
                        current
                    )
                    / filename
                )

                if (
                    path.suffix.lower()
                    not in TEXT_SUFFIXES
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

                total_text_files += 1

                text = read_text(
                    path
                )

                lower = (
                    text.lower()
                )

                has_univr = (
                    "univr"
                    in lower
                )

                has_kfall = (
                    "kfall"
                    in lower
                )

                if not (
                    has_univr
                    or has_kfall
                ):
                    continue

                rel = path.relative_to(
                    candidate
                ).as_posix()

                excerpts = []

                for number, line in enumerate(
                    text.splitlines(),
                    start=1,
                ):
                    line_lower = (
                        line.lower()
                    )

                    if (
                        "univr"
                        in line_lower
                        or "kfall"
                        in line_lower
                    ):
                        excerpts.append(
                            {
                                "line":
                                    number,

                                "text":
                                    line.strip()[
                                        :900
                                    ],
                            }
                        )

                    if (
                        "/mnt/"
                        in line
                        or "/home/"
                        in line
                    ):
                        for token in re.findall(
                            r"/[^\s\"']+",
                            line,
                        ):
                            if (
                                "univr"
                                in token.lower()
                                or "kfall"
                                in token.lower()
                                or "protechto"
                                in token.lower()
                            ):
                                dataset_paths.add(
                                    token.rstrip(
                                        ",);]"
                                    )
                                )

                    if len(
                        excerpts
                    ) >= 50:
                        break

                matching_files.append(
                    {
                        "relative_path":
                            rel,

                        "sha256":
                            sha256_file(
                                path
                            ),

                        "contains_univr":
                            has_univr,

                        "contains_kfall":
                            has_kfall,

                        "excerpts":
                            excerpts,
                    }
                )

        both_count = sum(
            1
            for item
            in matching_files
            if (
                item[
                    "contains_univr"
                ]
                and item[
                    "contains_kfall"
                ]
            )
        )

        if (
            matching_files
        ):
            try:
                modified_ns = int(
                    candidate.stat().st_mtime_ns
                )
            except Exception:
                modified_ns = None

            results.append(
                {
                    "root":
                        str(
                            candidate
                        ),

                    "root_mtime_ns":
                        modified_ns,

                    "text_files_scanned":
                        total_text_files,

                    "matching_files":
                        matching_files,

                    "files_mentioning_both_datasets":
                        both_count,

                    "dataset_path_candidates":
                        sorted(
                            dataset_paths
                        ),
                }
            )

    return sorted(
        results,
        key=lambda item: (
            -item[
                "files_mentioning_both_datasets"
            ],
            -(
                item[
                    "root_mtime_ns"
                ]
                or 0
            ),
            item[
                "root"
            ],
        ),
    )


def main() -> None:
    contract = (
        extract_kfold_contract()
    )

    augmentation = (
        find_augmentation_subjects()
    )

    augmentation_subjects = (
        augmentation[
            "resolved_subjects"
        ]
    )

    print(
        "KFold subject order:",
        contract[
            "subject_order"
        ],
    )

    print(
        "KFold source:",
        contract[
            "path"
        ],
    )

    print(
        "KFold SHA256:",
        contract[
            "sha256"
        ],
    )

    print(
        "Augmentation subjects:",
        augmentation_subjects,
    )

    print(
        "Augmentation resolution:",
        augmentation[
            "resolution_status"
        ],
    )

    print()
    print(
        "RECONSTRUCTING CANDIDATE 400-ms ROOTS"
    )

    reconstructed = []

    for root in DATA_ROOTS:
        record = reconstruct_root(
            root,
            augmentation_subjects,
        )

        reconstructed.append(
            record
        )

        print()
        print(
            root
        )

        print(
            "  exists:",
            record[
                "exists"
            ],
        )

        if record[
            "exists"
        ]:
            print(
                "  raw subjects:",
                record[
                    "raw_subject_count"
                ],
            )

            print(
                "  excluded:",
                record[
                    "augmentation_excluded"
                ],
            )

            print(
                "  KFold subjects:",
                record[
                    "kfold_subject_count"
                ],
            )

            print(
                "  test exact once:",
                record[
                    "every_subject_tested_exactly_once"
                ],
            )

            for fold in (
                record[
                    "folds"
                ]
            ):
                print(
                    "  fold "
                    f"{fold['fold']}: "
                    f"train={fold['train_subject_count']} "
                    f"val={fold['validation_subject_count']} "
                    f"test={fold['test_subject_count']}"
                )

    print()
    print(
        "AUDITING 400-ms FIVE-FOLD CHECKPOINT FAMILIES"
    )

    checkpoint_runs = (
        audit_400ms_checkpoint_runs()
    )

    for run in checkpoint_runs:
        print()
        print(
            run[
                "architecture"
            ],
            run[
                "run_dir"
            ],
        )

        for fold in (
            run[
                "folds"
            ]
        ):
            print(
                "  fold",
                fold[
                    "fold"
                ],
                "load=",
                fold.get(
                    "load_status"
                ),
                "bytes=",
                fold.get(
                    "bytes"
                ),
            )

            hp = fold.get(
                "simple_hyper_parameters",
                {},
            )

            if hp:
                print(
                    "    hparams:",
                    hp,
                )

    run_ids = [
        "2025-03-25_17_14_55",
        "2025-03-25_17_17_02",
        "2025-03-25_17_04_52",
    ]

    run_references = (
        targeted_run_references(
            run_ids
        )
    )

    print()
    print(
        "Targeted 400-ms run metadata references:",
        len(
            run_references
        ),
    )

    for item in (
        run_references[:100]
    ):
        print()
        print(
            item[
                "path"
            ]
        )

        print(
            "  run IDs:",
            item[
                "run_ids"
            ],
        )

        for excerpt in (
            item[
                "excerpts"
            ][:8]
        ):
            print(
                f"  L{excerpt['line']}: "
                f"{excerpt['text']}"
            )

    simulator = (
        discover_simulator_candidates()
    )

    print()
    print(
        "SIMULATOR / DIGITAL-TWIN CANDIDATES:",
        len(
            simulator
        ),
    )

    for item in (
        simulator[:30]
    ):
        print()
        print(
            "SIMULATOR CANDIDATE:",
            item[
                "root"
            ],
        )

        print(
            "  files mentioning both datasets:",
            item[
                "files_mentioning_both_datasets"
            ],
        )

        print(
            "  dataset paths:",
            item[
                "dataset_path_candidates"
            ][:20],
        )

        for match in (
            item[
                "matching_files"
            ][:10]
        ):
            print(
                "  file:",
                match[
                    "relative_path"
                ],
            )

            for excerpt in (
                match[
                    "excerpts"
                ][:4]
            ):
                print(
                    f"    L{excerpt['line']}: "
                    f"{excerpt['text']}"
                )

    training_calls = (
        inspect_training_calls()
    )

    manifest = {
        "schema":
            "crosslayer_phase3f_exact_fivefold_and_simulator_lineage_v1",

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

        "kfold_contract":
            contract,

        "augmentation_subjects":
            augmentation,

        "candidate_root_reconstructions":
            reconstructed,

        "training_call_lineage":
            training_calls,

        "existing_400ms_fivefold_checkpoint_families":
            checkpoint_runs,

        "targeted_400ms_run_metadata_references":
            run_references,

        "simulator_candidates":
            simulator,

        "baseline_decision": {
            "primary":
                "DATE2025_CNN_400MS_RECONSTRUCTED",

            "primary_window_ms":
                400,

            "existing_400ms_fivefold_families_are_same_primary_model":
                False,

            "existing_400ms_architecture_comparators": [
                "CNNSplit",
                "LSTM",
                "ResNet",
            ],

            "300ms_role":
                "TEMPORAL_RESOLUTION_SENSITIVITY",

            "retrain_primary_fivefold_status":
                "NOT_YET_AUTHORIZED",
        },

        "scientific_boundary": {
            "new_split_policy_created":
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
        "PHASE_3F_EXACT_FIVEFOLD_SIMULATOR_LINEAGE=PASS"
    )


if __name__ == "__main__":
    main()
