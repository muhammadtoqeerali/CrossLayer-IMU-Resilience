from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

TOQEER = Path.home() / "toqeer"

BACKUP_HOME = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer"
)

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_3e_checkpoint_simulator_lineage_v1.json"
)

KFOLD_SOURCE = (
    TOQEER
    / "Protechto-master"
    / "dataloaders"
    / "KFoldDataloader.py"
)

# ------------------------------------------------------------
# Search locations
# ------------------------------------------------------------

CHECKPOINT_SEARCH_BASES = [
    Path(
        "/mnt/hdd16T/protechto/checkpoints"
    ),
    BACKUP_HOME,
]

SIMULATOR_EXPLICIT_CANDIDATES = [
    TOQEER / "Falling_simulator",
    TOQEER / "Falling_Simulator",
    TOQEER / "falling_simulator",
    TOQEER / "FallingSimulator",
    TOQEER / "falling-simulator",
    BACKUP_HOME / "Falling_simulator",
    BACKUP_HOME / "Falling_Simulator",
    BACKUP_HOME / "falling_simulator",
    BACKUP_HOME / "FallingSimulator",
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

TEXT_EXTENSIONS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".sh",
    ".csv",
}

CHECKPOINT_PATTERN = re.compile(
    r"^(?P<kind>best-checkpoint|quantized)"
    r"-fold_(?P<fold>[1-5])\.ckpt$"
)

WINDOW_PATTERN = re.compile(
    r"(?P<window>300|400)ms",
    re.IGNORECASE,
)

RUN_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}"
)

SIM_TERMS = (
    "UniVrFall",
    "UniVR",
    "KFall",
    "400ms",
    "300ms",
    "50ov",
    "segments",
    "KFoldDataloader",
    "FrameCounter",
    "fall onset",
    "fall impact",
    "onset",
    "impact",
    "checkpoint",
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def discover_checkpoint_dirs() -> list[Path]:
    """
    Discover checkpoint directories without recursively scanning all data trees.
    """
    result = set()

    direct = Path(
        "/mnt/hdd16T/protechto/checkpoints"
    )

    if direct.is_dir():
        result.add(
            direct.resolve()
        )

    # Find directories literally named "checkpoints" under the user's
    # workstation project area, with depth/pruning protection.
    root = BACKUP_HOME

    if root.is_dir():
        root_depth = len(
            root.parts
        )

        skip = {
            ".git",
            "data",
            "datasets",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "results",
            "outputs",
            "runs",
        }

        for current, dirs, _files in os.walk(
            root
        ):
            current_path = Path(
                current
            )

            depth = (
                len(
                    current_path.parts
                )
                - root_depth
            )

            dirs[:] = [
                value
                for value
                in dirs
                if value not in skip
            ]

            if depth >= 4:
                dirs[:] = []
                continue

            if (
                current_path.name.lower()
                == "checkpoints"
            ):
                result.add(
                    current_path.resolve()
                )

                dirs[:] = []

    return sorted(
        result,
        key=str,
    )


def checkpoint_inventory(
    roots: list[Path],
) -> dict[str, Any]:
    files = []

    seen = set()

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob(
            "*.ckpt"
        ):
            match = CHECKPOINT_PATTERN.match(
                path.name
            )

            if match is None:
                continue

            resolved = path.resolve()

            if str(
                resolved
            ) in seen:
                continue

            seen.add(
                str(
                    resolved
                )
            )

            window_match = WINDOW_PATTERN.search(
                str(
                    resolved
                )
            )

            window_ms = (
                int(
                    window_match.group(
                        "window"
                    )
                )
                if window_match
                else None
            )

            run_match = RUN_PATTERN.search(
                str(
                    resolved
                )
            )

            run_id = (
                run_match.group(0)
                if run_match
                else resolved.parent.name
            )

            info = {
                "path":
                    str(
                        resolved
                    ),

                "parent":
                    str(
                        resolved.parent
                    ),

                "run_id":
                    run_id,

                "window_ms":
                    window_ms,

                "kind":
                    match.group(
                        "kind"
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

            files.append(
                info
            )

    grouped: dict[
        tuple[str, int | None],
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for item in files:
        grouped[
            (
                item[
                    "parent"
                ],
                item[
                    "window_ms"
                ],
            )
        ].append(
            item
        )

    runs = []

    for (
        parent,
        window_ms,
    ), values in grouped.items():

        best = {
            item[
                "fold"
            ]:
                item
            for item
            in values
            if (
                item[
                    "kind"
                ]
                == "best-checkpoint"
            )
        }

        quantized = {
            item[
                "fold"
            ]:
                item
            for item
            in values
            if (
                item[
                    "kind"
                ]
                == "quantized"
            )
        }

        runs.append(
            {
                "run_dir":
                    parent,

                "window_ms":
                    window_ms,

                "fp32_folds":
                    {
                        str(key):
                            value
                        for key, value
                        in sorted(
                            best.items()
                        )
                    },

                "quantized_folds":
                    {
                        str(key):
                            value
                        for key, value
                        in sorted(
                            quantized.items()
                        )
                    },

                "complete_fp32_fivefold":
                    (
                        set(
                            best
                        )
                        == {
                            1,
                            2,
                            3,
                            4,
                            5,
                        }
                    ),

                "complete_quantized_fivefold":
                    (
                        set(
                            quantized
                        )
                        == {
                            1,
                            2,
                            3,
                            4,
                            5,
                        }
                    ),
            }
        )

    return {
        "checkpoint_roots":
            [
                str(
                    root
                )
                for root
                in roots
            ],

        "fold_checkpoint_file_count":
            len(
                files
            ),

        "runs":
            sorted(
                runs,
                key=lambda value: (
                    value[
                        "window_ms"
                    ]
                    if value[
                        "window_ms"
                    ]
                    is not None
                    else 9999,
                    value[
                        "run_dir"
                    ],
                ),
            ),
    }


def discover_simulators() -> list[Path]:
    result = set()

    for path in (
        SIMULATOR_EXPLICIT_CANDIDATES
    ):
        if path.is_dir():
            result.add(
                path.resolve()
            )

    # Shallow discovery for variant names.
    root = BACKUP_HOME

    if root.is_dir():
        root_depth = len(
            root.parts
        )

        skip = {
            ".git",
            "data",
            "datasets",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "checkpoints",
            "results",
            "outputs",
        }

        for current, dirs, _files in os.walk(
            root
        ):
            current_path = Path(
                current
            )

            depth = (
                len(
                    current_path.parts
                )
                - root_depth
            )

            dirs[:] = [
                item
                for item
                in dirs
                if item not in skip
            ]

            if depth > 2:
                dirs[:] = []
                continue

            name = (
                current_path.name.lower()
            )

            if (
                "fall" in name
                and "sim" in name
            ):
                result.add(
                    current_path.resolve()
                )

    return sorted(
        result,
        key=str,
    )


def simulator_source_audit(
    root: Path,
) -> dict[str, Any]:
    matches = []

    absolute_paths = set()

    file_count = 0

    skip = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "checkpoints",
        "results",
        "outputs",
        "runs",
    }

    path_regex = re.compile(
        r"(/[A-Za-z0-9_.\-]+)+"
    )

    for current, dirs, files in os.walk(
        root
    ):
        dirs[:] = [
            item
            for item
            in dirs
            if item not in skip
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
                not in TEXT_EXTENSIONS
            ):
                continue

            try:
                size = (
                    path.stat().st_size
                )
            except Exception:
                continue

            if (
                size
                > 2_000_000
            ):
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            file_count += 1

            hits = [
                term
                for term
                in SIM_TERMS
                if term.lower()
                in text.lower()
            ]

            if not hits:
                continue

            rel = path.relative_to(
                root
            ).as_posix()

            excerpts = []

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

                for found in path_regex.findall(
                    line
                ):
                    if (
                        "/data/" in found
                        or "/dataset" in found.lower()
                        or "/segments/" in found
                        or "/protechto/" in found.lower()
                    ):
                        absolute_paths.add(
                            found
                        )

                if len(
                    excerpts
                ) >= 40:
                    break

            matches.append(
                {
                    "path":
                        rel,

                    "sha256":
                        sha256_file(
                            path
                        ),

                    "terms":
                        hits,

                    "excerpts":
                        excerpts,
                }
            )

    return {
        "root":
            str(
                root
            ),

        "text_files_scanned":
            file_count,

        "matching_files":
            matches,

        "absolute_dataset_path_candidates":
            sorted(
                absolute_paths
            ),
    }


def kfold_source_audit() -> dict[str, Any]:
    if not KFOLD_SOURCE.is_file():
        raise RuntimeError(
            "Primary KFoldDataloader.py missing"
        )

    text = KFOLD_SOURCE.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(
        text
    )

    subject_assignments = []

    kfold_calls = []

    split_calls = []

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
                    subject_assignments.append(
                        {
                            "line":
                                int(
                                    node.lineno
                                ),

                            "expression":
                                ast.unparse(
                                    node.value
                                ),

                            "source":
                                (
                                    ast.get_source_segment(
                                        text,
                                        node,
                                    )
                                    or ""
                                ),
                        }
                    )

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if isinstance(
            func,
            ast.Name,
        ):
            name = func.id

        elif isinstance(
            func,
            ast.Attribute,
        ):
            name = func.attr

        else:
            name = None

        if name == "KFold":
            kfold_calls.append(
                {
                    "line":
                        int(
                            node.lineno
                        ),

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
                            key.arg:
                                ast.unparse(
                                    key.value
                                )
                            for key
                            in node.keywords
                            if key.arg
                        },
                }
            )

        if name == "train_test_split":
            split_calls.append(
                {
                    "line":
                        int(
                            node.lineno
                        ),

                    "source":
                        (
                            ast.get_source_segment(
                                text,
                                node,
                            )
                            or ""
                        ),

                    "args":
                        [
                            ast.unparse(
                                value
                            )
                            for value
                            in node.args
                        ],

                    "keywords":
                        {
                            key.arg:
                                ast.unparse(
                                    key.value
                                )
                            for key
                            in node.keywords
                            if key.arg
                        },
                }
            )

    expressions = [
        item[
            "expression"
        ]
        for item
        in subject_assignments
    ]

    if any(
        expression.startswith(
            "sorted("
        )
        for expression
        in expressions
    ):
        enumeration = (
            "SORTED_EXPLICITLY"
        )

    elif any(
        "os.listdir"
        in expression
        for expression
        in expressions
    ):
        enumeration = (
            "FILESYSTEM_ORDER_OS_LISTDIR"
        )

    else:
        enumeration = (
            "OTHER_OR_UNRESOLVED"
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

        "subject_assignments":
            subject_assignments,

        "subject_enumeration_semantics":
            enumeration,

        "kfold_calls":
            kfold_calls,

        "train_test_split_calls":
            split_calls,
    }


def inspect_dataset_root(
    path: Path,
) -> dict[str, Any]:
    item = {
        "path":
            str(
                path
            ),

        "exists":
            path.is_dir(),
    }

    if not path.is_dir():
        return item

    filesystem_order = [
        child.name
        for child
        in path.iterdir()
        if (
            child.is_dir()
            and child.name.isdigit()
        )
    ]

    numeric_order = sorted(
        filesystem_order,
        key=lambda value: int(
            value
        ),
    )

    item.update(
        {
            "subject_count":
                len(
                    filesystem_order
                ),

            "filesystem_subject_order":
                filesystem_order,

            "numeric_subject_order":
                numeric_order,

            "filesystem_equals_numeric_order":
                (
                    filesystem_order
                    == numeric_order
                ),

            "contains_999":
                (
                    "999"
                    in filesystem_order
                ),

            "contains_1000":
                (
                    "1000"
                    in filesystem_order
                ),
        }
    )

    return item


def metadata_references_for_runs(
    runs: list[dict[str, Any]],
    simulator_roots: list[Path],
) -> list[dict[str, Any]]:
    run_ids = set()

    for run in runs:
        match = RUN_PATTERN.search(
            run[
                "run_dir"
            ]
        )

        if match:
            run_ids.add(
                match.group(0)
            )

    if not run_ids:
        return []

    roots = [
        TOQEER / "Protechto-master",
        TOQEER / "Protechto_master",
        TOQEER / "IMU_Reliability",
        *simulator_roots,
    ]

    records = []

    seen = set()

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

    for root in roots:
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
                not in skip
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

                if size > 1_500_000:
                    continue

                try:
                    text = path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception:
                    continue

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
                        value
                        in line
                        for value
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
                    ) >= 30:
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


def main() -> int:
    checkpoint_roots = (
        discover_checkpoint_dirs()
    )

    print(
        "Checkpoint roots discovered:",
        len(
            checkpoint_roots
        ),
    )

    for root in checkpoint_roots:
        print(
            " ",
            root
        )

    checkpoint_data = (
        checkpoint_inventory(
            checkpoint_roots
        )
    )

    runs_300 = [
        item
        for item
        in checkpoint_data[
            "runs"
        ]
        if item[
            "window_ms"
        ]
        == 300
    ]

    runs_400 = [
        item
        for item
        in checkpoint_data[
            "runs"
        ]
        if item[
            "window_ms"
        ]
        == 400
    ]

    complete_300 = [
        item
        for item
        in runs_300
        if item[
            "complete_fp32_fivefold"
        ]
    ]

    complete_400 = [
        item
        for item
        in runs_400
        if item[
            "complete_fp32_fivefold"
        ]
    ]

    print()
    print(
        "300-ms fold runs:",
        len(
            runs_300
        ),
    )

    print(
        "Complete 300-ms FP32 five-fold runs:",
        len(
            complete_300
        ),
    )

    print(
        "400-ms fold runs:",
        len(
            runs_400
        ),
    )

    print(
        "Complete 400-ms FP32 five-fold runs:",
        len(
            complete_400
        ),
    )

    simulator_roots = (
        discover_simulators()
    )

    print()
    print(
        "Falling-simulator roots discovered:",
        len(
            simulator_roots
        ),
    )

    for root in simulator_roots:
        print(
            " ",
            root
        )

    simulator_audits = [
        simulator_source_audit(
            root
        )
        for root
        in simulator_roots
    ]

    kfold = (
        kfold_source_audit()
    )

    print()
    print(
        "KFold source:",
        kfold[
            "path"
        ],
    )

    print(
        "KFold SHA256:",
        kfold[
            "sha256"
        ],
    )

    print(
        "Subject enumeration:",
        kfold[
            "subject_enumeration_semantics"
        ],
    )

    print(
        "Subject assignments:",
        kfold[
            "subject_assignments"
        ],
    )

    dataset_roots = [
        inspect_dataset_root(
            path
        )
        for path
        in DATA_ROOTS
    ]

    metadata = (
        metadata_references_for_runs(
            checkpoint_data[
                "runs"
            ],
            simulator_roots,
        )
    )

    manifest = {
        "schema":
            "crosslayer_phase3e_checkpoint_simulator_lineage_v1",

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

        "baseline_policy": {
            "primary_window_ms":
                400,

            "primary_role":
                "FROZEN_CROSSLAYER_REFERENCE",

            "secondary_window_ms":
                300,

            "secondary_role":
                "TEMPORAL_RESOLUTION_SENSITIVITY_COMPARATOR",

            "replace_400ms_with_300ms":
                False,
        },

        "checkpoint_inventory":
            checkpoint_data,

        "complete_300ms_fp32_fivefold_runs":
            complete_300,

        "complete_400ms_fp32_fivefold_runs":
            complete_400,

        "falling_simulator_roots":
            [
                str(
                    root
                )
                for root
                in simulator_roots
            ],

        "falling_simulator_audits":
            simulator_audits,

        "authoritative_kfold_source":
            kfold,

        "protected_dataset_root_audit":
            dataset_roots,

        "checkpoint_metadata_references":
            metadata,

        "decision_state": {
            "primary_400ms_remains_frozen":
                True,

            "300ms_may_be_used_as_comparator":
                True,

            "fivefold_checkpoint_family_status":
                (
                    "DISCOVERED_OR_LINEAGE_AUDITED_NOT_YET_PHASE3_FROZEN"
                ),

            "falling_simulator_dataset_status":
                (
                    "PREFERRED_LINEAGE_CANDIDATE_PENDING_EXACT_TRIAL_RECONCILIATION"
                ),
        },

        "scientific_boundary": {
            "checkpoint_files_modified":
                False,

            "new_folds_generated":
                False,

            "models_retrained":
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
        "PHASE_3E_CHECKPOINT_SIMULATOR_LINEAGE=PASS"
    )


if __name__ == "__main__":
    main()
