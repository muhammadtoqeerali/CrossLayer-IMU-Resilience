from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

HOME_TOQEER = Path.home() / "toqeer"


HISTORICAL_ROOTS = [
    (
        "KFall_oriented",
        Path(
            "/mnt/hdd16T/protechto/data/"
            "KFall_oriented/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
    (
        "UniVrFall_oriented",
        Path(
            "/mnt/hdd16T/protechto/data/"
            "UniVrFall_oriented/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
    (
        "UniVrFall_KFall_OF",
        Path(
            "/mnt/hdd16T/protechto/data/"
            "UniVrFall_KFall_OF/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
    (
        "OnField",
        Path(
            "/mnt/hdd16T/protechto/data/"
            "OnField/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
    (
        "back_UniVrFall",
        Path(
            "/mnt/hdd16T/protechto/data/back/"
            "UniVrFall/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
    (
        "back_UniVrFall_KFall",
        Path(
            "/mnt/hdd16T/protechto/data/back/"
            "UniVrFall_KFall/segments/"
            "400ms_50ov_npseg_filt_binary"
        ),
    ),
]


ADDITIONAL_CANDIDATES = [
    (
        "user_uniVr_dataset",
        HOME_TOQEER / "uniVr-dataset",
    ),
]


SOURCE_TREES = [
    (
        "RC-RGD-IMU_publish",
        HOME_TOQEER / "RC-RGD-IMU_publish",
    ),
    (
        "IMU_Reliability",
        HOME_TOQEER / "IMU_Reliability",
    ),
    (
        "Protechto-master",
        HOME_TOQEER / "Protechto-master",
    ),
    (
        "Protechto_master",
        HOME_TOQEER / "Protechto_master",
    ),
]


SPECIAL_SUBJECT_IDS = {
    "999",
    "1000",
}


SMALL_FILE_FULL_HASH_LIMIT = (
    8 * 1024 * 1024
)


SAMPLE_HASH_BYTES = (
    1024 * 1024
)


TEXT_EXTENSIONS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".sh",
}


SEARCH_TERMS = (
    "400ms_50ov_npseg_filt_binary",
    "UniVrFall",
    "KFall",
    "OnField",
    "TrainTestDataloader",
    "train_test_split",
    "random_state=42",
    "prediction_bias",
)


OUTPUT = (
    ROOT
    / "manifests"
    / "phase_3a_dataset_provenance_audit_v1.json"
)


REPORT = (
    ROOT
    / "docs"
    / "PHASE_3A_DATASET_PROVENANCE_AUDIT.md"
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


def sampled_sha256(
    path: Path,
) -> dict[str, Any]:
    """
    Non-authoritative duplicate/provenance aid.

    For large files, hash:
    - file size
    - first 1 MiB
    - last 1 MiB

    This is explicitly NOT a replacement for a full content hash.
    """
    size = path.stat().st_size

    if size <= SMALL_FILE_FULL_HASH_LIMIT:
        return {
            "mode":
                "full",

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                size,

            "authoritative":
                True,
        }

    digest = hashlib.sha256()

    digest.update(
        str(size).encode(
            "ascii"
        )
    )
    digest.update(b"\0")

    with path.open("rb") as handle:
        head = handle.read(
            SAMPLE_HASH_BYTES
        )

        digest.update(
            head
        )

        if (
            size
            > SAMPLE_HASH_BYTES
        ):
            handle.seek(
                max(
                    0,
                    size
                    - SAMPLE_HASH_BYTES,
                )
            )

            tail = handle.read(
                SAMPLE_HASH_BYTES
            )

            digest.update(
                tail
            )

    return {
        "mode":
            "sampled_head_tail",

        "sha256":
            digest.hexdigest(),

        "bytes":
            size,

        "authoritative":
            False,
    }


def npy_metadata(
    path: Path,
) -> dict[str, Any]:
    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return {
            "readable":
                True,

            "shape":
                [
                    int(value)
                    for value
                    in array.shape
                ],

            "dtype":
                str(
                    array.dtype
                ),

            "ndim":
                int(
                    array.ndim
                ),

            "elements":
                int(
                    array.size
                ),
        }

    except Exception as exc:
        return {
            "readable":
                False,

            "error":
                repr(
                    exc
                ),
        }


def safe_subject_sort_key(
    value: str,
):
    if value.isdigit():
        return (
            0,
            int(value),
        )

    return (
        1,
        value,
    )


def inspect_segment_root(
    name: str,
    root: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name":
            name,

        "path":
            str(root),

        "exists":
            root.is_dir(),
    }

    if not root.is_dir():
        return record

    file_count = 0
    directory_count = 0
    total_bytes = 0

    extension_counts = Counter()
    basename_counts = Counter()

    subject_windows = defaultdict(int)
    subject_segment_files = defaultdict(int)

    segment_files = []

    companion_patterns = Counter()

    npy_shapes = Counter()
    npy_dtypes = Counter()

    metadata_lines = []

    for current, dirs, files in os.walk(
        root
    ):
        directory_count += 1

        current_path = Path(current)

        dirs.sort()
        files.sort()

        for filename in files:
            path = (
                current_path
                / filename
            )

            try:
                stat = path.stat()
            except Exception:
                continue

            file_count += 1
            total_bytes += int(
                stat.st_size
            )

            suffix = (
                path.suffix.lower()
            )

            extension_counts[
                suffix or "<none>"
            ] += 1

            basename_counts[
                filename
            ] += 1

            try:
                relative = path.relative_to(
                    root
                )
            except ValueError:
                continue

            relative_text = (
                relative.as_posix()
            )

            if suffix == ".npy":
                meta = npy_metadata(
                    path
                )

                if meta.get(
                    "readable"
                ):
                    shape = tuple(
                        meta["shape"]
                    )

                    npy_shapes[
                        str(shape)
                    ] += 1

                    npy_dtypes[
                        meta["dtype"]
                    ] += 1

                if (
                    filename
                    == "segments.npy"
                ):
                    fingerprint = (
                        sampled_sha256(
                            path
                        )
                    )

                    subject = (
                        relative.parts[0]
                        if relative.parts
                        else "<unknown>"
                    )

                    n_windows = None

                    if (
                        meta.get(
                            "readable"
                        )
                        and meta[
                            "shape"
                        ]
                    ):
                        n_windows = int(
                            meta[
                                "shape"
                            ][0]
                        )

                        subject_windows[
                            subject
                        ] += n_windows

                    subject_segment_files[
                        subject
                    ] += 1

                    companions = []

                    try:
                        sibling_names = sorted(
                            child.name
                            for child
                            in path.parent.iterdir()
                            if child.is_file()
                        )
                    except Exception:
                        sibling_names = []

                    for sibling_name in sibling_names:
                        companions.append(
                            sibling_name
                        )

                        companion_patterns[
                            sibling_name
                        ] += 1

                    segment_files.append(
                        {
                            "relative_path":
                                relative_text,

                            "subject":
                                subject,

                            "window_count":
                                n_windows,

                            "array":
                                meta,

                            "fingerprint":
                                fingerprint,

                            "companions":
                                companions,
                        }
                    )

            metadata_lines.append(
                "\t".join(
                    [
                        relative_text,
                        str(
                            int(
                                stat.st_size
                            )
                        ),
                        str(
                            int(
                                stat.st_mtime_ns
                            )
                        ),
                    ]
                )
            )

    tree_digest = hashlib.sha256(
        "\n".join(
            sorted(
                metadata_lines
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    subjects = sorted(
        subject_segment_files,
        key=safe_subject_sort_key,
    )

    ordinary_subjects = [
        subject
        for subject
        in subjects
        if subject
        not in SPECIAL_SUBJECT_IDS
        and not subject.startswith(
            "."
        )
    ]

    special_subjects = {
        subject: {
            "windows":
                int(
                    subject_windows[
                        subject
                    ]
                ),

            "segment_files":
                int(
                    subject_segment_files[
                        subject
                    ]
                ),
        }
        for subject
        in sorted(
            SPECIAL_SUBJECT_IDS
        )
        if subject
        in subject_segment_files
    }

    record.update(
        {
            "directory_count":
                directory_count,

            "file_count":
                file_count,

            "total_bytes":
                total_bytes,

            "extension_counts":
                dict(
                    sorted(
                        extension_counts.items()
                    )
                ),

            "common_basenames":
                dict(
                    basename_counts.most_common(
                        30
                    )
                ),

            "npy_shape_counts":
                dict(
                    sorted(
                        npy_shapes.items()
                    )
                ),

            "npy_dtype_counts":
                dict(
                    sorted(
                        npy_dtypes.items()
                    )
                ),

            "segment_file_count":
                len(
                    segment_files
                ),

            "subject_count_all":
                len(
                    subjects
                ),

            "subject_count_ordinary":
                len(
                    ordinary_subjects
                ),

            "subjects_all":
                subjects,

            "subjects_ordinary":
                ordinary_subjects,

            "special_subjects":
                special_subjects,

            "total_windows_from_segments":
                int(
                    sum(
                        subject_windows.values()
                    )
                ),

            "subject_windows":
                {
                    key:
                        int(
                            subject_windows[
                                key
                            ]
                        )
                    for key
                    in sorted(
                        subject_windows,
                        key=safe_subject_sort_key,
                    )
                },

            "subject_segment_file_counts":
                {
                    key:
                        int(
                            subject_segment_files[
                                key
                            ]
                        )
                    for key
                    in sorted(
                        subject_segment_files,
                        key=safe_subject_sort_key,
                    )
                },

            "companion_file_patterns":
                dict(
                    companion_patterns.most_common(
                        50
                    )
                ),

            "tree_metadata_sha256":
                tree_digest,

            "tree_digest_scope":
                (
                    "relative_path + byte_size + mtime_ns; "
                    "not a full dataset content hash"
                ),

            "segment_files":
                segment_files,
        }
    )

    return record


def inspect_additional_candidate(
    name: str,
    root: Path,
) -> dict[str, Any]:
    record = {
        "name":
            name,

        "path":
            str(root),

        "exists":
            root.exists(),
    }

    if not root.exists():
        return record

    file_count = 0
    directory_count = 0
    total_bytes = 0

    top_level = []

    if root.is_dir():
        try:
            top_level = sorted(
                child.name
                for child
                in root.iterdir()
            )[:100]
        except Exception:
            pass

        for current, dirs, files in os.walk(
            root
        ):
            directory_count += 1

            for filename in files:
                path = (
                    Path(current)
                    / filename
                )

                try:
                    stat = path.stat()
                except Exception:
                    continue

                file_count += 1
                total_bytes += int(
                    stat.st_size
                )

    record.update(
        {
            "directory_count":
                directory_count,

            "file_count":
                file_count,

            "total_bytes":
                total_bytes,

            "top_level_entries":
                top_level,
        }
    )

    return record


def scan_source_references(
    name: str,
    root: Path,
) -> dict[str, Any]:
    result = {
        "name":
            name,

        "path":
            str(root),

        "exists":
            root.is_dir(),

        "matches":
            [],
    }

    if not root.is_dir():
        return result

    matches = []

    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "data",
        "datasets",
        "results",
        "outputs",
        "runs",
        "checkpoints",
        "artifacts",
    }

    for current, dirs, files in os.walk(
        root
    ):
        dirs[:] = [
            directory
            for directory
            in dirs
            if directory
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
                size = path.stat().st_size
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

            hits = [
                term
                for term
                in SEARCH_TERMS
                if term
                in text
            ]

            if not hits:
                continue

            try:
                relative = (
                    path.relative_to(
                        root
                    )
                    .as_posix()
                )
            except Exception:
                relative = str(
                    path
                )

            line_hits = []

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if any(
                    term
                    in line
                    for term
                    in hits
                ):
                    line_hits.append(
                        {
                            "line":
                                line_number,

                            "text":
                                line.strip()[
                                    :500
                                ],
                        }
                    )

                if len(
                    line_hits
                ) >= 20:
                    break

            matches.append(
                {
                    "path":
                        relative,

                    "terms":
                        hits,

                    "line_hits":
                        line_hits,
                }
            )

            if len(
                matches
            ) >= 250:
                break

        if len(
            matches
        ) >= 250:
            break

    result[
        "matches"
    ] = matches

    return result


def duplicate_candidates(
    roots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = defaultdict(
        list
    )

    for root in roots:
        for item in root.get(
            "segment_files",
            [],
        ):
            fingerprint = item.get(
                "fingerprint",
                {}
            )

            key = (
                fingerprint.get(
                    "bytes"
                ),
                fingerprint.get(
                    "sha256"
                ),
                tuple(
                    item.get(
                        "array",
                        {}
                    ).get(
                        "shape",
                        [],
                    )
                ),
                item.get(
                    "array",
                    {}
                ).get(
                    "dtype"
                ),
            )

            if (
                key[0]
                is None
                or key[1]
                is None
            ):
                continue

            grouped[
                key
            ].append(
                {
                    "dataset":
                        root[
                            "name"
                        ],

                    "relative_path":
                        item[
                            "relative_path"
                        ],

                    "fingerprint_mode":
                        fingerprint[
                            "mode"
                        ],

                    "authoritative":
                        fingerprint[
                            "authoritative"
                        ],
                }
            )

    result = []

    for key, entries in grouped.items():
        datasets = {
            item[
                "dataset"
            ]
            for item
            in entries
        }

        if (
            len(entries)
            > 1
            and len(
                datasets
            )
            > 1
        ):
            result.append(
                {
                    "bytes":
                        key[0],

                    "fingerprint_sha256":
                        key[1],

                    "shape":
                        list(
                            key[2]
                        ),

                    "dtype":
                        key[3],

                    "entries":
                        entries,

                    "interpretation":
                        (
                            "duplicate candidate only; "
                            "sampled hashes are not authoritative "
                            "for large files"
                        ),
                }
            )

    return result


def git_commit(
    root: Path,
) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    historical = []

    for name, path in HISTORICAL_ROOTS:
        print()
        print(
            "AUDIT ROOT:",
            name,
        )

        record = inspect_segment_root(
            name,
            path,
        )

        historical.append(
            record
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
                "  subjects:",
                record[
                    "subject_count_all"
                ],
            )

            print(
                "  ordinary subjects:",
                record[
                    "subject_count_ordinary"
                ],
            )

            print(
                "  segment files:",
                record[
                    "segment_file_count"
                ],
            )

            print(
                "  windows:",
                record[
                    "total_windows_from_segments"
                ],
            )

            print(
                "  bytes:",
                record[
                    "total_bytes"
                ],
            )

            print(
                "  special subjects:",
                record[
                    "special_subjects"
                ],
            )

    additional = [
        inspect_additional_candidate(
            name,
            path,
        )
        for name, path
        in ADDITIONAL_CANDIDATES
    ]

    source_references = []

    for name, path in SOURCE_TREES:
        print()
        print(
            "SOURCE REFERENCE AUDIT:",
            name,
        )

        record = scan_source_references(
            name,
            path,
        )

        source_references.append(
            record
        )

        print(
            "  exists:",
            record[
                "exists"
            ],
        )

        print(
            "  matching files:",
            len(
                record[
                    "matches"
                ]
            ),
        )

    duplicates = duplicate_candidates(
        historical
    )

    manifest = {
        "schema":
            "crosslayer_phase3a_dataset_provenance_audit_v1",

        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),

        "status":
            "PASS",

        "audit_mode":
            "READ_ONLY",

        "crosslayer_commit":
            git_commit(
                ROOT
            ),

        "historical_roots":
            historical,

        "additional_candidates":
            additional,

        "source_reference_audit":
            source_references,

        "cross_root_duplicate_candidates":
            duplicates,

        "historical_split_reference": {
            "status":
                "REFERENCE_ONLY",

            "historical_logic":
                (
                    "subject-level 20% test, then 10% of "
                    "remaining subjects for validation"
                ),

            "historical_random_state":
                42,

            "auto_accepted_for_crosslayer":
                False,
        },

        "scientific_boundary": {
            "dataset_files_modified":
                False,

            "new_split_created":
                False,

            "calibration_subjects_selected":
                False,

            "test_subjects_selected":
                False,

            "labels_modified":
                False,

            "windows_regenerated":
                False,

            "faults_injected":
                False,

            "quantization_calibration_performed":
                False,
        },

        "next_questions": [
            (
                "Which physical datasets constitute "
                "the primary development population?"
            ),
            (
                "What do path levels below subject encode: "
                "trial, activity, event, direction or sequence?"
            ),
            (
                "Which companion arrays define labels and timing?"
            ),
            (
                "What are the authoritative fall-onset/contact "
                "time semantics for each dataset?"
            ),
            (
                "Are combined trees copies, transformed derivatives "
                "or merged views of the source datasets?"
            ),
            (
                "What should be development/calibration, held-out "
                "confirmation and external-generalization populations?"
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
        "Cross-root duplicate candidates:",
        len(
            duplicates
        ),
    )

    print()
    print(
        "PHASE_3A_DATASET_PROVENANCE_AUDIT=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
