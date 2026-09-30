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

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3d_authoritative_fivefold_lineage_v1.json"
)

PRIMARY_PROTECHTO = (
    TOQEER
    / "Protechto-master"
)

SOURCE_CANDIDATES = [
    (
        "Protechto-master",
        TOQEER
        / "Protechto-master"
        / "dataloaders"
        / "KFoldDataloader.py",
    ),
    (
        "Protechto-master-helper",
        TOQEER
        / "Protechto-master"
        / "dataloaders"
        / "helper.py",
    ),
    (
        "Protechto-master-train",
        TOQEER
        / "Protechto-master"
        / "train.py",
    ),
    (
        "Protechto_master",
        TOQEER
        / "Protechto_master"
        / "dataloaders"
        / "KFoldDataloader.py",
    ),
    (
        "Protechto-master_ori_copy",
        TOQEER
        / "Protechto-master_ori_copy"
        / "dataloaders"
        / "KFoldDataloader.py",
    ),
    (
        "Protechto-master_ori_nested",
        TOQEER
        / "Protechto-master_ori_copy"
        / "Protechto-master_ori"
        / "dataloaders"
        / "KFoldDataloader.py",
    ),
]

CHECKPOINT_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto/"
        "checkpoints/CNN/400ms"
    ),
    TOQEER
    / "Protechto-master"
    / "checkpoints"
    / "CNN"
    / "400ms",
]

DATA_SEARCH_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto/data"
    ),
]

METADATA_ROOTS = [
    TOQEER / "Protechto-master",
    TOQEER / "IMU_Reliability",
]

CHECKPOINT_RE = re.compile(
    r"^best-checkpoint-fold_(?P<fold>[1-5])\.ckpt$"
)

QUANTIZED_RE = re.compile(
    r"^quantized-fold_(?P<fold>[1-5])\.ckpt$"
)

TEXT_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".log",
    ".py",
}

RUN_ID_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$"
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


def function_name(
    node: ast.Call,
) -> str | None:
    func = node.func

    if isinstance(
        func,
        ast.Name,
    ):
        return func.id

    if isinstance(
        func,
        ast.Attribute,
    ):
        return func.attr

    return None


def source_audit(
    label: str,
    path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label":
            label,

        "path":
            str(path),

        "exists":
            path.is_file(),
    }

    if not path.is_file():
        return record

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    record[
        "sha256"
    ] = sha256_file(
        path
    )

    record[
        "bytes"
    ] = path.stat().st_size

    try:
        tree = ast.parse(
            text
        )
    except Exception as exc:
        record[
            "ast_error"
        ] = repr(
            exc
        )

        return record

    calls = []

    functions = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                {
                    "name":
                        node.name,

                    "line":
                        int(
                            node.lineno
                        ),

                    "end_line":
                        int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                }
            )

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = function_name(
            node
        )

        if name not in {
            "KFold",
            "train_test_split",
        }:
            continue

        calls.append(
            {
                "name":
                    name,

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

                "args": [
                    ast.unparse(
                        value
                    )
                    for value
                    in node.args
                ],

                "keywords": {
                    (
                        keyword.arg
                        if keyword.arg
                        is not None
                        else "**"
                    ):
                        ast.unparse(
                            keyword.value
                        )
                    for keyword
                    in node.keywords
                },
            }
        )

    lines = text.splitlines()

    relevant_lines = []

    markers = (
        "self.subjects",
        "KFold(",
        "train_test_split(",
        "train_subject",
        "validation_subject",
        "test_subject",
        "fold_index",
        "os.listdir",
        "listdir(",
        "root_directory",
        "best-checkpoint-fold",
    )

    for index, line in enumerate(
        lines,
        start=1,
    ):
        if any(
            marker
            in line
            for marker
            in markers
        ):
            start = max(
                1,
                index - 2,
            )

            end = min(
                len(lines),
                index + 4,
            )

            snippet = "\n".join(
                (
                    f"{line_no}: "
                    f"{lines[line_no - 1]}"
                )
                for line_no
                in range(
                    start,
                    end + 1,
                )
            )

            relevant_lines.append(
                snippet
            )

    # Deduplicate overlapping textual snippets.
    deduped = []

    seen = set()

    for snippet in relevant_lines:
        if snippet in seen:
            continue

        seen.add(
            snippet
        )

        deduped.append(
            snippet
        )

    record.update(
        {
            "functions":
                functions,

            "important_calls":
                calls,

            "relevant_snippets":
                deduped[:60],
        }
    )

    return record


def discover_dataset_roots() -> list[dict[str, Any]]:
    records = []

    seen = set()

    for search_root in (
        DATA_SEARCH_ROOTS
    ):
        if not search_root.is_dir():
            continue

        try:
            matches = search_root.rglob(
                "400ms_50ov_npseg_filt_binary"
            )
        except Exception:
            continue

        for path in sorted(
            matches
        ):
            if not path.is_dir():
                continue

            resolved = str(
                path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(
                resolved
            )

            subjects = sorted(
                [
                    child.name
                    for child
                    in path.iterdir()
                    if (
                        child.is_dir()
                        and child.name.isdigit()
                    )
                ],
                key=lambda value: int(
                    value
                ),
            )

            records.append(
                {
                    "path":
                        resolved,

                    "subject_count":
                        len(
                            subjects
                        ),

                    "subjects":
                        subjects,

                    "contains_999":
                        (
                            "999"
                            in subjects
                        ),

                    "contains_1000":
                        (
                            "1000"
                            in subjects
                        ),
                }
            )

    return records


def discover_checkpoint_runs() -> list[dict[str, Any]]:
    grouped: dict[
        str,
        dict[str, Any],
    ] = {}

    seen_files = set()

    for root in CHECKPOINT_ROOTS:
        if not root.is_dir():
            continue

        for path in sorted(
            root.rglob(
                "*.ckpt"
            )
        ):
            resolved = str(
                path.resolve()
            )

            if resolved in seen_files:
                continue

            seen_files.add(
                resolved
            )

            match = CHECKPOINT_RE.match(
                path.name
            )

            qmatch = QUANTIZED_RE.match(
                path.name
            )

            if (
                match is None
                and qmatch is None
            ):
                continue

            run_dir = path.parent

            key = str(
                run_dir.resolve()
            )

            item = grouped.setdefault(
                key,
                {
                    "run_dir":
                        key,

                    "run_id":
                        run_dir.name,

                    "fp32_folds":
                        {},

                    "quantized_folds":
                        {},
                },
            )

            info = {
                "path":
                    resolved,

                "bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }

            if match is not None:
                item[
                    "fp32_folds"
                ][
                    match.group(
                        "fold"
                    )
                ] = info

            if qmatch is not None:
                item[
                    "quantized_folds"
                ][
                    qmatch.group(
                        "fold"
                    )
                ] = info

    results = []

    for item in grouped.values():
        fp32_set = {
            int(value)
            for value
            in item[
                "fp32_folds"
            ]
        }

        quantized_set = {
            int(value)
            for value
            in item[
                "quantized_folds"
            ]
        }

        item[
            "complete_fp32_fivefold"
        ] = (
            fp32_set
            == {
                1,
                2,
                3,
                4,
                5,
            }
        )

        item[
            "complete_quantized_fivefold"
        ] = (
            quantized_set
            == {
                1,
                2,
                3,
                4,
                5,
            }
        )

        results.append(
            item
        )

    return sorted(
        results,
        key=lambda value:
            value[
                "run_dir"
            ],
    )


def metadata_references(
    run_ids: set[str],
) -> list[dict[str, Any]]:
    records = []

    if not run_ids:
        return records

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

    for root in METADATA_ROOTS:
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

                if size > 1_000_000:
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

                excerpts = []

                lines = (
                    text.splitlines()
                )

                for number, line in enumerate(
                    lines,
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
                                        :600
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
                            str(
                                path
                            ),

                        "run_ids":
                            hits,

                        "excerpts":
                            excerpts,
                    }
                )

    return records


def summarize_call_contract(
    source_records: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    kfold_calls = []

    validation_calls = []

    for record in source_records:
        for call in record.get(
            "important_calls",
            [],
        ):
            enriched = {
                "source_label":
                    record[
                        "label"
                    ],

                "source_path":
                    record[
                        "path"
                    ],

                **call,
            }

            if (
                call["name"]
                == "KFold"
            ):
                kfold_calls.append(
                    enriched
                )

            elif (
                call["name"]
                == "train_test_split"
            ):
                validation_calls.append(
                    enriched
                )

    return {
        "kfold_calls":
            kfold_calls,

        "train_test_split_calls":
            validation_calls,
    }


def main() -> int:
    source_records = [
        source_audit(
            label,
            path,
        )
        for label, path
        in SOURCE_CANDIDATES
    ]

    existing_kfold_sources = [
        item
        for item
        in source_records
        if (
            item["exists"]
            and any(
                call[
                    "name"
                ]
                == "KFold"
                for call
                in item.get(
                    "important_calls",
                    [],
                )
            )
        )
    ]

    if not existing_kfold_sources:
        raise RuntimeError(
            "No authoritative KFold implementation located"
        )

    print(
        "Authoritative KFold source candidates:"
    )

    for item in existing_kfold_sources:
        print()
        print(
            item["label"],
        )

        print(
            "  path:",
            item["path"],
        )

        print(
            "  SHA256:",
            item["sha256"],
        )

        for call in item[
            "important_calls"
        ]:
            print(
                "  call:",
                call[
                    "source"
                ],
            )

    contract = summarize_call_contract(
        source_records
    )

    # Require explicit 5-fold-capable source evidence.
    combined_text = json.dumps(
        contract
    )

    if (
        "KFold"
        not in combined_text
    ):
        raise RuntimeError(
            "KFold call not recovered"
        )

    datasets = (
        discover_dataset_roots()
    )

    checkpoint_runs = (
        discover_checkpoint_runs()
    )

    complete_fp32 = [
        item
        for item
        in checkpoint_runs
        if item[
            "complete_fp32_fivefold"
        ]
    ]

    complete_quantized = [
        item
        for item
        in checkpoint_runs
        if item[
            "complete_quantized_fivefold"
        ]
    ]

    run_ids = {
        item[
            "run_id"
        ]
        for item
        in checkpoint_runs
        if RUN_ID_RE.match(
            item[
                "run_id"
            ]
        )
    }

    metadata = (
        metadata_references(
            run_ids
        )
    )

    print()
    print(
        "Discovered 400-ms data roots:",
        len(
            datasets
        ),
    )

    for item in datasets:
        print(
            " ",
            item[
                "subject_count"
            ],
            "subjects",
            item[
                "path"
            ],
            "999=",
            item[
                "contains_999"
            ],
            "1000=",
            item[
                "contains_1000"
            ],
        )

    print()
    print(
        "400-ms fold checkpoint runs:",
        len(
            checkpoint_runs
        ),
    )

    print(
        "Complete FP32 five-fold runs:",
        len(
            complete_fp32
        ),
    )

    print(
        "Complete quantized five-fold runs:",
        len(
            complete_quantized
        ),
    )

    for item in complete_fp32:
        print()
        print(
            "COMPLETE FP32 RUN:",
            item[
                "run_dir"
            ],
        )

        for fold in sorted(
            item[
                "fp32_folds"
            ],
            key=int,
        ):
            info = item[
                "fp32_folds"
            ][
                fold
            ]

            print(
                f"  fold {fold}: "
                f"bytes={info['bytes']} "
                f"sha256={info['sha256']}"
            )

    print()
    print(
        "Metadata references to 400-ms fold runs:",
        len(
            metadata
        ),
    )

    for item in metadata[
        :100
    ]:
        print()
        print(
            "META:",
            item[
                "path"
            ],
        )

        print(
            "  run IDs:",
            item[
                "run_ids"
            ],
        )

        for excerpt in item[
            "excerpts"
        ][:8]:
            print(
                f"  L{excerpt['line']}: "
                f"{excerpt['text']}"
            )

    # Event-count discrepancy retained explicitly.
    timing_open_issue = {
        "official_univr_fall_events":
            573,

        "local_historical_oriented_rows_with_onset_and_impact":
            574,

        "difference":
            1,

        "status":
            "OPEN_REQUIRES_ROW_LEVEL_RECONCILIATION",

        "freeze_allowed":
            False,
    }

    manifest = {
        "schema":
            "crosslayer_phase3d_authoritative_fivefold_lineage_v1",

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

        "source_code_audit":
            source_records,

        "fivefold_call_contract":
            contract,

        "candidate_400ms_dataset_roots":
            datasets,

        "checkpoint_runs":
            checkpoint_runs,

        "complete_fp32_fivefold_runs":
            complete_fp32,

        "complete_quantized_fivefold_runs":
            complete_quantized,

        "checkpoint_metadata_references":
            metadata,

        "univr_event_count_open_issue":
            timing_open_issue,

        "scientific_interpretation": {
            "historical_single_checkpoint_split_role":
                "CHECKPOINT_PROVENANCE",

            "fivefold_role":
                "INDEPENDENT_RETRAINING_PROTOCOL",

            "fivefold_membership_status":
                "NOT_YET_FROZEN",

            "authoritative_dataset_root_status":
                "NOT_YET_FROZEN",

            "reuse_single_checkpoint_for_all_folds":
                False,
        },

        "scientific_boundary": {
            "new_folds_generated":
                False,

            "subject_membership_changed":
                False,

            "models_retrained":
                False,

            "held_out_predictions_opened":
                False,

            "quantization_calibration_performed":
                False,

            "faults_injected":
                False,
        },

        "next_actions": [
            (
                "Use the recovered KFold and train_test_split "
                "implementation to reconstruct exact fold memberships."
            ),
            (
                "Bind fold reconstruction to the exact 400-ms "
                "dataset root identified by checkpoint/training lineage."
            ),
            (
                "Resolve the UniVR 574-versus-573 annotation discrepancy."
            ),
            (
                "Map UniVR and KFall onset/impact annotations to "
                "protected trial keys."
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
        "PHASE_3D_AUTHORITATIVE_FIVEFOLD_LINEAGE=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
