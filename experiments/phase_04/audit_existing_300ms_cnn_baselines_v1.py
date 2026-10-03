from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

FOLD_CONFIG = (
    ROOT
    / "configs/datasets/"
      "primary_300ms_fivefold_v1.json"
)

PHASE3 = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v3.json"
)

PHASE3E = (
    ROOT
    / "manifests/"
      "phase_3e_checkpoint_simulator_lineage_v1.json"
)

PRIMARY_ROOT = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

CHECKPOINT_ROOTS = [
    Path(
        "/mnt/hdd16T/protechto/"
        "checkpoints/CNN/300ms"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master/checkpoints/CNN/300ms"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori/checkpoints/CNN/300ms"
    ),
    Path(
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "IMU_Reliability/checkpoints/CNN/300ms"
    ),
]

CODE_ROOTS = [
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
        "IMU_Reliability"
    ),
]

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_4a_existing_300ms_cnn_baseline_lineage_v1.json"
)

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".log",
}

CHECKPOINT_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def safe_text(path: Path) -> str:
    try:
        if path.stat().st_size > 5_000_000:
            return ""

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        return ""


def flatten_simple(
    value: Any,
    prefix: str = "",
    depth: int = 0,
):
    if depth > 4:
        return {}

    result = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(
                child,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                result[child_prefix] = child

            elif isinstance(
                child,
                (
                    dict,
                    list,
                    tuple,
                ),
            ):
                result.update(
                    flatten_simple(
                        child,
                        child_prefix,
                        depth + 1,
                    )
                )

    elif isinstance(value, (list, tuple)):
        if len(value) <= 100:
            simple = all(
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
                for item in value
            )

            if simple:
                result[prefix] = list(value)

    return result


def canonical_subject(value: Any) -> str:
    text = str(value).strip()

    if text.startswith("UNIVR_"):
        return str(
            int(
                text.split("_")[-1]
            )
        )

    if text.startswith("KFALL_"):
        return str(
            100
            + int(
                text.split("_")[-1]
            )
        )

    match = re.search(
        r"(\d+)$",
        text,
    )

    if match:
        return str(
            int(
                match.group(1)
            )
        )

    return text


def frozen_folds():
    d = json.loads(
        FOLD_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    folds = {}

    for item in d["folds"]:
        fold = int(item["fold"])

        folds[fold] = {
            "train":
                sorted(
                    canonical_subject(v)
                    for v
                    in item[
                        "train"
                    ][
                        "storage_ids"
                    ]
                ),

            "validation":
                sorted(
                    canonical_subject(v)
                    for v
                    in item[
                        "validation"
                    ][
                        "storage_ids"
                    ]
                ),

            "outer_test":
                sorted(
                    canonical_subject(v)
                    for v
                    in item[
                        "outer_test"
                    ][
                        "storage_ids"
                    ]
                ),
        }

    return folds


def checkpoint_fold_id(path: Path) -> int | None:
    candidates = list(
        path.parts
    ) + [
        path.stem,
        path.name,
    ]

    patterns = [
        re.compile(
            r"(?i)^fold[_\- ]?([1-5])$"
        ),
        re.compile(
            r"(?i)fold[_\- ]?([1-5])"
        ),
        re.compile(
            r"(?i)^([1-5])$"
        ),
    ]

    for value in reversed(candidates):
        for pattern in patterns:
            match = pattern.search(
                str(value)
            )

            if match:
                return int(
                    match.group(1)
                )

    return None


def infer_run_root(
    checkpoint: Path,
    fold: int | None,
) -> Path:
    if fold is None:
        return checkpoint.parent

    fold_patterns = [
        re.compile(
            rf"(?i)^fold[_\- ]?{fold}$"
        ),
        re.compile(
            rf"(?i)fold[_\- ]?{fold}"
        ),
    ]

    current = checkpoint.parent

    while current.parent != current:
        if any(
            pattern.search(
                current.name
            )
            for pattern
            in fold_patterns
        ):
            return current.parent

        if current.name == "300ms":
            break

        current = current.parent

    # Common historical form:
    # .../300ms/<run>/fold_1/<checkpoint>
    # otherwise group by first directory below 300ms.
    parts = checkpoint.parts

    try:
        idx = parts.index(
            "300ms"
        )

        if idx + 1 < len(parts):
            return Path(
                *parts[
                    : idx + 2
                ]
            )

    except ValueError:
        pass

    return checkpoint.parent


def discover_checkpoints():
    found = {}

    for root in CHECKPOINT_ROOTS:
        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(
            root
        ):
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    "__pycache__",
                }
            ]

            for filename in files:
                path = (
                    Path(current)
                    / filename
                )

                if (
                    path.suffix.lower()
                    not in CHECKPOINT_SUFFIXES
                ):
                    continue

                low = filename.lower()

                # Prefer actual checkpoint artifacts.
                if not any(
                    token in low
                    for token in [
                        "checkpoint",
                        "best",
                        "model",
                        "fold",
                    ]
                ):
                    continue

                resolved = str(
                    path.resolve()
                )

                found[
                    resolved
                ] = path

    return sorted(
        found.values()
    )


def load_checkpoint_metadata(path: Path):
    record = {
        "path":
            str(
                path.resolve()
            ),

        "bytes":
            int(
                path.stat().st_size
            ),

        "sha256":
            sha256_file(path),
    }

    try:
        import torch

        try:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            obj = torch.load(
                path,
                map_location="cpu",
            )

    except Exception as exc:
        record["load_status"] = (
            "FAILED"
        )

        record["load_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return record

    record["load_status"] = "PASS"

    if isinstance(obj, dict):
        record[
            "top_level_keys"
        ] = sorted(
            str(key)
            for key
            in obj.keys()
        )

        flattened = flatten_simple(
            obj
        )

        interesting = {}

        for key, value in (
            flattened.items()
        ):
            low = key.lower()

            if any(
                token in low
                for token in [
                    "dataset",
                    "data",
                    "window",
                    "overlap",
                    "fold",
                    "model",
                    "seed",
                    "subject",
                    "train",
                    "valid",
                    "test",
                    "lr",
                    "batch",
                    "epoch",
                    "optimizer",
                ]
            ):
                interesting[key] = value

        record[
            "simple_metadata"
        ] = interesting

        state = None

        for state_key in [
            "state_dict",
            "model_state_dict",
            "model",
        ]:
            candidate = obj.get(
                state_key
            )

            if isinstance(
                candidate,
                dict,
            ):
                tensor_like = [
                    value
                    for value
                    in candidate.values()
                    if hasattr(
                        value,
                        "shape",
                    )
                ]

                if tensor_like:
                    state = candidate
                    break

        if state is None:
            tensor_like = [
                value
                for value
                in obj.values()
                if hasattr(
                    value,
                    "shape",
                )
            ]

            if tensor_like:
                state = obj

        if state is not None:
            shapes = {}

            parameter_count = 0

            for name, tensor in (
                state.items()
            ):
                if not hasattr(
                    tensor,
                    "shape",
                ):
                    continue

                shape = [
                    int(v)
                    for v
                    in tensor.shape
                ]

                shapes[
                    str(name)
                ] = shape

                n = 1

                for value in shape:
                    n *= value

                parameter_count += n

            record[
                "state_shapes"
            ] = shapes

            record[
                "state_tensor_element_count"
            ] = parameter_count

            signature_text = json.dumps(
                shapes,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )

            record[
                "state_shape_signature"
            ] = hashlib.sha256(
                signature_text.encode(
                    "utf-8"
                )
            ).hexdigest()

    else:
        record[
            "object_type"
        ] = type(
            obj
        ).__name__

    return record


def metadata_files_for_run(
    run: Path,
):
    records = []

    if not run.is_dir():
        return records

    for current, dirs, files in os.walk(
        run
    ):
        relative_depth = len(
            Path(current)
            .relative_to(run)
            .parts
        )

        if relative_depth > 3:
            dirs[:] = []
            continue

        dirs[:] = [
            d
            for d in dirs
            if d not in {
                "__pycache__",
                ".git",
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

            text = safe_text(path)

            if not text:
                continue

            low = text.lower()

            evidence = {
                "path":
                    str(
                        path.resolve()
                    ),

                "sha256":
                    sha256_file(path),

                "mentions_300ms":
                    (
                        "300ms"
                        in low
                        or "300 ms"
                        in low
                    ),

                "mentions_50_overlap":
                    (
                        "50ov"
                        in low
                        or "50% overlap"
                        in low
                        or "overlap: 50"
                        in low
                        or "overlap=50"
                        in low
                    ),

                "mentions_95_overlap":
                    (
                        "95ov"
                        in low
                        or "95% overlap"
                        in low
                        or "overlap: 95"
                        in low
                        or "overlap=95"
                        in low
                    ),

                "mentions_primary_noof":
                    (
                        "univrfall_kfall_noof"
                        in low
                    ),

                "mentions_onfield":
                    (
                        "onfield"
                        in low
                        or "_of"
                        in low
                    ),

                "mentions_primary_segment":
                    (
                        "300ms_50ov_npseg_filt_binary"
                        in low
                    ),
            }

            if any(
                evidence[key]
                for key
                in [
                    "mentions_300ms",
                    "mentions_50_overlap",
                    "mentions_95_overlap",
                    "mentions_primary_noof",
                    "mentions_onfield",
                    "mentions_primary_segment",
                ]
            ):
                records.append(
                    evidence
                )

    return records


def source_inventory():
    result = []

    patterns = [
        re.compile(
            r"class\s+CNN\b"
        ),
        re.compile(
            r"def\s+train\b"
        ),
        re.compile(
            r"KFold\("
        ),
        re.compile(
            r"train_test_split\("
        ),
    ]

    for root in CODE_ROOTS:
        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(
            root
        ):
            dirs[:] = [
                d
                for d in dirs
                if d not in {
                    ".git",
                    "__pycache__",
                    "data",
                    "datasets",
                    "checkpoints",
                    "outputs",
                }
            ]

            depth = len(
                Path(current)
                .relative_to(root)
                .parts
            )

            if depth > 5:
                dirs[:] = []
                continue

            for filename in files:
                if not filename.endswith(
                    ".py"
                ):
                    continue

                path = (
                    Path(current)
                    / filename
                )

                text = safe_text(path)

                if not text:
                    continue

                matched = [
                    pattern.pattern
                    for pattern
                    in patterns
                    if pattern.search(
                        text
                    )
                ]

                if not matched:
                    continue

                contexts = []

                lines = text.splitlines()

                for index, line in enumerate(
                    lines,
                    start=1,
                ):
                    if not any(
                        pattern.search(
                            line
                        )
                        for pattern
                        in patterns
                    ):
                        continue

                    start = max(
                        1,
                        index - 5,
                    )

                    end = min(
                        len(lines),
                        index + 12,
                    )

                    contexts.append(
                        {
                            "line":
                                index,

                            "context":
                                "\n".join(
                                    (
                                        f"{number}: "
                                        f"{lines[number - 1]}"
                                    )
                                    for number
                                    in range(
                                        start,
                                        end + 1,
                                    )
                                ),
                        }
                    )

                result.append(
                    {
                        "path":
                            str(
                                path.resolve()
                            ),

                        "sha256":
                            sha256_file(path),

                        "matches":
                            matched,

                        "contexts":
                            contexts[:20],
                    }
                )

    return result


def classify_run(
    run_record,
):
    reasons = []

    complete = (
        set(
            run_record[
                "folds_present"
            ]
        )
        == {
            1,
            2,
            3,
            4,
            5,
        }
    )

    if not complete:
        reasons.append(
            "INCOMPLETE_FIVEFOLD"
        )

    signatures = set(
        run_record[
            "state_shape_signatures"
        ]
    )

    if len(signatures) > 1:
        reasons.append(
            "FOLD_ARCHITECTURE_SIGNATURE_MISMATCH"
        )

    metadata = (
        run_record[
            "metadata_evidence"
        ]
    )

    has_noof = any(
        item[
            "mentions_primary_noof"
        ]
        for item
        in metadata
    )

    has_primary_segment = any(
        item[
            "mentions_primary_segment"
        ]
        for item
        in metadata
    )

    has_95 = any(
        item[
            "mentions_95_overlap"
        ]
        for item
        in metadata
    )

    has_onfield = any(
        item[
            "mentions_onfield"
        ]
        for item
        in metadata
    )

    if has_95:
        reasons.append(
            "REFERENCES_95_PERCENT_OVERLAP"
        )

    if has_onfield and not has_noof:
        reasons.append(
            "ONFIELD_OR_OF_REFERENCE_PRESENT"
        )

    if complete and (
        has_noof
        or has_primary_segment
    ) and not has_95:
        compatibility = (
            "PROVENANCE_CANDIDATE"
        )

    elif complete:
        compatibility = (
            "COMPLETE_BUT_DATASET_PROVENANCE_UNRESOLVED"
        )

    else:
        compatibility = (
            "NOT_REUSABLE"
        )

    return {
        "compatibility":
            compatibility,

        "reasons":
            reasons,

        "complete_fivefold":
            complete,

        "explicit_noof_reference":
            has_noof,

        "explicit_primary_segment_reference":
            has_primary_segment,

        "explicit_95_overlap_reference":
            has_95,

        "onfield_reference":
            has_onfield,
    }


def main():
    phase3 = json.loads(
        PHASE3.read_text(
            encoding="utf-8"
        )
    )

    assert (
        phase3[
            "freeze_status"
        ]
        == "PHASE_3_PROTOCOL_FROZEN"
    )

    frozen = frozen_folds()

    checkpoints = (
        discover_checkpoints()
    )

    print(
        "Discovered 300-ms CNN checkpoint files:",
        len(checkpoints),
    )

    groups = defaultdict(
        list
    )

    ungrouped = []

    for path in checkpoints:
        fold = checkpoint_fold_id(
            path
        )

        run = infer_run_root(
            path,
            fold,
        )

        record = (
            load_checkpoint_metadata(
                path
            )
        )

        record[
            "fold"
        ] = fold

        record[
            "inferred_run_root"
        ] = str(
            run.resolve()
        )

        if fold is None:
            ungrouped.append(
                record
            )

        else:
            groups[
                str(
                    run.resolve()
                )
            ].append(
                record
            )

    runs = []

    for run_path, records in sorted(
        groups.items()
    ):
        fold_map = defaultdict(
            list
        )

        for item in records:
            fold_map[
                int(
                    item[
                        "fold"
                    ]
                )
            ].append(
                item
            )

        # One representative checkpoint per fold:
        # choose deterministically by filename/path,
        # never by metric.
        representatives = {}

        for fold, values in (
            fold_map.items()
        ):
            ordered = sorted(
                values,
                key=lambda item:
                    (
                        0
                        if "best"
                        in Path(
                            item[
                                "path"
                            ]
                        ).name.lower()
                        else 1,
                        item[
                            "path"
                        ],
                    ),
            )

            representatives[
                fold
            ] = ordered[0]

        signatures = sorted(
            {
                item[
                    "state_shape_signature"
                ]
                for item
                in representatives.values()
                if item.get(
                    "state_shape_signature"
                )
            }
        )

        metadata = (
            metadata_files_for_run(
                Path(
                    run_path
                )
            )
        )

        run_record = {
            "run_root":
                run_path,

            "folds_present":
                sorted(
                    representatives
                ),

            "checkpoint_count":
                len(records),

            "representative_checkpoints":
                {
                    str(fold):
                        item
                    for fold, item
                    in sorted(
                        representatives.items()
                    )
                },

            "state_shape_signatures":
                signatures,

            "metadata_evidence":
                metadata,
        }

        run_record[
            "compatibility_assessment"
        ] = classify_run(
            run_record
        )

        runs.append(
            run_record
        )

    compatibility_counts = Counter(
        run[
            "compatibility_assessment"
        ][
            "compatibility"
        ]
        for run
        in runs
    )

    provenance_candidates = [
        run
        for run
        in runs
        if run[
            "compatibility_assessment"
        ][
            "compatibility"
        ]
        == "PROVENANCE_CANDIDATE"
    ]

    unresolved_complete = [
        run
        for run
        in runs
        if run[
            "compatibility_assessment"
        ][
            "compatibility"
        ]
        == (
            "COMPLETE_BUT_DATASET_PROVENANCE_UNRESOLVED"
        )
    ]

    source = source_inventory()

    # Phase 4A never automatically reuses a checkpoint.
    # Exact reuse requires fold-membership and preprocessing
    # lineage to be proven in the next decision step.
    if provenance_candidates:
        recommendation = (
            "AUDIT_PROVENANCE_CANDIDATES_BEFORE_RETRAINING"
        )
    else:
        recommendation = (
            "TRAIN_NEW_PROSPECTIVE_300MS_FIVEFOLD_BASELINE"
        )

    manifest = {
        "schema":
            (
                "crosslayer_phase4a_existing_"
                "300ms_cnn_baseline_lineage_v1"
            ),

        "generated_utc":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),

        "status":
            "PASS",

        "audit_mode":
            "NO_MODEL_PREDICTIONS",

        "frozen_primary_protocol": {
            "dataset_root":
                str(
                    PRIMARY_ROOT
                ),

            "subjects":
                61,

            "window_ms":
                300,

            "overlap_percent":
                50,

            "folds":
                frozen,
        },

        "checkpoint_search_roots":
            [
                str(path)
                for path
                in CHECKPOINT_ROOTS
            ],

        "checkpoint_file_count":
            len(
                checkpoints
            ),

        "grouped_run_count":
            len(
                runs
            ),

        "ungrouped_checkpoint_count":
            len(
                ungrouped
            ),

        "compatibility_counts":
            dict(
                compatibility_counts
            ),

        "runs":
            runs,

        "ungrouped_checkpoints":
            ungrouped,

        "provenance_candidate_run_count":
            len(
                provenance_candidates
            ),

        "provenance_candidate_run_roots":
            [
                item[
                    "run_root"
                ]
                for item
                in provenance_candidates
            ],

        "complete_but_unresolved_run_count":
            len(
                unresolved_complete
            ),

        "source_code_inventory":
            source,

        "recommendation":
            recommendation,

        "selection_policy": {
            "accuracy_used":
                False,

            "loss_used":
                False,

            "outer_test_predictions_executed":
                False,

            "onfield_predictions_executed":
                False,

            "checkpoint_reuse_automatically_authorized":
                False,

            "reuse_requires":
                [
                    "complete five-fold family",
                    "61-subject primary population",
                    "50-percent-overlap 300-ms preprocessing",
                    "exact frozen fold membership or demonstrably identical generation",
                    "compatible CNN architecture",
                    "compatible feature representation",
                    "checkpoint selection independent of outer-test outcomes",
                ],
        },

        "scientific_boundary": {
            "model_trained":
                False,

            "model_predictions_executed":
                False,

            "outer_test_outcomes_opened":
                False,

            "onfield_outcomes_opened":
                False,

            "int8_calibration_executed":
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
        "GROUPED RUNS:",
        len(
            runs
        ),
    )

    print(
        "COMPATIBILITY COUNTS:",
        dict(
            compatibility_counts
        ),
    )

    print(
        "PROVENANCE CANDIDATES:",
        len(
            provenance_candidates
        ),
    )

    print(
        "COMPLETE BUT UNRESOLVED:",
        len(
            unresolved_complete
        ),
    )

    print()
    print(
        "RECOMMENDATION=",
        recommendation,
        sep="",
    )

    print()
    print(
        "PHASE_4A_EXISTING_BASELINE_LINEAGE_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
