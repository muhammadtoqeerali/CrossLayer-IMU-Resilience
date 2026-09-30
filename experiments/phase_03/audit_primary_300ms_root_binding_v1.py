from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MERGED = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

UNIVR = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_oriented/segments/"
    "300ms_50ov_npseg_filt_binary"
)

KFALL = Path(
    "/mnt/hdd16T/protechto/data/"
    "KFall_oriented/segments/"
    "300ms_50ov_npseg_filt_binary"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3h_primary_300ms_root_binding_v1.json"
)

UNIVR_EXPECTED = {
    str(value)
    for value
    in range(
        9,
        38,
    )
}

KFALL_ORIGINAL_EXPECTED = {
    str(value)
    for value
    in [
        6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 35,
        36, 37, 38,
    ]
}

KFALL_PROCESSED_EXPECTED = {
    str(
        int(value)
        + 100
    )
    for value
    in KFALL_ORIGINAL_EXPECTED
}

PRIMARY_EXPECTED = (
    UNIVR_EXPECTED
    | KFALL_PROCESSED_EXPECTED
)

EXPECTED_LABELS = {
    "Activity",
    "Falling",
}


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
            digest.update(
                chunk
            )

    return digest.hexdigest()


def subjects(
    root: Path,
) -> set[str]:
    return {
        child.name
        for child
        in root.iterdir()
        if (
            child.is_dir()
            and child.name.isdigit()
        )
    }


def relative_data_files(
    root: Path,
) -> list[Path]:
    result = []

    for name in (
        "segments.npy",
        "labels.npy",
    ):
        result.extend(
            path.relative_to(
                root
            )
            for path
            in root.rglob(
                name
            )
        )

    return sorted(
        result,
        key=lambda path:
            path.as_posix(),
    )


def compare_subject_to_source(
    merged_subject: Path,
    source_subject: Path,
) -> dict[str, Any]:

    merged_files = relative_data_files(
        merged_subject
    )

    source_files = relative_data_files(
        source_subject
    )

    merged_set = {
        path.as_posix()
        for path
        in merged_files
    }

    source_set = {
        path.as_posix()
        for path
        in source_files
    }

    if merged_set != source_set:
        return {
            "status":
                "FILE_SET_MISMATCH",

            "merged_only":
                sorted(
                    merged_set
                    - source_set
                )[:100],

            "source_only":
                sorted(
                    source_set
                    - merged_set
                )[:100],
        }

    mismatches = []

    for relative in merged_files:
        left = (
            merged_subject
            / relative
        )

        right = (
            source_subject
            / relative
        )

        if (
            left.stat().st_size
            != right.stat().st_size
        ):
            mismatches.append(
                {
                    "file":
                        relative.as_posix(),

                    "reason":
                        "SIZE_MISMATCH",

                    "merged_bytes":
                        left.stat().st_size,

                    "source_bytes":
                        right.stat().st_size,
                }
            )
            continue

        left_sha = sha256_file(
            left
        )

        right_sha = sha256_file(
            right
        )

        if left_sha != right_sha:
            mismatches.append(
                {
                    "file":
                        relative.as_posix(),

                    "reason":
                        "SHA256_MISMATCH",

                    "merged_sha256":
                        left_sha,

                    "source_sha256":
                        right_sha,
                }
            )

        if len(
            mismatches
        ) >= 100:
            break

    return {
        "status":
            (
                "EXACT"
                if not mismatches
                else "CONTENT_MISMATCH"
            ),

        "data_file_count":
            len(
                merged_files
            ),

        "mismatches":
            mismatches,
    }


def audit_trial_arrays(
    root: Path,
) -> dict[str, Any]:

    trials = 0
    windows = 0

    labels = Counter()

    bad_segment_shapes = []

    bad_label_shapes = []

    length_mismatches = []

    empty_trials = []

    subject_trial_counts = Counter()

    subject_window_counts = Counter()

    subject_label_counts = {}

    for segment_path in sorted(
        root.rglob(
            "segments.npy"
        )
    ):
        trial_dir = (
            segment_path.parent
        )

        label_path = (
            trial_dir
            / "labels.npy"
        )

        if not label_path.is_file():
            raise RuntimeError(
                "Missing labels.npy beside "
                f"{segment_path}"
            )

        relative = (
            trial_dir
            .relative_to(
                root
            )
        )

        if len(
            relative.parts
        ) != 3:
            raise RuntimeError(
                "Unexpected trial layout: "
                f"{relative}"
            )

        subject = (
            relative.parts[0]
        )

        segments = np.load(
            segment_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        y = np.load(
            label_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        trials += 1

        if segments.size == 0:
            count = 0

            if tuple(
                segments.shape
            ) != (0,):
                bad_segment_shapes.append(
                    {
                        "trial":
                            relative.as_posix(),

                        "shape":
                            list(
                                segments.shape
                            ),
                    }
                )

            empty_trials.append(
                relative.as_posix()
            )

        else:
            if (
                segments.ndim
                != 3
                or tuple(
                    segments.shape[1:]
                )
                != (
                    30,
                    9,
                )
            ):
                bad_segment_shapes.append(
                    {
                        "trial":
                            relative.as_posix(),

                        "shape":
                            list(
                                segments.shape
                            ),
                    }
                )

            count = int(
                segments.shape[0]
            )

        if y.ndim != 1:
            bad_label_shapes.append(
                {
                    "trial":
                        relative.as_posix(),

                    "shape":
                        list(
                            y.shape
                        ),
                }
            )

        if int(
            y.shape[0]
        ) != count:
            length_mismatches.append(
                {
                    "trial":
                        relative.as_posix(),

                    "segments":
                        count,

                    "labels":
                        int(
                            y.shape[0]
                        ),
                }
            )

        unique, counts = np.unique(
            y,
            return_counts=True,
        )

        subject_counts = (
            subject_label_counts
            .setdefault(
                subject,
                Counter(),
            )
        )

        for label, label_count in zip(
            unique.tolist(),
            counts.tolist(),
        ):
            label = str(
                label
            )

            label_count = int(
                label_count
            )

            labels[
                label
            ] += label_count

            subject_counts[
                label
            ] += label_count

        windows += count

        subject_trial_counts[
            subject
        ] += 1

        subject_window_counts[
            subject
        ] += count

    if bad_segment_shapes:
        raise RuntimeError(
            "Bad segment shapes: "
            f"{bad_segment_shapes[:10]}"
        )

    if bad_label_shapes:
        raise RuntimeError(
            "Bad label shapes: "
            f"{bad_label_shapes[:10]}"
        )

    if length_mismatches:
        raise RuntimeError(
            "Segment-label mismatches: "
            f"{length_mismatches[:10]}"
        )

    if set(
        labels
    ) != EXPECTED_LABELS:
        raise RuntimeError(
            "Unexpected label set: "
            f"{sorted(labels)}"
        )

    return {
        "trial_count":
            trials,

        "window_count":
            windows,

        "label_counts":
            dict(
                sorted(
                    labels.items()
                )
            ),

        "empty_trial_count":
            len(
                empty_trials
            ),

        "empty_trials":
            empty_trials,

        "subject_trial_counts":
            dict(
                sorted(
                    subject_trial_counts.items(),
                    key=lambda item:
                        int(
                            item[0]
                        ),
                )
            ),

        "subject_window_counts":
            dict(
                sorted(
                    subject_window_counts.items(),
                    key=lambda item:
                        int(
                            item[0]
                        ),
                )
            ),

        "subject_label_counts":
            {
                subject:
                    dict(
                        sorted(
                            counts.items()
                        )
                    )
                for subject, counts
                in sorted(
                    subject_label_counts.items(),
                    key=lambda item:
                        int(
                            item[0]
                        ),
                )
            },
    }


def main() -> None:

    for path in (
        MERGED,
        UNIVR,
        KFALL,
    ):
        if not path.is_dir():
            raise RuntimeError(
                f"Required root missing: {path}"
            )

    merged_subjects = subjects(
        MERGED
    )

    univr_subjects = subjects(
        UNIVR
    )

    kfall_subjects = subjects(
        KFALL
    )

    if (
        univr_subjects
        != UNIVR_EXPECTED
    ):
        raise RuntimeError(
            "UniVR processed subject population "
            "does not match expected 29 subjects"
        )

    if (
        kfall_subjects
        != KFALL_PROCESSED_EXPECTED
    ):
        raise RuntimeError(
            "KFall processed subject population "
            "does not match expected 32 +100 subjects"
        )

    if (
        merged_subjects
        != PRIMARY_EXPECTED
    ):
        raise RuntimeError(
            "Merged primary population is not the exact "
            "61-subject UniVR+KFall union"
        )

    if (
        univr_subjects
        & kfall_subjects
    ):
        raise RuntimeError(
            "Processed UniVR/KFall subject collision"
        )

    print(
        "UniVR subjects:",
        len(
            univr_subjects
        ),
    )

    print(
        "KFall subjects:",
        len(
            kfall_subjects
        ),
    )

    print(
        "Merged subjects:",
        len(
            merged_subjects
        ),
    )

    print()
    print(
        "VERIFYING MERGED ROOT AGAINST SOURCE ROOTS"
    )

    subject_bindings = []

    exact_count = 0

    for subject in sorted(
        merged_subjects,
        key=int,
    ):
        if subject in (
            univr_subjects
        ):
            dataset = "UNIVR"

            original_subject = (
                f"{int(subject):02d}"
            )

            source = (
                UNIVR
                / subject
            )

        elif subject in (
            kfall_subjects
        ):
            dataset = "KFALL"

            original_subject = (
                f"{int(subject) - 100:02d}"
            )

            source = (
                KFALL
                / subject
            )

        else:
            raise RuntimeError(
                f"Unresolved subject {subject}"
            )

        comparison = (
            compare_subject_to_source(
                MERGED / subject,
                source,
            )
        )

        if (
            comparison[
                "status"
            ]
            == "EXACT"
        ):
            exact_count += 1

        subject_bindings.append(
            {
                "processed_subject_id":
                    subject,

                "canonical_subject_id":
                    (
                        f"{dataset}_"
                        f"{original_subject}"
                    ),

                "dataset":
                    dataset,

                "original_subject_id":
                    original_subject,

                "source_processed_root":
                    str(
                        source
                    ),

                "comparison":
                    comparison,
            }
        )

        print(
            f"  {subject:>4} -> "
            f"{dataset}_{original_subject} "
            f"{comparison['status']}"
        )

    if (
        exact_count
        != 61
    ):
        raise RuntimeError(
            "Merged primary dataset is not an exact "
            "file-level copy of all 61 source subjects"
        )

    print()
    print(
        "AUDITING 300-ms ARRAYS"
    )

    arrays = (
        audit_trial_arrays(
            MERGED
        )
    )

    print(
        "Trials:",
        arrays[
            "trial_count"
        ],
    )

    print(
        "Windows:",
        arrays[
            "window_count"
        ],
    )

    print(
        "Labels:",
        arrays[
            "label_counts"
        ],
    )

    print(
        "Empty trials:",
        arrays[
            "empty_trial_count"
        ],
    )

    protocol = {
        "primary_window_ms":
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

        "stored_channels":
            9,

        "task_classes": [
            "Activity",
            "Falling",
        ],

        "primary_datasets": [
            "UniVRFall",
            "KFall",
        ],

        "primary_subject_count":
            61,

        "onfield_in_primary_folds":
            False,

        "overlap_95_primary_protocol":
            False,

        "secondary_window_sensitivity": {
            "window_ms":
                400,

            "samples_per_window":
                40,

            "overlap_percent":
                50,

            "stride_samples":
                20,

            "stride_ms":
                200,
        },
    }

    manifest = {
        "schema":
            "crosslayer_phase3h_primary_300ms_root_binding_v1",

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

        "protocol":
            protocol,

        "roots": {
            "merged_primary":
                str(
                    MERGED
                ),

            "univr_source_segments":
                str(
                    UNIVR
                ),

            "kfall_source_segments":
                str(
                    KFALL
                ),
        },

        "subject_population": {
            "univr_count":
                len(
                    univr_subjects
                ),

            "kfall_count":
                len(
                    kfall_subjects
                ),

            "total_count":
                len(
                    merged_subjects
                ),

            "processed_subject_ids":
                sorted(
                    merged_subjects,
                    key=int,
                ),

            "no_processed_id_collision":
                True,
        },

        "subject_bindings":
            subject_bindings,

        "exact_subject_copy_count":
            exact_count,

        "array_audit":
            arrays,

        "decision": {
            "merged_root_exactly_bound_to_sources":
                True,

            "regenerate_300ms_now":
                False,

            "reason":
                (
                    "Existing 61-subject root is an exact file-level "
                    "copy of the correctly generated 300-ms / "
                    "50-percent-overlap UniVR and KFall roots."
                ),

            "manual_71_or_73_subject_root_primary":
                False,

            "onfield_role":
                (
                    "AUGMENTATION_OR_EXTERNAL_ROBUSTNESS_ONLY"
                ),

            "95_percent_overlap_role":
                "EXCLUDED_FROM_PRIMARY_PROTOCOL",

            "400ms_role":
                (
                    "HISTORICAL_REFERENCE_AND_CONTROLLED_"
                    "WINDOW_DURATION_SENSITIVITY"
                ),
        },

        "scientific_boundary": {
            "data_regenerated":
                False,

            "model_retrained":
                False,

            "checkpoint_selected_by_performance":
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
        "PHASE_3H_PRIMARY_300MS_ROOT_BINDING=PASS"
    )


if __name__ == "__main__":
    main()
