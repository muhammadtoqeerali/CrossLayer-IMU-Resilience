from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PRIMARY = Path(
    "/mnt/hdd16T/protechto/data/"
    "UniVrFall_KFall_NoOF/segments/"
    "300ms_50ov_npseg_filt_binary"
)

EVENT_INDEX = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "HR_LR_Fallings/risk_data_generated/"
    "FALL_EVENT_INDEX_COMBINED_LABELED.csv"
)

UNIVR_SENSOR = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/sensors_data"
)

KFALL_SENSOR = Path(
    "/mnt/hdd16T/protechto/"
    "ThirdPartyDatasets/KFall_oriented/sensors_data"
)

FOLD_CONFIG = (
    ROOT
    / "configs/datasets/"
      "primary_300ms_fivefold_v1.json"
)

PROTOCOL_AUDIT = (
    ROOT
    / "manifests/"
      "phase_3_300ms_primary_protocol_audit_v1.json"
)

PHASE3MR = (
    ROOT
    / "manifests/"
      "phase_3m_label_time_repair_v2.json"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v1.json"
)

FINAL_CONFIG = (
    ROOT
    / "configs/datasets/"
      "phase3_final_protocol_v1.json"
)

CALIBRATION_CONFIG = (
    ROOT
    / "configs/datasets/"
      "int8_calibration_identity_v1.json"
)

WINDOW_MS = 300
FS_HZ = 100
WINDOW_SAMPLES = 30
OVERLAP_PERCENT = 50
STRIDE_SAMPLES = 15
STRIDE_MS = 150
SAFETY_DEADLINE_MS = 150

CALIBRATION_WINDOWS_PER_FOLD = 4096
CALIBRATION_NAMESPACE = (
    "crosslayer-static-ptq-calibration-v1"
)


def normalise_label(
    value: Any,
) -> str:

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    value = str(value).strip()

    if value.lower().startswith(
        "fall"
    ):
        return "Falling"

    return "Activity"


def source_subject(
    dataset: str,
    storage_subject: int,
) -> int:

    if dataset == "UNIVR":
        return storage_subject

    if dataset == "KFALL":
        return storage_subject - 100

    raise ValueError(dataset)


def dataset_for_storage_subject(
    subject: int,
) -> str:

    return (
        "UNIVR"
        if subject < 100
        else "KFALL"
    )


def sensor_indexes():
    pattern = re.compile(
        r"^S(?P<subject>\d+)"
        r"T(?P<task>\d+)"
        r"R(?P<trial>\d+)\.csv$",
        re.IGNORECASE,
    )

    result = {
        "UNIVR": {},
        "KFALL": {},
    }

    for dataset, root in (
        ("UNIVR", UNIVR_SENSOR),
        ("KFALL", KFALL_SENSOR),
    ):
        if not root.is_dir():
            raise RuntimeError(
                f"Missing sensor root {root}"
            )

        for path in root.rglob(
            "*.csv"
        ):
            m = pattern.match(
                path.name
            )

            if not m:
                continue

            key = (
                int(
                    m.group("subject")
                ),
                int(
                    m.group("task")
                ),
                int(
                    m.group("trial")
                ),
            )

            if key in result[dataset]:
                raise RuntimeError(
                    "Duplicate sensor key "
                    f"{dataset} {key}"
                )

            result[dataset][key] = path

    return result


def expected_windows(
    n_samples: int,
) -> int:

    if n_samples < WINDOW_SAMPLES:
        return 0

    return (
        1
        + (
            n_samples
            - WINDOW_SAMPLES
        )
        // STRIDE_SAMPLES
    )


def load_sensor_frame(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(path)

    frame = frame.drop(
        columns=[
            c
            for c in frame.columns
            if str(c).lower().startswith(
                "unnamed"
            )
        ],
        errors="ignore",
    )

    return frame


def audit_all_primary_window_geometry(
    indexes,
):
    trial_count = 0
    window_count = 0

    missing_sensor = []
    count_mismatch = []

    dataset_trials = Counter()
    dataset_windows = Counter()

    for label_path in sorted(
        PRIMARY.rglob(
            "labels.npy"
        )
    ):
        relative = (
            label_path.parent
            .relative_to(PRIMARY)
        )

        if len(relative.parts) != 3:
            raise RuntimeError(
                f"Bad trial path {relative}"
            )

        storage_subject = int(
            relative.parts[0]
        )

        task = int(
            relative.parts[1]
        )

        trial = int(
            relative.parts[2]
        )

        dataset = (
            dataset_for_storage_subject(
                storage_subject
            )
        )

        src_subject = (
            source_subject(
                dataset,
                storage_subject,
            )
        )

        key = (
            src_subject,
            task,
            trial,
        )

        sensor_path = (
            indexes[
                dataset
            ].get(
                key
            )
        )

        if sensor_path is None:
            missing_sensor.append(
                {
                    "dataset":
                        dataset,
                    "storage_subject":
                        storage_subject,
                    "source_subject":
                        src_subject,
                    "task":
                        task,
                    "trial":
                        trial,
                }
            )
            continue

        frame = load_sensor_frame(
            sensor_path
        )

        labels = np.load(
            label_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        actual = int(
            labels.shape[0]
        )

        expected = (
            expected_windows(
                len(frame)
            )
        )

        if actual != expected:
            count_mismatch.append(
                {
                    "dataset":
                        dataset,
                    "storage_subject":
                        storage_subject,
                    "task":
                        task,
                    "trial":
                        trial,
                    "sensor_rows":
                        int(
                            len(frame)
                        ),
                    "expected_windows":
                        expected,
                    "processed_windows":
                        actual,
                    "sensor_path":
                        str(
                            sensor_path
                        ),
                }
            )

        trial_count += 1
        window_count += actual

        dataset_trials[
            dataset
        ] += 1

        dataset_windows[
            dataset
        ] += actual

    return {
        "trial_count":
            trial_count,

        "window_count":
            window_count,

        "dataset_trial_counts":
            dict(dataset_trials),

        "dataset_window_counts":
            dict(dataset_windows),

        "missing_sensor_count":
            len(
                missing_sensor
            ),

        "missing_sensor":
            missing_sensor[:50],

        "window_count_mismatch_count":
            len(
                count_mismatch
            ),

        "window_count_mismatches":
            count_mismatch[:100],

        "window_grid_mapping_candidate":
            {
                "window_start_sample":
                    "window_index * 15",

                "window_last_sample":
                    (
                        "window_index * 15 + 29"
                    ),
            },
    }


def audit_event_frame_clock(
    indexes,
):
    events = pd.read_csv(
        EVENT_INDEX
    )

    if len(events) != 2919:
        raise RuntimeError(
            "Expected 2919 curated events"
        )

    missing_framecounter = []
    event_position_mismatch = []
    discontinuous_framecounter = []
    bad_sampling_rate = []

    dataset_counts = Counter()

    for _, row in events.iterrows():
        dataset = str(
            row[
                "dataset_id"
            ]
        ).upper()

        source = int(
            re.search(
                r"(\d+)$",
                str(
                    row[
                        "source_subject_id"
                    ]
                ),
            ).group(1)
        )

        task = int(
            round(
                float(
                    row[
                        "task_id"
                    ]
                )
            )
        )

        trial = int(
            round(
                float(
                    row[
                        "trial_id"
                    ]
                )
            )
        )

        onset_position = int(
            round(
                float(
                    row[
                        "fall_start_position"
                    ]
                )
            )
        )

        impact_position = int(
            round(
                float(
                    row[
                        "impact_position"
                    ]
                )
            )
        )

        onset_frame = int(
            round(
                float(
                    row[
                        "fall_start_frame"
                    ]
                )
            )
        )

        impact_frame = int(
            round(
                float(
                    row[
                        "impact_frame"
                    ]
                )
            )
        )

        fs = float(
            row[
                "sampling_rate_hz"
            ]
        )

        event_id = str(
            row[
                "event_id"
            ]
        )

        if abs(
            fs - FS_HZ
        ) > 1e-9:
            bad_sampling_rate.append(
                {
                    "event_id":
                        event_id,
                    "sampling_rate_hz":
                        fs,
                }
            )

        sensor_path = (
            indexes[
                dataset
            ][
                (
                    source,
                    task,
                    trial,
                )
            ]
        )

        frame = load_sensor_frame(
            sensor_path
        )

        if (
            "FrameCounter"
            not in frame.columns
        ):
            missing_framecounter.append(
                event_id
            )
            continue

        fc = pd.to_numeric(
            frame[
                "FrameCounter"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if (
            onset_position < 0
            or impact_position < 0
            or onset_position >= len(fc)
            or impact_position >= len(fc)
        ):
            event_position_mismatch.append(
                {
                    "event_id":
                        event_id,
                    "reason":
                        "POSITION_OUT_OF_RANGE",
                }
            )
            continue

        if (
            not np.isfinite(
                fc[
                    onset_position
                ]
            )
            or not np.isfinite(
                fc[
                    impact_position
                ]
            )
        ):
            event_position_mismatch.append(
                {
                    "event_id":
                        event_id,
                    "reason":
                        "NONFINITE_FRAMECOUNTER",
                }
            )
            continue

        observed_onset = int(
            round(
                fc[
                    onset_position
                ]
            )
        )

        observed_impact = int(
            round(
                fc[
                    impact_position
                ]
            )
        )

        if (
            observed_onset
            != onset_frame
            or observed_impact
            != impact_frame
        ):
            event_position_mismatch.append(
                {
                    "event_id":
                        event_id,
                    "annotated_onset":
                        onset_frame,
                    "observed_onset":
                        observed_onset,
                    "annotated_impact":
                        impact_frame,
                    "observed_impact":
                        observed_impact,
                }
            )

        finite = fc[
            np.isfinite(fc)
        ]

        differences = np.diff(
            finite
        )

        bad = np.flatnonzero(
            differences != 1
        )

        if bad.size:
            discontinuous_framecounter.append(
                {
                    "event_id":
                        event_id,
                    "bad_transition_count":
                        int(
                            bad.size
                        ),
                    "first_bad_indices":
                        bad[
                            :20
                        ].astype(
                            int
                        ).tolist(),
                }
            )

        dataset_counts[
            dataset
        ] += 1

    return {
        "event_count":
            int(
                len(events)
            ),

        "dataset_counts":
            dict(
                dataset_counts
            ),

        "missing_framecounter_count":
            len(
                missing_framecounter
            ),

        "missing_framecounter":
            missing_framecounter[:50],

        "event_position_framecounter_mismatch_count":
            len(
                event_position_mismatch
            ),

        "event_position_framecounter_mismatches":
            event_position_mismatch[:50],

        "framecounter_discontinuity_event_count":
            len(
                discontinuous_framecounter
            ),

        "framecounter_discontinuity_events":
            discontinuous_framecounter[:50],

        "non_100hz_event_count":
            len(
                bad_sampling_rate
            ),

        "non_100hz_events":
            bad_sampling_rate[:50],
    }


def source_windowing_evidence():
    prior = json.loads(
        PROTOCOL_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    paths = []

    for item in prior.get(
        "windowing_source_audit",
        []
    ):
        path = Path(
            item[
                "path"
            ]
        )

        if path.is_file():
            paths.append(path)

    if not paths:
        raise RuntimeError(
            "No previously audited windowing source available"
        )

    records = []

    qualified = False

    for path in paths:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        lines = text.splitlines()

        interesting = []

        for number, line in enumerate(
            lines,
            start=1,
        ):
            low = line.lower()

            if any(
                token in low
                for token in [
                    "step_size",
                    "samples_per_window",
                    "range(0",
                    "range (0",
                    "overlap",
                    "segments.append",
                    "segment",
                ]
            ):
                interesting.append(
                    {
                        "line":
                            number,
                        "text":
                            line.strip()[
                                :800
                            ],
                    }
                )

        compact = re.sub(
            r"\s+",
            "",
            text.lower(),
        )

        has_zero_start_range = bool(
            re.search(
                r"range\(0,",
                compact,
            )
        )

        has_step = (
            "step_size"
            in text
        )

        has_window = (
            "samples_per_window"
            in text
        )

        source_qualified = (
            has_zero_start_range
            and has_step
            and has_window
        )

        qualified = (
            qualified
            or source_qualified
        )

        records.append(
            {
                "path":
                    str(
                        path.resolve()
                    ),

                "sha256":
                    hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),

                "zero_start_range_present":
                    has_zero_start_range,

                "step_size_present":
                    has_step,

                "samples_per_window_present":
                    has_window,

                "source_window_grid_qualified":
                    source_qualified,

                "relevant_lines":
                    interesting[:160],
            }
        )

    return {
        "qualified":
            qualified,

        "sources":
            records,
    }


def stable_hash(
    fold: int,
    subject: str,
    task: str,
    trial: str,
    window: int,
) -> str:

    text = (
        f"{CALIBRATION_NAMESPACE}|"
        f"fold={fold}|"
        f"subject={subject}|"
        f"task={task}|"
        f"trial={trial}|"
        f"window={window}"
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def enumerate_subject_windows(
    subject: str,
):
    subject_root = (
        PRIMARY
        / subject
    )

    if not subject_root.is_dir():
        raise RuntimeError(
            f"Missing training subject {subject}"
        )

    for label_path in sorted(
        subject_root.rglob(
            "labels.npy"
        )
    ):
        relative = (
            label_path.parent
            .relative_to(
                subject_root
            )
        )

        if len(relative.parts) != 2:
            raise RuntimeError(
                f"Unexpected trial path {relative}"
            )

        task, trial = (
            relative.parts
        )

        labels = np.load(
            label_path,
            allow_pickle=True,
        )

        for index, raw in enumerate(
            labels
        ):
            yield {
                "subject":
                    subject,

                "task":
                    str(task),

                "trial":
                    str(trial),

                "window_index":
                    int(index),

                "label":
                    normalise_label(
                        raw
                    ),
            }


def calibration_freeze():
    config = json.loads(
        FOLD_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    result = []

    for fold in config[
        "folds"
    ]:
        fold_number = int(
            fold[
                "fold"
            ]
        )

        train_subjects = list(
            fold[
                "train"
            ][
                "storage_ids"
            ]
        )

        validation = set(
            fold[
                "validation"
            ][
                "storage_ids"
            ]
        )

        outer_test = set(
            fold[
                "outer_test"
            ][
                "storage_ids"
            ]
        )

        candidates = []

        best_per_subject = {}

        train_window_count = 0

        for subject in train_subjects:
            for item in (
                enumerate_subject_windows(
                    subject
                )
            ):
                train_window_count += 1

                digest = stable_hash(
                    fold_number,
                    subject,
                    item[
                        "task"
                    ],
                    item[
                        "trial"
                    ],
                    item[
                        "window_index"
                    ],
                )

                candidate = {
                    **item,
                    "selection_sha256":
                        digest,
                }

                candidates.append(
                    candidate
                )

                previous = (
                    best_per_subject.get(
                        subject
                    )
                )

                if (
                    previous is None
                    or digest
                    < previous[
                        "selection_sha256"
                    ]
                ):
                    best_per_subject[
                        subject
                    ] = candidate

        if (
            train_window_count
            < CALIBRATION_WINDOWS_PER_FOLD
        ):
            raise RuntimeError(
                f"Fold {fold_number} has too few training windows"
            )

        selected_by_key = {}

        # Guarantee every training subject is represented once.
        for subject in train_subjects:
            item = best_per_subject[
                subject
            ]

            identity = (
                item[
                    "subject"
                ],
                item[
                    "task"
                ],
                item[
                    "trial"
                ],
                item[
                    "window_index"
                ],
            )

            selected_by_key[
                identity
            ] = item

        # Fill remaining capacity using global stable hash order.
        for item in sorted(
            candidates,
            key=lambda value:
                value[
                    "selection_sha256"
                ],
        ):
            if (
                len(
                    selected_by_key
                )
                >= CALIBRATION_WINDOWS_PER_FOLD
            ):
                break

            identity = (
                item[
                    "subject"
                ],
                item[
                    "task"
                ],
                item[
                    "trial"
                ],
                item[
                    "window_index"
                ],
            )

            selected_by_key.setdefault(
                identity,
                item,
            )

        selected = sorted(
            selected_by_key.values(),
            key=lambda value:
                value[
                    "selection_sha256"
                ],
        )

        if (
            len(selected)
            != CALIBRATION_WINDOWS_PER_FOLD
        ):
            raise RuntimeError(
                f"Fold {fold_number}: calibration size failure"
            )

        selected_subjects = {
            item[
                "subject"
            ]
            for item
            in selected
        }

        if (
            selected_subjects
            != set(
                train_subjects
            )
        ):
            raise RuntimeError(
                f"Fold {fold_number}: not all training subjects represented"
            )

        if (
            selected_subjects
            & validation
        ):
            raise RuntimeError(
                f"Fold {fold_number}: validation leakage"
            )

        if (
            selected_subjects
            & outer_test
        ):
            raise RuntimeError(
                f"Fold {fold_number}: outer-test leakage"
            )

        class_counts = Counter(
            item[
                "label"
            ]
            for item
            in selected
        )

        result.append(
            {
                "fold":
                    fold_number,

                "training_subject_count":
                    len(
                        train_subjects
                    ),

                "training_window_count":
                    train_window_count,

                "calibration_window_count":
                    len(
                        selected
                    ),

                "calibration_subject_count":
                    len(
                        selected_subjects
                    ),

                "class_counts":
                    dict(
                        class_counts
                    ),

                "validation_subject_leakage":
                    False,

                "outer_test_subject_leakage":
                    False,

                "onfield_used":
                    False,

                "selection":
                    selected,
            }
        )

    return result


def main():
    phase3mr = json.loads(
        PHASE3MR.read_text(
            encoding="utf-8"
        )
    )

    if phase3mr[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Phase 3M-R not PASS"
        )

    indexes = (
        sensor_indexes()
    )

    print(
        "AUDITING ALL PRIMARY WINDOW GEOMETRY"
    )

    geometry = (
        audit_all_primary_window_geometry(
            indexes
        )
    )

    print(
        "trials =",
        geometry[
            "trial_count"
        ],
    )

    print(
        "windows =",
        geometry[
            "window_count"
        ],
    )

    print(
        "missing_sensor =",
        geometry[
            "missing_sensor_count"
        ],
    )

    print(
        "window_count_mismatch =",
        geometry[
            "window_count_mismatch_count"
        ],
    )

    print()
    print(
        "AUDITING FALL-EVENT FRAME CLOCK"
    )

    clock = (
        audit_event_frame_clock(
            indexes
        )
    )

    for key in [
        "event_count",
        "dataset_counts",
        "missing_framecounter_count",
        "event_position_framecounter_mismatch_count",
        "framecounter_discontinuity_event_count",
        "non_100hz_event_count",
    ]:
        print(
            key,
            "=",
            clock[
                key
            ],
        )

    print()
    print(
        "AUDITING WINDOWING SOURCE"
    )

    source = (
        source_windowing_evidence()
    )

    print(
        "source_window_grid_qualified =",
        source[
            "qualified"
        ],
    )

    for item in source[
        "sources"
    ]:
        print(
            item[
                "path"
            ],
            "qualified=",
            item[
                "source_window_grid_qualified"
            ],
        )

        for line in (
            item[
                "relevant_lines"
            ][:30]
        ):
            print(
                " ",
                line[
                    "line"
                ],
                line[
                    "text"
                ],
            )

    print()
    print(
        "FREEZING TRAINING-ONLY CALIBRATION IDENTITIES"
    )

    calibration = (
        calibration_freeze()
    )

    for item in calibration:
        print(
            "fold",
            item[
                "fold"
            ],
            "train_subjects=",
            item[
                "training_subject_count"
            ],
            "train_windows=",
            item[
                "training_window_count"
            ],
            "cal_windows=",
            item[
                "calibration_window_count"
            ],
            "cal_subjects=",
            item[
                "calibration_subject_count"
            ],
            "classes=",
            item[
                "class_counts"
            ],
        )

    geometry_pass = (
        geometry[
            "trial_count"
        ]
        == 6309
        and geometry[
            "window_count"
        ]
        == 273830
        and geometry[
            "missing_sensor_count"
        ]
        == 0
        and geometry[
            "window_count_mismatch_count"
        ]
        == 0
    )

    clock_pass = (
        clock[
            "event_count"
        ]
        == 2919
        and clock[
            "dataset_counts"
        ]
        == {
            "KFALL":
                2346,
            "UNIVR":
                573,
        }
        and clock[
            "missing_framecounter_count"
        ]
        == 0
        and clock[
            "event_position_framecounter_mismatch_count"
        ]
        == 0
        and clock[
            "framecounter_discontinuity_event_count"
        ]
        == 0
        and clock[
            "non_100hz_event_count"
        ]
        == 0
    )

    calibration_pass = (
        len(
            calibration
        )
        == 5
        and all(
            item[
                "calibration_window_count"
            ]
            == CALIBRATION_WINDOWS_PER_FOLD
            and item[
                "calibration_subject_count"
            ]
            == item[
                "training_subject_count"
            ]
            and not item[
                "validation_subject_leakage"
            ]
            and not item[
                "outer_test_subject_leakage"
            ]
            and not item[
                "onfield_used"
            ]
            for item
            in calibration
        )
    )

    source_pass = bool(
        source[
            "qualified"
        ]
    )

    status = (
        "PASS"
        if (
            geometry_pass
            and clock_pass
            and calibration_pass
            and source_pass
        )
        else "FAIL"
    )

    timing_contract = {
        "authoritative_sample_clock":
            "oriented_sensor_FrameCounter",

        "sampling_frequency_hz":
            FS_HZ,

        "window_index_mapping": {
            "start_position":
                "15 * window_index",

            "last_observed_position":
                (
                    "15 * window_index + 29"
                ),
        },

        "offline_sensor_only_lead_ms": (
            "(impact_FrameCounter - "
            "decision_window_end_FrameCounter) "
            "* 1000 / 100"
        ),

        "runtime_adjusted_lead_ms": (
            "offline_sensor_only_lead_ms "
            "- measured_runtime_latency_ms"
        ),

        "deadline_margin_ms": (
            "runtime_adjusted_lead_ms - 150"
        ),

        "on_time_definition": (
            "deadline_margin_ms >= 0"
        ),

        "runtime_latency_status":
            "DEFERRED_TO_ACTUAL_MCU_MEASUREMENT",

        "legacy_univr_time_ms_authoritative":
            False,

        "historical_classification_labels_rewritten":
            False,
    }

    manifest = {
        "schema":
            "crosslayer_phase3n_final_protocol_freeze_v1",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),

        "status":
            status,

        "freeze_status":
            (
                "PHASE_3_PROTOCOL_FROZEN"
                if status
                == "PASS"
                else "NOT_FROZEN"
            ),

        "primary_protocol": {
            "window_ms":
                WINDOW_MS,

            "sampling_hz":
                FS_HZ,

            "window_samples":
                WINDOW_SAMPLES,

            "overlap_percent":
                OVERLAP_PERCENT,

            "stride_samples":
                STRIDE_SAMPLES,

            "stride_ms":
                STRIDE_MS,

            "primary_subjects":
                61,

            "external_onfield_subjects":
                10,

            "rejected_onfield_ids": [
                "999",
                "1000",
            ],
        },

        "classification_ground_truth": {
            "source":
                "frozen labels.npy",

            "event_annotations_rewrite_labels":
                False,

            "annotated_fall_events":
                2919,

            "annotated_events_with_falling_window":
                2918,

            "annotated_activity_only_exception":
                "KFALL_106_T27_R05",
        },

        "window_geometry_audit":
            geometry,

        "event_frame_clock_audit":
            clock,

        "windowing_source_evidence":
            source,

        "timing_contract":
            timing_contract,

        "int8_calibration_policy": {
            "calibration_windows_per_fold":
                CALIBRATION_WINDOWS_PER_FOLD,

            "selection_namespace":
                CALIBRATION_NAMESPACE,

            "selection_method":
                (
                    "deterministic SHA256 rank over training "
                    "window identity with mandatory one-window "
                    "coverage for every training subject"
                ),

            "class_balancing":
                False,

            "training_only":
                True,

            "validation_allowed":
                False,

            "outer_test_allowed":
                False,

            "onfield_allowed":
                False,

            "calibration_executed":
                False,

            "folds":
                calibration,
        },

        "phase3_scientific_boundaries": {
            "fivefold_membership_frozen":
                True,

            "classification_labels_frozen":
                True,

            "event_binding_frozen":
                True,

            "timing_protocol_frozen":
                status
                == "PASS",

            "calibration_identity_frozen":
                status
                == "PASS",

            "model_trained_in_phase3":
                False,

            "int8_calibration_executed_in_phase3":
                False,

            "faults_injected_in_phase3":
                False,

            "outer_test_predictions_opened_in_phase3":
                False,

            "onfield_predictions_opened_in_phase3":
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

    FINAL_CONFIG.write_text(
        json.dumps(
            {
                "schema":
                    "crosslayer_phase3_final_protocol_v1",

                "status":
                    (
                        "FROZEN"
                        if status
                        == "PASS"
                        else "NOT_FROZEN"
                    ),

                "primary_protocol":
                    manifest[
                        "primary_protocol"
                    ],

                "classification_ground_truth":
                    manifest[
                        "classification_ground_truth"
                    ],

                "timing_contract":
                    timing_contract,

                "fivefold_config":
                    (
                        "configs/datasets/"
                        "primary_300ms_fivefold_v1.json"
                    ),

                "onfield_config":
                    (
                        "configs/datasets/"
                        "onfield_role_candidate_v1.json"
                    ),

                "event_binding_manifest":
                    (
                        "manifests/"
                        "phase_3m_label_time_repair_v2.json"
                    ),

                "int8_calibration_config":
                    (
                        "configs/datasets/"
                        "int8_calibration_identity_v1.json"
                    ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    CALIBRATION_CONFIG.write_text(
        json.dumps(
            {
                "schema":
                    (
                        "crosslayer_int8_"
                        "calibration_identity_v1"
                    ),

                "status":
                    (
                        "FROZEN"
                        if calibration_pass
                        else "NOT_FROZEN"
                    ),

                "windows_per_fold":
                    CALIBRATION_WINDOWS_PER_FOLD,

                "selection_namespace":
                    CALIBRATION_NAMESPACE,

                "selection_method":
                    (
                        "training-only deterministic "
                        "SHA256-ranked window identities"
                    ),

                "folds":
                    calibration,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "PHASE_3N_WINDOW_GEOMETRY=",
        (
            "PASS"
            if geometry_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_FRAME_CLOCK=",
        (
            "PASS"
            if clock_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_SOURCE_WINDOW_MAPPING=",
        (
            "PASS"
            if source_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_INT8_CALIBRATION_IDENTITY=",
        (
            "PASS"
            if calibration_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_FINAL_STATUS=",
        status,
        sep="",
    )

    if status != "PASS":
        raise RuntimeError(
            "Phase 3N final freeze failed"
        )


if __name__ == "__main__":
    main()
