from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
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

SOURCE = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "Protechto-master/preprocessing/windowing.py"
)

UNIVR_SENSOR = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/sensors_data"
)

KFALL_SENSOR = Path(
    "/mnt/hdd16T/protechto/"
    "ThirdPartyDatasets/KFall_oriented/sensors_data"
)

FRAME_CLOCK_MANIFEST = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v1.json"
)

P3MR = (
    ROOT
    / "manifests/"
      "phase_3m_label_time_repair_v2.json"
)

CALIBRATION = (
    ROOT
    / "configs/datasets/"
      "int8_calibration_identity_v1.json"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v3.json"
)

FINAL_CONFIG = (
    ROOT
    / "configs/datasets/"
      "phase3_final_protocol_v1.json"
)

WINDOW = 30
STRIDE = 15
FS_HZ = 100
DEADLINE_MS = 150

EXCEPTION = (
    "KFALL_106_T27_R05"
)


def n_windows(
    length: int,
) -> int:

    if length < WINDOW:
        return 0

    return (
        1
        + (
            length
            - WINDOW
        )
        // STRIDE
    )


def normalize_label(
    value: Any,
) -> str:

    if isinstance(
        value,
        bytes,
    ):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    value = str(
        value
    ).strip().lower()

    if value.startswith(
        "fall"
    ):
        return "Falling"

    return "Activity"


def parse_source_subject(
    value: Any,
) -> int:

    match = re.search(
        r"(\d+)$",
        str(value).strip(),
    )

    if not match:
        raise RuntimeError(
            f"Bad source subject {value!r}"
        )

    return int(
        match.group(1)
    )


def storage_subject(
    dataset: str,
    source_subject: int,
) -> int:

    if dataset == "UNIVR":
        return source_subject

    if dataset == "KFALL":
        return source_subject + 100

    raise RuntimeError(
        dataset
    )


def key(
    dataset: str,
    subject: int,
    task: int,
    trial: int,
):

    return (
        dataset,
        subject,
        task,
        trial,
    )


def verify_source():
    text = SOURCE.read_text(
        encoding="utf-8",
        errors="strict",
    )

    compact = re.sub(
        r"\s+",
        "",
        text,
    )

    required = {
        "activity_piecewise":
            (
                "windowing("
                "x[:start_fall_frame],"
                "y[:start_fall_frame],"
            ),

        "falling_piecewise":
            (
                "windowing("
                "x[start_fall_frame:end_fall_frame],"
                "y[start_fall_frame:end_fall_frame],"
            ),

        "concatenate_x":
            (
                "np.concatenate("
                "[x_activity,x_falling],axis=0)"
            ),

        "concatenate_y":
            (
                "np.concatenate("
                "[y_activity,y_falling],axis=0)"
            ),

        "fallback_guard":
            (
                "file_path.name"
                "notinprocessed_files"
            ),

        "fallback_windowing":
            (
                "x,y=windowing("
                "x,y,window_size,100,overlap,True)"
            ),
    }

    result = {}

    for name, fragment in (
        required.items()
    ):
        result[
            name
        ] = (
            fragment
            in compact
        )

    result[
        "all_required"
    ] = all(
        result.values()
    )

    result[
        "path"
    ] = str(
        SOURCE.resolve()
    )

    result[
        "sha256"
    ] = hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()

    if not result[
        "all_required"
    ]:
        raise RuntimeError(
            (
                "Historical source contract incomplete: "
                f"{result}"
            )
        )

    return result


def build_sensor_index():
    expression = re.compile(
        r"^S(?P<subject>\d+)"
        r"T(?P<task>\d+)"
        r"R(?P<trial>\d+)\.csv$",
        re.IGNORECASE,
    )

    result = {
        "UNIVR": {},
        "KFALL": {},
    }

    for dataset, root in [
        (
            "UNIVR",
            UNIVR_SENSOR,
        ),
        (
            "KFALL",
            KFALL_SENSOR,
        ),
    ]:
        if not root.is_dir():
            raise RuntimeError(
                f"Missing sensor root {root}"
            )

        for path in (
            root.rglob(
                "*.csv"
            )
        ):
            match = expression.match(
                path.name
            )

            if not match:
                continue

            k = (
                int(
                    match.group(
                        "subject"
                    )
                ),
                int(
                    match.group(
                        "task"
                    )
                ),
                int(
                    match.group(
                        "trial"
                    )
                ),
            )

            if k in result[
                dataset
            ]:
                raise RuntimeError(
                    (
                        "Duplicate sensor identity "
                        f"{dataset} {k}"
                    )
                )

            result[
                dataset
            ][
                k
            ] = path

    return result


def sensor_rows(
    path: Path,
) -> int:

    frame = pd.read_csv(
        path
    )

    return int(
        len(frame)
    )


def load_events():
    frame = pd.read_csv(
        EVENT_INDEX
    )

    if len(frame) != 2919:
        raise RuntimeError(
            "Expected 2919 events"
        )

    events = {}

    offsets = Counter()

    for _, row in frame.iterrows():
        dataset = str(
            row[
                "dataset_id"
            ]
        ).upper()

        source = (
            parse_source_subject(
                row[
                    "source_subject_id"
                ]
            )
        )

        subject = storage_subject(
            dataset,
            source,
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

        offsets[
            (
                dataset,
                onset_frame
                - onset_position,
                impact_frame
                - impact_position,
            )
        ] += 1

        k = key(
            dataset,
            subject,
            task,
            trial,
        )

        if k in events:
            raise RuntimeError(
                f"Duplicate event {k}"
            )

        events[
            k
        ] = {
            "event_id":
                str(
                    row[
                        "event_id"
                    ]
                ),

            "dataset":
                dataset,

            "source_subject":
                source,

            "storage_subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            # IMPORTANT:
            # these are the values actually used by
            # the historical Python slicing source.
            "historical_slice_onset":
                onset_frame,

            "historical_slice_impact":
                impact_frame,

            # These remain the canonical physical event
            # positions used by FrameCounter lineage.
            "event_onset_position":
                onset_position,

            "event_impact_position":
                impact_position,
        }

    expected_offsets = {
        (
            "UNIVR",
            0,
            0,
        ):
            573,

        (
            "KFALL",
            1,
            1,
        ):
            2346,
    }

    offset_pass = (
        dict(
            offsets
        )
        == expected_offsets
    )

    return (
        events,
        {
            "observed":
                [
                    {
                        "dataset":
                            item[
                                0
                            ],

                        "onset_frame_minus_position":
                            item[
                                1
                            ],

                        "impact_frame_minus_position":
                            item[
                                2
                            ],

                        "count":
                            count,
                    }
                    for item, count
                    in sorted(
                        offsets.items()
                    )
                ],

            "expected":
                [
                    {
                        "dataset":
                            dataset,

                        "onset_frame_minus_position":
                            onset,

                        "impact_frame_minus_position":
                            impact,

                        "count":
                            count,
                    }
                    for (
                        dataset,
                        onset,
                        impact,
                    ), count
                    in expected_offsets.items()
                ],

            "pass":
                offset_pass,
        },
    )


def expected_piecewise_labels(
    event,
):
    onset_index = event[
        "historical_slice_onset"
    ]

    impact_index = event[
        "historical_slice_impact"
    ]

    if (
        onset_index < 0
        or impact_index
        <= onset_index
    ):
        raise RuntimeError(
            (
                "Invalid historical slice geometry "
                f"{event}"
            )
        )

    activity_n = (
        n_windows(
            onset_index
        )
    )

    falling_n = (
        n_windows(
            impact_index
            - onset_index
        )
    )

    labels = np.array(
        (
            ["Activity"]
            * activity_n
        )
        + (
            ["Falling"]
            * falling_n
        ),
        dtype=object,
    )

    return (
        activity_n,
        falling_n,
        labels,
    )


def audit_routes(
    events,
    sensors,
):
    total_trials = 0
    total_windows = 0

    ordinary_trials = 0
    ordinary_windows = 0

    event_trials = 0

    exact_piecewise = []

    piecewise_mismatches = []

    fallback_qualified = []

    unresolved = []

    mapped_windows = 0

    for label_path in sorted(
        PRIMARY.rglob(
            "labels.npy"
        )
    ):
        relative = (
            label_path.parent
            .relative_to(
                PRIMARY
            )
        )

        if len(
            relative.parts
        ) != 3:
            raise RuntimeError(
                f"Bad trial path {relative}"
            )

        subject = int(
            relative.parts[
                0
            ]
        )

        task = int(
            relative.parts[
                1
            ]
        )

        trial = int(
            relative.parts[
                2
            ]
        )

        dataset = (
            "UNIVR"
            if subject < 100
            else "KFALL"
        )

        labels = np.asarray(
            [
                normalize_label(
                    value
                )
                for value
                in np.load(
                    label_path,
                    allow_pickle=True,
                )
            ],
            dtype=object,
        )

        total_trials += 1

        total_windows += int(
            labels.shape[
                0
            ]
        )

        event = events.get(
            key(
                dataset,
                subject,
                task,
                trial,
            )
        )

        if event is None:
            ordinary_trials += 1

            ordinary_windows += int(
                labels.shape[
                    0
                ]
            )

            # Every stored window in the fallback/full-trial
            # route maps to raw start 15*j.
            mapped_windows += int(
                labels.shape[
                    0
                ]
            )

            continue

        event_trials += 1

        (
            expected_activity,
            expected_falling,
            expected_labels,
        ) = expected_piecewise_labels(
            event
        )

        exact = (
            labels.shape
            == expected_labels.shape
            and np.array_equal(
                labels,
                expected_labels,
            )
        )

        if exact:
            exact_piecewise.append(
                event[
                    "event_id"
                ]
            )

            mapped_windows += int(
                labels.shape[
                    0
                ]
            )

            continue

        actual_activity = int(
            np.sum(
                labels
                == "Activity"
            )
        )

        actual_falling = int(
            np.sum(
                labels
                == "Falling"
            )
        )

        mismatch = {
            "event_id":
                event[
                    "event_id"
                ],

            "dataset":
                dataset,

            "storage_subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "historical_slice_onset":
                event[
                    "historical_slice_onset"
                ],

            "historical_slice_impact":
                event[
                    "historical_slice_impact"
                ],

            "event_onset_position":
                event[
                    "event_onset_position"
                ],

            "event_impact_position":
                event[
                    "event_impact_position"
                ],

            "expected_activity_windows":
                expected_activity,

            "expected_falling_windows":
                expected_falling,

            "actual_activity_windows":
                actual_activity,

            "actual_falling_windows":
                actual_falling,

            "actual_total_windows":
                int(
                    labels.shape[
                        0
                    ]
                ),
        }

        piecewise_mismatches.append(
            mismatch
        )

        # A historical event annotation may not have been
        # available to the preprocessing run. The source has
        # an explicit fallback route for files not in
        # processed_files. Qualify this route empirically.
        source = event[
            "source_subject"
        ]

        sensor_path = (
            sensors[
                dataset
            ].get(
                (
                    source,
                    task,
                    trial,
                )
            )
        )

        if sensor_path is None:
            unresolved.append(
                {
                    **mismatch,
                    "fallback_reason":
                        "SENSOR_FILE_NOT_FOUND",
                }
            )
            continue

        rows = sensor_rows(
            sensor_path
        )

        expected_full = (
            n_windows(
                rows
            )
        )

        all_activity = bool(
            np.all(
                labels
                == "Activity"
            )
        )

        fallback_match = (
            all_activity
            and expected_full
            == int(
                labels.shape[
                    0
                ]
            )
        )

        fallback_record = {
            **mismatch,

            "sensor_file":
                str(
                    sensor_path
                ),

            "sensor_rows":
                rows,

            "expected_full_trial_windows":
                expected_full,

            "all_labels_activity":
                all_activity,

            "full_trial_fallback_match":
                fallback_match,
        }

        if fallback_match:
            fallback_qualified.append(
                fallback_record
            )

            mapped_windows += int(
                labels.shape[
                    0
                ]
            )

        else:
            unresolved.append(
                fallback_record
            )

    return {
        "total_trial_count":
            total_trials,

        "total_window_count":
            total_windows,

        "ordinary_trial_count":
            ordinary_trials,

        "ordinary_window_count":
            ordinary_windows,

        "annotated_event_trial_count":
            event_trials,

        "piecewise_exact_event_count":
            len(
                exact_piecewise
            ),

        "piecewise_mismatch_count":
            len(
                piecewise_mismatches
            ),

        "piecewise_mismatches":
            piecewise_mismatches,

        "full_trial_fallback_qualified_count":
            len(
                fallback_qualified
            ),

        "full_trial_fallback_qualified":
            fallback_qualified,

        "unresolved_route_count":
            len(
                unresolved
            ),

        "unresolved_routes":
            unresolved,

        "mapped_window_count":
            mapped_windows,
    }


def verify_frame_clock():
    v1 = json.loads(
        FRAME_CLOCK_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    clock = v1[
        "event_frame_clock_audit"
    ]

    passed = (
        clock[
            "event_count"
        ]
        == 2919
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

    return (
        passed,
        clock,
    )


def verify_calibration():
    config = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    folds = config[
        "folds"
    ]

    passed = (
        config[
            "status"
        ]
        == "FROZEN"
        and len(
            folds
        )
        == 5
        and all(
            fold[
                "calibration_window_count"
            ]
            == 4096
            and fold[
                "calibration_subject_count"
            ]
            == fold[
                "training_subject_count"
            ]
            and fold[
                "validation_subject_leakage"
            ]
            is False
            and fold[
                "outer_test_subject_leakage"
            ]
            is False
            and fold[
                "onfield_used"
            ]
            is False
            for fold
            in folds
        )
    )

    return (
        passed,
        config,
    )


def main():
    phase3mr = json.loads(
        P3MR.read_text(
            encoding="utf-8"
        )
    )

    assert (
        phase3mr[
            "status"
        ]
        == "PASS"
    )

    source_contract = (
        verify_source()
    )

    sensors = (
        build_sensor_index()
    )

    events, conventions = (
        load_events()
    )

    routes = audit_routes(
        events,
        sensors,
    )

    clock_pass, clock = (
        verify_frame_clock()
    )

    calibration_pass, calibration = (
        verify_calibration()
    )

    mismatch_ids = [
        item[
            "event_id"
        ]
        for item
        in routes[
            "piecewise_mismatches"
        ]
    ]

    fallback_ids = [
        item[
            "event_id"
        ]
        for item
        in routes[
            "full_trial_fallback_qualified"
        ]
    ]

    route_pass = (
        routes[
            "total_trial_count"
        ]
        == 6309

        and routes[
            "total_window_count"
        ]
        == 273830

        and routes[
            "annotated_event_trial_count"
        ]
        == 2919

        and routes[
            "piecewise_exact_event_count"
        ]
        == 2918

        and routes[
            "piecewise_mismatch_count"
        ]
        == 1

        and mismatch_ids
        == [
            EXCEPTION
        ]

        and routes[
            "full_trial_fallback_qualified_count"
        ]
        == 1

        and fallback_ids
        == [
            EXCEPTION
        ]

        and routes[
            "unresolved_route_count"
        ]
        == 0

        and routes[
            "mapped_window_count"
        ]
        == 273830
    )

    convention_pass = bool(
        conventions[
            "pass"
        ]
    )

    source_pass = bool(
        source_contract[
            "all_required"
        ]
    )

    status = (
        "PASS"
        if (
            route_pass
            and convention_pass
            and source_pass
            and clock_pass
            and calibration_pass
        )
        else "FAIL"
    )

    timing_contract = {
        "authoritative_physical_event_clock":
            "oriented sensor FrameCounter",

        "sampling_frequency_hz":
            100,

        "historical_slice_index_rule":
            (
                "Use annotation frame value directly "
                "as Python array index, exactly as in "
                "the recovered preprocessing source."
            ),

        "dataset_index_conventions": {
            "UNIVR":
                (
                    "annotation frame == curated "
                    "zero-based event position"
                ),

            "KFALL":
                (
                    "annotation frame == curated "
                    "zero-based event position + 1"
                ),
        },

        "ordinary_or_fallback_full_trial_window": {
            "raw_start_index":
                "15 * stored_window_index",

            "raw_last_index":
                (
                    "15 * stored_window_index + 29"
                ),
        },

        "annotated_piecewise_trial": {
            "activity_window_count":
                (
                    "n_windows(fall_start_frame)"
                ),

            "activity_raw_start_index":
                (
                    "15 * activity_local_index"
                ),

            "falling_window_count":
                (
                    "n_windows("
                    "fall_impact_frame - "
                    "fall_start_frame)"
                ),

            "falling_raw_start_index":
                (
                    "fall_start_frame + "
                    "15 * falling_local_index"
                ),

            "raw_last_index":
                (
                    "raw_start_index + 29"
                ),
        },

        "physical_event_positions":
            (
                "Use curated zero-based positions / "
                "FrameCounter correspondence, not the "
                "historical preprocessing slice index."
            ),

        "sensor_only_lead_ms":
            (
                "(impact_FrameCounter - "
                "decision_window_end_FrameCounter) * 10"
            ),

        "runtime_adjusted_lead_ms":
            (
                "sensor_only_lead_ms - "
                "measured_runtime_latency_ms"
            ),

        "deadline_margin_ms":
            (
                "runtime_adjusted_lead_ms - 150"
            ),

        "deadline_met":
            "deadline_margin_ms >= 0",

        "runtime_latency":
            "DEFERRED_TO_ACTUAL_MCU_MEASUREMENT",
    }

    exception_record = (
        routes[
            "full_trial_fallback_qualified"
        ][
            0
        ]
        if routes[
            "full_trial_fallback_qualified"
        ]
        else None
    )

    manifest = {
        "schema":
            (
                "crosslayer_phase3n_final_"
                "protocol_freeze_v3"
            ),

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

        "supersedes": [
            {
                "manifest":
                    (
                        "manifests/"
                        "phase_3n_final_protocol_freeze_v1.json"
                    ),

                "reason":
                    (
                        "v1 assumed continuous whole-trial "
                        "windowing for annotated fall trials"
                    ),
            },
            {
                "manifest":
                    (
                        "manifests/"
                        "phase_3n_final_protocol_freeze_v2.json"
                    ),

                "reason":
                    (
                        "v2 used curated zero-based event "
                        "positions as historical Python "
                        "slice indices instead of source "
                        "annotation frame values"
                    ),
            },
        ],

        "primary_protocol": {
            "window_ms":
                300,

            "sampling_hz":
                100,

            "window_samples":
                30,

            "overlap_percent":
                50,

            "stride_samples":
                15,

            "stride_ms":
                150,

            "subjects":
                61,

            "trials":
                6309,

            "windows":
                273830,

            "external_onfield_subjects":
                10,

            "rejected_onfield_ids": [
                "999",
                "1000",
            ],
        },

        "historical_preprocessing_source":
            source_contract,

        "event_frame_position_conventions":
            conventions,

        "route_reconstruction":
            routes,

        "historical_route_exception": {
            "event_id":
                EXCEPTION,

            "classification_labels":
                "retained unchanged",

            "current_event_annotation":
                "retained",

            "piecewise_route_match":
                False,

            "full_trial_fallback_route_match":
                (
                    exception_record
                    is not None
                    and exception_record[
                        "full_trial_fallback_match"
                    ]
                ),

            "interpretation":
                (
                    "The frozen trial is compatible with "
                    "the recovered source's whole-trial "
                    "fallback route rather than the "
                    "annotated piecewise route. This is "
                    "recorded as historical lineage "
                    "discordance and is not relabeled."
                ),
        },

        "classification_ground_truth": {
            "source":
                "frozen labels.npy",

            "labels_rewritten":
                False,

            "annotated_current_fall_events":
                2919,

            "annotated_trials_with_falling_label":
                2918,

            "annotated_activity_only_historical_trial":
                EXCEPTION,
        },

        "event_frame_clock_audit":
            clock,

        "timing_contract":
            timing_contract,

        "int8_calibration_policy":
            calibration,

        "scientific_boundary": {
            "fivefold_membership_changed":
                False,

            "classification_labels_changed":
                False,

            "model_trained":
                False,

            "predictions_executed":
                False,

            "int8_calibration_executed":
                False,

            "faults_injected":
                False,

            "outer_test_model_outcomes_opened":
                False,

            "onfield_model_outcomes_opened":
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
                    "crosslayer_phase3_final_protocol_v3",

                "status":
                    (
                        "FROZEN"
                        if status
                        == "PASS"
                        else "NOT_FROZEN"
                    ),

                "manifest":
                    (
                        "manifests/"
                        "phase_3n_final_protocol_freeze_v3.json"
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

    print()
    print(
        "SOURCE_CONTRACT=",
        (
            "PASS"
            if source_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "FRAME_POSITION_CONVENTIONS=",
        (
            "PASS"
            if convention_pass
            else "FAIL"
        ),
        sep="",
    )

    print()
    print(
        "ROUTE RECONSTRUCTION"
    )

    for name in [
        "total_trial_count",
        "total_window_count",
        "ordinary_trial_count",
        "annotated_event_trial_count",
        "piecewise_exact_event_count",
        "piecewise_mismatch_count",
        "full_trial_fallback_qualified_count",
        "unresolved_route_count",
        "mapped_window_count",
    ]:
        print(
            name,
            "=",
            routes[
                name
            ],
        )

    print()
    print(
        "PIECEWISE MISMATCHES"
    )

    for item in routes[
        "piecewise_mismatches"
    ]:
        print(
            json.dumps(
                item,
                indent=2,
            )
        )

    print()
    print(
        "FULL-TRIAL FALLBACK QUALIFICATION"
    )

    for item in routes[
        "full_trial_fallback_qualified"
    ]:
        print(
            json.dumps(
                item,
                indent=2,
            )
        )

    print()
    print(
        "FRAME_CLOCK=",
        (
            "PASS"
            if clock_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "INT8_CALIBRATION_IDENTITIES=",
        (
            "PASS"
            if calibration_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_R2_FINAL_STATUS=",
        status,
        sep="",
    )

    if status != "PASS":
        raise RuntimeError(
            "Phase 3N-R2 final freeze failed"
        )


if __name__ == "__main__":
    main()
