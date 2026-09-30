from __future__ import annotations

import ast
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

V1 = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v1.json"
)

P3MR = (
    ROOT
    / "manifests/"
      "phase_3m_label_time_repair_v2.json"
)

FOLDS = (
    ROOT
    / "configs/datasets/"
      "primary_300ms_fivefold_v1.json"
)

CALIBRATION = (
    ROOT
    / "configs/datasets/"
      "int8_calibration_identity_v1.json"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3n_final_protocol_freeze_v2.json"
)

FINAL_CONFIG = (
    ROOT
    / "configs/datasets/"
      "phase3_final_protocol_v1.json"
)

WINDOW_SAMPLES = 30
STRIDE_SAMPLES = 15
FS_HZ = 100
DEADLINE_MS = 150

KNOWN_DISCORDANT_EVENT = (
    "KFALL_106_T27_R05"
)


def normalize_label(
    value: Any,
) -> str:

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    text = str(value).strip().lower()

    if text.startswith("fall"):
        return "Falling"

    return "Activity"


def n_windows(
    length: int,
) -> int:

    if length < WINDOW_SAMPLES:
        return 0

    return (
        1
        + (
            length
            - WINDOW_SAMPLES
        )
        // STRIDE_SAMPLES
    )


def source_subject_number(
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


def processed_subject(
    dataset: str,
    source_subject: int,
) -> int:

    if dataset == "UNIVR":
        return source_subject

    if dataset == "KFALL":
        return source_subject + 100

    raise RuntimeError(dataset)


def event_key(
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


def verify_source_contract():
    text = SOURCE.read_text(
        encoding="utf-8",
        errors="strict",
    )

    tree = ast.parse(text)

    create_dataset = None

    for node in tree.body:
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "create_dataset"
        ):
            create_dataset = node
            break

    if create_dataset is None:
        raise RuntimeError(
            "create_dataset not found"
        )

    function_text = ast.get_source_segment(
        text,
        create_dataset,
    )

    if function_text is None:
        raise RuntimeError(
            "Cannot recover create_dataset source"
        )

    compact = re.sub(
        r"\s+",
        "",
        function_text,
    )

    required = [
        (
            "x_activity,y_activity="
            "windowing(x[:start_fall_frame],"
        ),
        (
            "x_falling,y_falling="
            "windowing("
            "x[start_fall_frame:end_fall_frame],"
        ),
    ]

    for fragment in required:
        if fragment not in compact:
            raise RuntimeError(
                "Historical piecewise segmentation "
                "source contract not found: "
                f"{fragment}"
            )

    # Recover concatenate calls and establish storage order.
    concatenate_calls = []

    for node in ast.walk(
        create_dataset
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        is_concat = (
            (
                isinstance(
                    func,
                    ast.Attribute,
                )
                and func.attr
                == "concatenate"
            )
            or (
                isinstance(
                    func,
                    ast.Name,
                )
                and func.id
                == "concatenate"
            )
        )

        if not is_concat:
            continue

        names = []

        for subnode in ast.walk(
            node
        ):
            if isinstance(
                subnode,
                ast.Name,
            ):
                names.append(
                    subnode.id
                )

        concatenate_calls.append(
            names
        )

    activity_before_falling_x = any(
        (
            "x_activity"
            in names
            and "x_falling"
            in names
            and names.index(
                "x_activity"
            )
            < names.index(
                "x_falling"
            )
        )
        for names
        in concatenate_calls
    )

    activity_before_falling_y = any(
        (
            "y_activity"
            in names
            and "y_falling"
            in names
            and names.index(
                "y_activity"
            )
            < names.index(
                "y_falling"
            )
        )
        for names
        in concatenate_calls
    )

    # If AST ordering is obscured by syntax, the empirical sequence audit
    # below remains mandatory. Do not silently claim source order.
    source_order_qualified = (
        activity_before_falling_x
        and activity_before_falling_y
    )

    return {
        "source":
            str(
                SOURCE.resolve()
            ),

        "sha256":
            hashlib.sha256(
                SOURCE.read_bytes()
            ).hexdigest(),

        "piecewise_activity_call":
            True,

        "piecewise_falling_call":
            True,

        "activity_before_falling_x_ast":
            activity_before_falling_x,

        "activity_before_falling_y_ast":
            activity_before_falling_y,

        "source_storage_order_ast_qualified":
            source_order_qualified,

        "concatenate_name_sets":
            concatenate_calls,
    }


def load_events():
    frame = pd.read_csv(
        EVENT_INDEX
    )

    if len(frame) != 2919:
        raise RuntimeError(
            "Expected 2919 events"
        )

    result = {}

    for _, row in frame.iterrows():
        dataset = str(
            row["dataset_id"]
        ).upper()

        source = (
            source_subject_number(
                row[
                    "source_subject_id"
                ]
            )
        )

        subject = processed_subject(
            dataset,
            source,
        )

        task = int(
            round(
                float(
                    row["task_id"]
                )
            )
        )

        trial = int(
            round(
                float(
                    row["trial_id"]
                )
            )
        )

        key = event_key(
            dataset,
            subject,
            task,
            trial,
        )

        if key in result:
            raise RuntimeError(
                f"Duplicate event key {key}"
            )

        result[key] = {
            "event_id":
                str(
                    row["event_id"]
                ),

            "dataset":
                dataset,

            "subject":
                subject,

            "source_subject":
                source,

            "task":
                task,

            "trial":
                trial,

            "onset_position":
                int(
                    round(
                        float(
                            row[
                                "fall_start_position"
                            ]
                        )
                    )
                ),

            "impact_position":
                int(
                    round(
                        float(
                            row[
                                "impact_position"
                            ]
                        )
                    )
                ),

            "onset_frame":
                int(
                    round(
                        float(
                            row[
                                "fall_start_frame"
                            ]
                        )
                    )
                ),

            "impact_frame":
                int(
                    round(
                        float(
                            row[
                                "impact_frame"
                            ]
                        )
                    )
                ),
        }

    return result


def reconstruct_trial_windows(
    event,
    labels,
):
    onset = event[
        "onset_position"
    ]

    impact = event[
        "impact_position"
    ]

    expected_activity = (
        n_windows(
            onset
        )
    )

    expected_falling = (
        n_windows(
            impact
            - onset
        )
    )

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

    expected_total = (
        expected_activity
        + expected_falling
    )

    label_order_expected = np.array(
        (
            ["Activity"]
            * expected_activity
        )
        + (
            ["Falling"]
            * expected_falling
        ),
        dtype=object,
    )

    exact_label_sequence = (
        labels.shape
        == label_order_expected.shape
        and np.array_equal(
            labels,
            label_order_expected,
        )
    )

    window_map = []

    # Activity prefix.
    for local_index in range(
        min(
            expected_activity,
            labels.shape[0],
        )
    ):
        start = (
            local_index
            * STRIDE_SAMPLES
        )

        window_map.append(
            {
                "stored_window_index":
                    len(
                        window_map
                    ),

                "region":
                    "Activity",

                "region_local_index":
                    local_index,

                "raw_start_position":
                    start,

                "raw_last_position":
                    start
                    + WINDOW_SAMPLES
                    - 1,
            }
        )

    # Falling region restarts its local window grid at onset.
    available_after_activity = max(
        0,
        labels.shape[0]
        - expected_activity,
    )

    for local_index in range(
        min(
            expected_falling,
            available_after_activity,
        )
    ):
        start = (
            onset
            + local_index
            * STRIDE_SAMPLES
        )

        window_map.append(
            {
                "stored_window_index":
                    len(
                        window_map
                    ),

                "region":
                    "Falling",

                "region_local_index":
                    local_index,

                "raw_start_position":
                    start,

                "raw_last_position":
                    start
                    + WINDOW_SAMPLES
                    - 1,
            }
        )

    return {
        "expected_activity_windows":
            expected_activity,

        "expected_falling_windows":
            expected_falling,

        "expected_total_windows":
            expected_total,

        "actual_activity_windows":
            actual_activity,

        "actual_falling_windows":
            actual_falling,

        "actual_total_windows":
            int(
                labels.shape[0]
            ),

        "exact_label_sequence":
            exact_label_sequence,

        "window_map":
            window_map,
    }


def audit_piecewise_geometry():
    events = load_events()

    annotated_trials = 0
    ordinary_trials = 0

    exact_piecewise = 0

    annotated_mismatches = []

    ordinary_count_mismatches = []

    timing_qualified_event_count = 0

    timing_excluded = []

    mapped_windows = 0

    classification_windows = 0

    for label_path in sorted(
        PRIMARY.rglob(
            "labels.npy"
        )
    ):
        rel = (
            label_path.parent
            .relative_to(
                PRIMARY
            )
        )

        if len(rel.parts) != 3:
            raise RuntimeError(
                f"Bad path {rel}"
            )

        subject = int(
            rel.parts[0]
        )

        task = int(
            rel.parts[1]
        )

        trial = int(
            rel.parts[2]
        )

        dataset = (
            "UNIVR"
            if subject < 100
            else "KFALL"
        )

        key = event_key(
            dataset,
            subject,
            task,
            trial,
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

        classification_windows += int(
            labels.shape[0]
        )

        event = events.get(
            key
        )

        if event is not None:
            annotated_trials += 1

            audit = (
                reconstruct_trial_windows(
                    event,
                    labels,
                )
            )

            event_id = event[
                "event_id"
            ]

            if audit[
                "exact_label_sequence"
            ]:
                exact_piecewise += 1

                timing_qualified_event_count += 1

                mapped_windows += len(
                    audit[
                        "window_map"
                    ]
                )

            else:
                annotated_mismatches.append(
                    {
                        "event_id":
                            event_id,

                        "dataset":
                            dataset,

                        "subject":
                            subject,

                        "task":
                            task,

                        "trial":
                            trial,

                        "onset_position":
                            event[
                                "onset_position"
                            ],

                        "impact_position":
                            event[
                                "impact_position"
                            ],

                        **{
                            key:
                                value
                            for key, value
                            in audit.items()
                            if key
                            != "window_map"
                        },
                    }
                )

                timing_excluded.append(
                    {
                        "event_id":
                            event_id,

                        "reason":
                            (
                                "CURRENT_EVENT_ANNOTATION_DOES_NOT_"
                                "RECONSTRUCT_FROZEN_PROCESSED_LABEL_"
                                "SEQUENCE"
                            ),
                    }
                )

            continue

        # Non-event trial. Historical source windows the entire trial.
        ordinary_trials += 1

        segment_path = (
            label_path.parent
            / "segments.npy"
        )

        segments = np.load(
            segment_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if (
            int(
                segments.shape[0]
            )
            != int(
                labels.shape[0]
            )
        ):
            ordinary_count_mismatches.append(
                {
                    "path":
                        rel.as_posix(),

                    "labels":
                        int(
                            labels.shape[0]
                        ),

                    "segments":
                        int(
                            segments.shape[0]
                        ),
                }
            )

        mapped_windows += int(
            labels.shape[0]
        )

    return {
        "total_trial_count":
            (
                annotated_trials
                + ordinary_trials
            ),

        "classification_window_count":
            classification_windows,

        "annotated_event_trial_count":
            annotated_trials,

        "ordinary_trial_count":
            ordinary_trials,

        "exact_piecewise_event_trial_count":
            exact_piecewise,

        "piecewise_event_mismatch_count":
            len(
                annotated_mismatches
            ),

        "piecewise_event_mismatches":
            annotated_mismatches,

        "timing_qualified_event_count":
            timing_qualified_event_count,

        "timing_excluded_event_count":
            len(
                timing_excluded
            ),

        "timing_excluded_events":
            timing_excluded,

        "ordinary_segment_label_count_mismatch_count":
            len(
                ordinary_count_mismatches
            ),

        "ordinary_segment_label_count_mismatches":
            ordinary_count_mismatches[
                :100
            ],

        "mapped_window_count":
            mapped_windows,
    }


def verify_v1_frame_clock():
    v1 = json.loads(
        V1.read_text(
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
            "dataset_counts"
        ]
        == {
            "KFALL": 2346,
            "UNIVR": 573,
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

    return (
        passed,
        clock,
    )


def verify_calibration():
    calibration = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    folds = calibration[
        "folds"
    ]

    passed = (
        calibration[
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
            and not fold[
                "validation_subject_leakage"
            ]
            and not fold[
                "outer_test_subject_leakage"
            ]
            and not fold[
                "onfield_used"
            ]
            for fold
            in folds
        )
    )

    return (
        passed,
        calibration,
    )


def main():
    mr = json.loads(
        P3MR.read_text(
            encoding="utf-8"
        )
    )

    assert mr["status"] == "PASS"

    source = (
        verify_source_contract()
    )

    print(
        "SOURCE_PIECEWISE_CALLS=PASS"
    )

    print(
        "SOURCE_STORAGE_ORDER_AST_QUALIFIED=",
        source[
            "source_storage_order_ast_qualified"
        ],
        sep="",
    )

    geometry = (
        audit_piecewise_geometry()
    )

    print()
    print(
        "PIECEWISE GEOMETRY"
    )

    for name in [
        "total_trial_count",
        "classification_window_count",
        "annotated_event_trial_count",
        "ordinary_trial_count",
        "exact_piecewise_event_trial_count",
        "piecewise_event_mismatch_count",
        "timing_qualified_event_count",
        "timing_excluded_event_count",
        "ordinary_segment_label_count_mismatch_count",
        "mapped_window_count",
    ]:
        print(
            name,
            "=",
            geometry[
                name
            ],
        )

    if geometry[
        "piecewise_event_mismatch_count"
    ]:
        print()
        print(
            "PIECEWISE EVENT MISMATCHES"
        )

        for item in geometry[
            "piecewise_event_mismatches"
        ]:
            print(
                json.dumps(
                    item,
                    indent=2,
                )
            )

    clock_pass, clock = (
        verify_v1_frame_clock()
    )

    calibration_pass, calibration = (
        verify_calibration()
    )

    # Scientific expectation:
    # all annotated event trials except the known version-discordant
    # KFall event must exactly follow recovered historical piecewise
    # preprocessing.
    mismatch_ids = [
        item[
            "event_id"
        ]
        for item
        in geometry[
            "piecewise_event_mismatches"
        ]
    ]

    piecewise_pass = (
        geometry[
            "total_trial_count"
        ]
        == 6309
        and geometry[
            "classification_window_count"
        ]
        == 273830
        and geometry[
            "annotated_event_trial_count"
        ]
        == 2919
        and geometry[
            "piecewise_event_mismatch_count"
        ]
        == 1
        and mismatch_ids
        == [
            KNOWN_DISCORDANT_EVENT
        ]
        and geometry[
            "timing_qualified_event_count"
        ]
        == 2918
        and geometry[
            "timing_excluded_event_count"
        ]
        == 1
        and geometry[
            "ordinary_segment_label_count_mismatch_count"
        ]
        == 0
    )

    source_pass = (
        source[
            "piecewise_activity_call"
        ]
        and source[
            "piecewise_falling_call"
        ]
    )

    status = (
        "PASS"
        if (
            source_pass
            and piecewise_pass
            and clock_pass
            and calibration_pass
        )
        else "FAIL"
    )

    timing_contract = {
        "authoritative_sample_clock":
            "oriented_sensor_FrameCounter",

        "sampling_frequency_hz":
            100,

        "ordinary_trial_window_mapping": {
            "raw_start_position":
                "15 * stored_window_index",

            "raw_last_position":
                (
                    "15 * stored_window_index + 29"
                ),
        },

        "annotated_fall_trial_window_mapping": {
            "activity_window_count":
                (
                    "max(0, 1 + floor("
                    "(onset_position - 30) / 15))"
                ),

            "activity_window_raw_start":
                (
                    "15 * region_local_index"
                ),

            "falling_window_count":
                (
                    "max(0, 1 + floor("
                    "((impact_position - onset_position) "
                    "- 30) / 15))"
                ),

            "falling_window_raw_start":
                (
                    "onset_position + "
                    "15 * region_local_index"
                ),

            "raw_last_position":
                (
                    "raw_start_position + 29"
                ),

            "storage_order":
                (
                    "Activity-region windows followed by "
                    "Falling-region windows"
                ),
        },

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

        "runtime_latency_status":
            "DEFERRED_TO_ACTUAL_MCU_MEASUREMENT",

        "timing_qualified_annotated_events":
            2918,

        "timing_excluded_annotated_events": [
            KNOWN_DISCORDANT_EVENT
        ],

        "timing_exclusion_reason":
            (
                "Current curated annotation cannot "
                "reconstruct the frozen processed label "
                "sequence for this one historical trial."
            ),

        "classification_trial_retained":
            True,

        "historical_labels_rewritten":
            False,

        "legacy_univr_time_ms_authoritative":
            False,
    }

    manifest = {
        "schema":
            (
                "crosslayer_phase3n_final_"
                "protocol_freeze_v2"
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

        "supersedes": {
            "manifest":
                (
                    "manifests/"
                    "phase_3n_final_protocol_freeze_v1.json"
                ),

            "reason":
                (
                    "v1 incorrectly assumed continuous "
                    "whole-trial windowing for fall trials"
                ),
        },

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

            "primary_subjects":
                61,

            "primary_trials":
                6309,

            "primary_windows":
                273830,

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

            "annotated_fall_events":
                2919,

            "annotated_events_with_falling_window":
                2918,

            "annotated_activity_only_exception":
                KNOWN_DISCORDANT_EVENT,

            "event_annotations_rewrite_labels":
                False,
        },

        "historical_preprocessing_source":
            source,

        "piecewise_window_geometry":
            geometry,

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
                    "crosslayer_phase3_final_protocol_v2",

                "status":
                    (
                        "FROZEN"
                        if status
                        == "PASS"
                        else "NOT_FROZEN"
                    ),

                "source_manifest":
                    (
                        "manifests/"
                        "phase_3n_final_protocol_freeze_v2.json"
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
        "PHASE_3N_R_SOURCE_CONTRACT=",
        (
            "PASS"
            if source_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_R_PIECEWISE_GEOMETRY=",
        (
            "PASS"
            if piecewise_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_R_FRAME_CLOCK=",
        (
            "PASS"
            if clock_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_R_INT8_IDENTITIES=",
        (
            "PASS"
            if calibration_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3N_R_FINAL_STATUS=",
        status,
        sep="",
    )

    if status != "PASS":
        raise RuntimeError(
            "Phase 3N-R freeze failed"
        )


if __name__ == "__main__":
    main()
