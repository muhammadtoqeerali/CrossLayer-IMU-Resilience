from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
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

UNIVR_ORIENTED = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFall_oriented/sensors_data"
)

UNIVR_ORIGINAL = Path(
    "/mnt/hdd16T/protechto/"
    "UniVrFallOriginalDataset"
)

OUTPUT = (
    ROOT
    / "manifests/"
      "phase_3m_label_time_repair_v2.json"
)

WINDOW_SAMPLES = 30
STEP_SAMPLES = 15
DEADLINE_SAMPLES = 15


def source_subject(
    value: Any,
) -> int:

    match = re.search(
        r"(\d+)$",
        str(value).strip(),
    )

    if not match:
        raise ValueError(
            f"Cannot parse subject: {value!r}"
        )

    return int(
        match.group(1)
    )


def processed_subject(
    dataset: str,
    source: int,
) -> int:

    if dataset == "UNIVR":
        return source

    if dataset == "KFALL":
        return source + 100

    raise ValueError(
        dataset
    )


def normalise_label(
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

    text = str(
        value
    ).strip().lower()

    if text.startswith(
        "fall"
    ):
        return "Falling"

    return "Activity"


def read_legacy_univr(
    path: Path,
):
    lines = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines()

    header_index = None

    for index, line in enumerate(
        lines[:60]
    ):
        low = line.strip().lower()

        if (
            low.startswith(
                "time[ms],"
            )
            and "accx[mg]"
            in low
            and "gyrox[mdps]"
            in low
        ):
            header_index = index
            break

    if header_index is None:
        raise RuntimeError(
            "legacy time[ms] header not found"
        )

    frame = pd.read_csv(
        path,
        skiprows=header_index,
        sep=",",
        engine="python",
    )

    frame = frame.drop(
        columns=[
            column
            for column
            in frame.columns
            if (
                str(column).lower().startswith(
                    "unnamed"
                )
                or str(column).strip()
                == ""
            )
        ],
        errors="ignore",
    )

    return (
        frame,
        header_index,
    )


def numeric(
    series,
):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )


def summary(
    values,
):
    finite = [
        float(value)
        for value
        in values
        if math.isfinite(
            float(value)
        )
    ]

    if not finite:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }

    return {
        "count":
            len(finite),

        "min":
            min(finite),

        "median":
            median(finite),

        "max":
            max(finite),
    }


def main():
    if not PRIMARY.is_dir():
        raise RuntimeError(
            f"Missing {PRIMARY}"
        )

    if not EVENT_INDEX.is_file():
        raise RuntimeError(
            f"Missing {EVENT_INDEX}"
        )

    events = pd.read_csv(
        EVENT_INDEX
    )

    if len(
        events
    ) != 2919:
        raise RuntimeError(
            "Expected exactly 2919 curated events"
        )

    seen = set()

    dataset_counts = Counter()

    missing_processed = []

    activity_only = []

    replay_exact = 0
    replay_nonexact = []

    total_windows = 0
    replay_window_mismatches = 0

    strict_duration_possible = 0

    any_grid_window_predeadline = 0

    historical_falling_predeadline = 0

    onset_overlap_predeadline = 0

    univr = {
        "event_count":
            0,

        "oriented_missing":
            [],

        "original_missing":
            [],

        "original_parse_failures":
            [],

        "row_count_mismatches":
            [],

        "position_out_of_range":
            [],

        "legacy_timestamp_delta_ms":
            [],

        "legacy_event_duration_error_ms":
            [],

        "oriented_event_duration_error_ms":
            [],

        "legacy_backward_steps":
            0,

        "legacy_duplicate_steps":
            0,

        "framecounter_offsets":
            Counter(),

        "header_lines":
            Counter(),
    }

    for _, row in events.iterrows():

        dataset = str(
            row[
                "dataset_id"
            ]
        ).upper()

        source = source_subject(
            row[
                "source_subject_id"
            ]
        )

        subject = processed_subject(
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

        event_id = str(
            row[
                "event_id"
            ]
        )

        onset = int(
            round(
                float(
                    row[
                        "fall_start_position"
                    ]
                )
            )
        )

        impact = int(
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

        key = (
            dataset,
            subject,
            task,
            trial,
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate event key {key}"
            )

        seen.add(
            key
        )

        dataset_counts[
            dataset
        ] += 1

        trial_dir = (
            PRIMARY
            / str(subject)
            / str(task)
            / str(trial)
        )

        label_path = (
            trial_dir
            / "labels.npy"
        )

        if not label_path.is_file():
            missing_processed.append(
                {
                    "event_id":
                        event_id,

                    "trial_dir":
                        str(
                            trial_dir
                        ),
                }
            )
            continue

        raw_labels = np.load(
            label_path,
            allow_pickle=True,
        )

        labels = np.array(
            [
                normalise_label(
                    value
                )
                for value
                in raw_labels
            ],
            dtype=object,
        )

        falling = (
            labels
            == "Falling"
        )

        if not np.any(
            falling
        ):
            activity_only.append(
                event_id
            )

        window_count = int(
            labels.shape[0]
        )

        starts = (
            np.arange(
                window_count,
                dtype=int,
            )
            * STEP_SAMPLES
        )

        ends = (
            starts
            + WINDOW_SAMPLES
        )

        # Candidate historical label replay:
        # sample-level fall interval begins at onset,
        # and each window inherits y[start].
        replay = (
            (starts >= onset)
            & (starts < impact)
        )

        mismatch = np.flatnonzero(
            replay != falling
        )

        total_windows += (
            window_count
        )

        replay_window_mismatches += int(
            mismatch.size
        )

        if mismatch.size == 0:
            replay_exact += 1

        else:
            replay_nonexact.append(
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

                    "onset":
                        onset,

                    "impact":
                        impact,

                    "actual_falling_indices":
                        np.flatnonzero(
                            falling
                        ).astype(
                            int
                        ).tolist(),

                    "replayed_falling_indices":
                        np.flatnonzero(
                            replay
                        ).astype(
                            int
                        ).tolist(),

                    "mismatch_indices":
                        mismatch[
                            :50
                        ].astype(
                            int
                        ).tolist(),
                }
            )

        deadline_limit = (
            impact
            - DEADLINE_SAMPLES
        )

        complete_predeadline = (
            ends
            <= deadline_limit
        )

        strict_postonset = (
            (starts >= onset)
            & complete_predeadline
        )

        positive_predeadline = (
            falling
            & complete_predeadline
        )

        onset_overlapping = (
            (starts < onset)
            & (ends > onset)
            & complete_predeadline
        )

        if np.any(
            strict_postonset
        ):
            strict_duration_possible += 1

        if np.any(
            complete_predeadline
        ):
            any_grid_window_predeadline += 1

        if np.any(
            positive_predeadline
        ):
            historical_falling_predeadline += 1

        if np.any(
            onset_overlapping
        ):
            onset_overlap_predeadline += 1

        # ----------------------------------------------------
        # UniVR original/oriented timing audit
        # ----------------------------------------------------

        if dataset == "UNIVR":
            univr[
                "event_count"
            ] += 1

            basename = Path(
                str(
                    row[
                        "sensor_file"
                    ]
                )
            ).name

            oriented_path = (
                UNIVR_ORIENTED
                / f"SA{source:02d}"
                / basename
            )

            original_path = (
                UNIVR_ORIGINAL
                / f"SA{source:02d}"
                / basename
            )

            if not oriented_path.is_file():
                univr[
                    "oriented_missing"
                ].append(
                    event_id
                )

                oriented = None

            else:
                oriented = pd.read_csv(
                    oriented_path
                )

                oriented = (
                    oriented.drop(
                        columns=[
                            column
                            for column
                            in oriented.columns
                            if str(
                                column
                            ).lower().startswith(
                                "unnamed"
                            )
                        ],
                        errors="ignore",
                    )
                )

                if (
                    onset >= len(
                        oriented
                    )
                    or impact >= len(
                        oriented
                    )
                ):
                    univr[
                        "position_out_of_range"
                    ].append(
                        {
                            "event_id":
                                event_id,

                            "representation":
                                "oriented",

                            "rows":
                                len(
                                    oriented
                                ),

                            "onset":
                                onset,

                            "impact":
                                impact,
                        }
                    )

                else:
                    if (
                        "FrameCounter"
                        in oriented.columns
                    ):
                        fc = numeric(
                            oriented[
                                "FrameCounter"
                            ]
                        )

                        if (
                            np.isfinite(
                                fc[
                                    onset
                                ]
                            )
                            and np.isfinite(
                                fc[
                                    impact
                                ]
                            )
                        ):
                            offset = (
                                int(
                                    round(
                                        fc[
                                            onset
                                        ]
                                    )
                                )
                                - onset_frame,

                                int(
                                    round(
                                        fc[
                                            impact
                                        ]
                                    )
                                )
                                - impact_frame,
                            )

                            univr[
                                "framecounter_offsets"
                            ][
                                offset
                            ] += 1

                    if (
                        "TimeStamp(s)"
                        in oriented.columns
                    ):
                        ts = numeric(
                            oriented[
                                "TimeStamp(s)"
                            ]
                        )

                        if (
                            np.isfinite(
                                ts[
                                    onset
                                ]
                            )
                            and np.isfinite(
                                ts[
                                    impact
                                ]
                            )
                        ):
                            observed = (
                                ts[
                                    impact
                                ]
                                - ts[
                                    onset
                                ]
                            ) * 1000.0

                            expected = float(
                                row[
                                    "fall_duration_ms"
                                ]
                            )

                            univr[
                                "oriented_event_duration_error_ms"
                            ].append(
                                observed
                                - expected
                            )

            if not original_path.is_file():
                univr[
                    "original_missing"
                ].append(
                    event_id
                )

                continue

            try:
                original, header = (
                    read_legacy_univr(
                        original_path
                    )
                )

            except Exception as exc:
                univr[
                    "original_parse_failures"
                ].append(
                    {
                        "event_id":
                            event_id,

                        "error":
                            (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                    }
                )

                continue

            univr[
                "header_lines"
            ][
                header
            ] += 1

            if (
                oriented is not None
                and len(
                    original
                )
                != len(
                    oriented
                )
            ):
                univr[
                    "row_count_mismatches"
                ].append(
                    {
                        "event_id":
                            event_id,

                        "original_rows":
                            len(
                                original
                            ),

                        "oriented_rows":
                            len(
                                oriented
                            ),
                    }
                )

            if (
                onset >= len(
                    original
                )
                or impact >= len(
                    original
                )
            ):
                univr[
                    "position_out_of_range"
                ].append(
                    {
                        "event_id":
                            event_id,

                        "representation":
                            "original",

                        "rows":
                            len(
                                original
                            ),

                        "onset":
                            onset,

                        "impact":
                            impact,
                    }
                )

                continue

            if (
                "time[ms]"
                not in original.columns
            ):
                univr[
                    "original_parse_failures"
                ].append(
                    {
                        "event_id":
                            event_id,

                        "error":
                            (
                                "time[ms] column absent"
                            ),
                    }
                )

                continue

            time_ms = numeric(
                original[
                    "time[ms]"
                ]
            )

            finite = time_ms[
                np.isfinite(
                    time_ms
                )
            ]

            if finite.size >= 2:
                diffs = np.diff(
                    finite
                )

                positive = diffs[
                    diffs > 0
                ]

                if positive.size:
                    univr[
                        "legacy_timestamp_delta_ms"
                    ].append(
                        float(
                            np.median(
                                positive
                            )
                        )
                    )

                univr[
                    "legacy_backward_steps"
                ] += int(
                    np.sum(
                        diffs < 0
                    )
                )

                univr[
                    "legacy_duplicate_steps"
                ] += int(
                    np.sum(
                        diffs == 0
                    )
                )

            if (
                np.isfinite(
                    time_ms[
                        onset
                    ]
                )
                and np.isfinite(
                    time_ms[
                        impact
                    ]
                )
            ):
                observed = (
                    time_ms[
                        impact
                    ]
                    - time_ms[
                        onset
                    ]
                )

                expected = float(
                    row[
                        "fall_duration_ms"
                    ]
                )

                univr[
                    "legacy_event_duration_error_ms"
                ].append(
                    observed
                    - expected
                )

    if len(
        seen
    ) != 2919:
        raise RuntimeError(
            "Canonical event population changed"
        )

    core_binding_pass = (
        len(
            seen
        )
        == 2919
        and len(
            missing_processed
        )
        == 0
        and dict(
            dataset_counts
        )
        == {
            "KFALL":
                2346,

            "UNIVR":
                573,
        }
    )

    expected_activity_only = (
        activity_only
        == [
            "KFALL_106_T27_R05"
        ]
    )

    univr_parser_pass = (
        univr[
            "event_count"
        ]
        == 573
        and len(
            univr[
                "original_missing"
            ]
        )
        == 0
        and len(
            univr[
                "original_parse_failures"
            ]
        )
        == 0
        and len(
            univr[
                "position_out_of_range"
            ]
        )
        == 0
    )

    univr_summary = {
        "event_count":
            univr[
                "event_count"
            ],

        "oriented_missing_count":
            len(
                univr[
                    "oriented_missing"
                ]
            ),

        "original_missing_count":
            len(
                univr[
                    "original_missing"
                ]
            ),

        "original_parse_failure_count":
            len(
                univr[
                    "original_parse_failures"
                ]
            ),

        "row_count_mismatch_count":
            len(
                univr[
                    "row_count_mismatches"
                ]
            ),

        "position_out_of_range_count":
            len(
                univr[
                    "position_out_of_range"
                ]
            ),

        "framecounter_offsets":
            [
                {
                    "onset_offset":
                        key[0],

                    "impact_offset":
                        key[1],

                    "count":
                        value,
                }
                for key, value
                in sorted(
                    univr[
                        "framecounter_offsets"
                    ].items()
                )
            ],

        "legacy_header_lines":
            dict(
                univr[
                    "header_lines"
                ]
            ),

        "legacy_timestamp_delta_ms":
            summary(
                univr[
                    "legacy_timestamp_delta_ms"
                ]
            ),

        "legacy_backward_step_total":
            univr[
                "legacy_backward_steps"
            ],

        "legacy_duplicate_step_total":
            univr[
                "legacy_duplicate_steps"
            ],

        "legacy_event_duration_abs_error_ms":
            summary(
                [
                    abs(
                        value
                    )
                    for value
                    in univr[
                        "legacy_event_duration_error_ms"
                    ]
                ]
            ),

        "oriented_event_duration_abs_error_ms":
            summary(
                [
                    abs(
                        value
                    )
                    for value
                    in univr[
                        "oriented_event_duration_error_ms"
                    ]
                ]
            ),

        "row_count_mismatch_examples":
            univr[
                "row_count_mismatches"
            ][:20],

        "parse_failure_examples":
            univr[
                "original_parse_failures"
            ][:20],
    }

    manifest = {
        "schema":
            "crosslayer_phase3m_label_time_repair_v2",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat(),

        "status":
            (
                "PASS"
                if (
                    core_binding_pass
                    and expected_activity_only
                    and univr_parser_pass
                )
                else "FAIL"
            ),

        "audit_mode":
            "NO_MODEL_EXECUTION",

        "protocol": {
            "window_ms":
                300,

            "window_samples":
                WINDOW_SAMPLES,

            "overlap_percent":
                50,

            "stride_samples":
                STEP_SAMPLES,

            "stride_ms":
                150,

            "safety_deadline_ms_before_impact":
                150,
        },

        "canonical_binding": {
            "annotated_event_count":
                len(
                    seen
                ),

            "dataset_counts":
                dict(
                    dataset_counts
                ),

            "processed_trial_missing_count":
                len(
                    missing_processed
                ),

            "core_binding_pass":
                core_binding_pass,
        },

        "historical_label_semantics": {
            "deadline_used_to_generate_historical_labels":
                False,

            "annotated_events_with_no_falling_window":
                activity_only,

            "annotated_events_with_falling_window":
                (
                    2919
                    - len(
                        activity_only
                    )
                ),

            "candidate_start_label_replay": {
                "exact_trial_count":
                    replay_exact,

                "nonexact_trial_count":
                    len(
                        replay_nonexact
                    ),

                "total_window_count":
                    total_windows,

                "window_mismatch_count":
                    replay_window_mismatches,

                "nonexact_examples":
                    replay_nonexact[
                        :50
                    ],
            },
        },

        "safety_timing_geometry": {
            "events_with_full_postonset_300ms_window_before_deadline":
                strict_duration_possible,

            "events_with_any_complete_grid_window_before_deadline":
                any_grid_window_predeadline,

            "events_with_historical_falling_window_before_deadline":
                historical_falling_predeadline,

            "events_with_onset_overlapping_window_before_deadline":
                onset_overlap_predeadline,

            "interpretation":
                (
                    "Timing geometry only. "
                    "No model predictions were evaluated."
                ),
        },

        "univr_legacy_time_audit":
            univr_summary,

        "scientific_interpretation": {
            "phase3m_strict_equivalence_rejected":
                True,

            "canonical_event_binding_retained":
                True,

            "historical_training_labels_retained_unchanged":
                True,

            "safety_deadline_is_separate_evaluation_layer":
                True,

            "physical_lead_time_fully_frozen":
                False,
        },

        "scientific_boundary": {
            "fivefold_membership_changed":
                False,

            "model_trained":
                False,

            "predictions_executed":
                False,

            "outer_test_outcomes_opened":
                False,

            "onfield_outcomes_opened":
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
        "CANONICAL_EVENT_COUNT=",
        len(
            seen
        ),
        sep="",
    )

    print(
        "DATASET_COUNTS=",
        dict(
            dataset_counts
        ),
        sep="",
    )

    print(
        "MISSING_PROCESSED_TRIALS=",
        len(
            missing_processed
        ),
        sep="",
    )

    print(
        "ACTIVITY_ONLY_ANNOTATED_EVENTS=",
        activity_only,
        sep="",
    )

    print()
    print(
        "START_LABEL_REPLAY_EXACT_TRIALS=",
        replay_exact,
        sep="",
    )

    print(
        "START_LABEL_REPLAY_NONEXACT_TRIALS=",
        len(
            replay_nonexact
        ),
        sep="",
    )

    print(
        "START_LABEL_REPLAY_WINDOW_MISMATCHES=",
        replay_window_mismatches,
        sep="",
    )

    print()
    print(
        "FULL_POSTONSET_PREDEADLINE_EVENTS=",
        strict_duration_possible,
        sep="",
    )

    print(
        "ANY_GRID_WINDOW_PREDEADLINE_EVENTS=",
        any_grid_window_predeadline,
        sep="",
    )

    print(
        "HISTORICAL_FALLING_WINDOW_PREDEADLINE_EVENTS=",
        historical_falling_predeadline,
        sep="",
    )

    print(
        "ONSET_OVERLAPPING_PREDEADLINE_EVENTS=",
        onset_overlap_predeadline,
        sep="",
    )

    print()
    print(
        "UNIVR_LEGACY_AUDIT="
    )

    print(
        json.dumps(
            univr_summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "PHASE_3M_R_CORE_BINDING=",
        (
            "PASS"
            if core_binding_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3M_R_ACTIVITY_EXCEPTION=",
        (
            "PASS"
            if expected_activity_only
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3M_R_UNIVR_LEGACY_PARSE=",
        (
            "PASS"
            if univr_parser_pass
            else "FAIL"
        ),
        sep="",
    )

    print(
        "PHASE_3M_R_STATUS=",
        manifest[
            "status"
        ],
        sep="",
    )

    if manifest[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Phase-3M-R audit did not pass"
        )


if __name__ == "__main__":
    main()
