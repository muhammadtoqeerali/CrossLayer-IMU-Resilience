"""Phase-4H dev/cal runner v2.

Scientific repair relative to v1
--------------------------------
Falling source-window slicing is bound to the frozen Phase-3 historical
route:

    raw_start = fall_start_frame + 15 * falling_local_index

The annotation frame is used directly as the Python source-array index,
exactly as in the recovered historical preprocessing source.

Physical event timing remains separately defined by curated zero-based
positions / oriented FrameCounter correspondence.

Runner v1 is imported and preserved unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sensor_fi_devcal_runner as V1


ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_FAMILIES = (
    "drift",
    "stuck_channel",
    "dropout",
    "frame_loss",
    "jitter",
    "delay",
    "orientation",
)


# Re-export unchanged qualified-v1 integration functions.
assert_partition = V1.assert_partition
load_fp32_model = V1.load_fp32_model
load_ptq_model = V1.load_ptq_model
logits_and_positive_probability = V1.logits_and_positive_probability
load_reference_scales = V1.load_reference_scales
load_fp32_members = V1.load_fp32_members
load_ptq_members = V1.load_ptq_members
load_trial_segments_and_labels = V1.load_trial_segments_and_labels
load_source_trial = V1.load_source_trial
source_trial_path = V1.source_trial_path
stored_trial_paths = V1.stored_trial_paths
representative_instance = V1.representative_instance
SAMPLING = V1.SAMPLING
OPERATORS = V1.OPERATORS
low_pass_filter = V1.low_pass_filter


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric_subject(value) -> int:
    matches = re.findall(
        r"\d+",
        str(value).strip(),
    )

    if not matches:
        raise ValueError(
            f"no numeric subject id in {value!r}"
        )

    return int(
        matches[-1]
    )


def storage_subject(
    *,
    dataset_id,
    source_subject_id,
) -> int:
    dataset = str(
        dataset_id
    ).upper()

    subject = numeric_subject(
        source_subject_id
    )

    if "KFALL" in dataset:
        return 100 + subject

    if "UNIVR" in dataset:
        return subject

    raise ValueError(
        f"unknown dataset_id: {dataset_id!r}"
    )


def load_risk_index(path):
    frame = pd.read_csv(
        path
    )

    required = {
        "dataset_id",
        "source_subject_id",
        "task_id",
        "trial_id",
        "fall_start_frame",
        "fall_start_position",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"risk index missing fields: {sorted(missing)}"
        )

    by_trial = defaultdict(
        list
    )

    for row in frame.itertuples(
        index=False
    ):
        subject = storage_subject(
            dataset_id=row.dataset_id,
            source_subject_id=row.source_subject_id,
        )

        key = (
            int(subject),
            int(row.task_id),
            int(row.trial_id),
        )

        by_trial[
            key
        ].append(
            row
        )

    return (
        frame,
        by_trial,
    )


def risk_row_for_falling_trial(
    risk_by_trial,
    *,
    subject,
    task,
    trial,
):
    key = (
        int(subject),
        int(task),
        int(trial),
    )

    rows = risk_by_trial.get(
        key,
        [],
    )

    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one risk row for Falling trial {key}, "
            f"found {len(rows)}"
        )

    return rows[0]


def rewindow_sequence(
    sequence,
    stored_labels,
    *,
    fall_start_frame=None,
):
    """Historical Phase-3 source-faithful rewindowing.

    Activity:
        raw_start = 15 * activity_local_index

    Falling:
        raw_start = fall_start_frame + 15 * falling_local_index

    Activity-only trials, including the sole historical fallback
    exception, naturally use whole-trial 15-sample-stride windows.
    """

    x = np.asarray(
        sequence,
        dtype=np.float64,
    )

    labels = np.asarray(
        stored_labels,
        dtype=np.int8,
    )

    if (
        x.ndim != 2
        or x.shape[1] != 9
    ):
        raise ValueError(
            "sequence must have shape N x 9"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            "sequence must be finite"
        )

    activity_count = int(
        np.sum(
            labels == 0
        )
    )

    falling_count = int(
        np.sum(
            labels == 1
        )
    )

    expected_order = np.concatenate([
        np.zeros(
            activity_count,
            dtype=np.int8,
        ),
        np.ones(
            falling_count,
            dtype=np.int8,
        ),
    ])

    if not np.array_equal(
        labels,
        expected_order,
    ):
        raise ValueError(
            "stored labels must be Activity windows followed by Falling windows"
        )

    starts = [
        15 * i
        for i in range(
            activity_count
        )
    ]

    if falling_count:
        if fall_start_frame is None:
            raise ValueError(
                "fall_start_frame required for Falling trial"
            )

        start_frame = int(
            fall_start_frame
        )

        if start_frame < 0:
            raise ValueError(
                "fall_start_frame must be non-negative"
            )

        starts.extend(
            start_frame
            + 15 * j
            for j in range(
                falling_count
            )
        )

    windows = []

    for start in starts:
        stop = (
            start
            + 30
        )

        if stop > len(
            x
        ):
            raise ValueError(
                "historical Phase-3 rewindowing exceeds source bounds"
            )

        windows.append(
            low_pass_filter(
                x[
                    start:stop
                ],
                5,
                100,
            )
        )

    if not windows:
        return np.empty(
            (
                0,
                30,
                9,
            ),
            dtype=np.float64,
        )

    return np.asarray(
        windows,
        dtype=np.float64,
    )


def calibration_parent_folds(
    calibration_config,
):
    d = json.loads(
        Path(
            calibration_config
        ).read_text()
    )

    mapping = defaultdict(
        set
    )

    for fold_rec in d[
        "folds"
    ]:
        fold = int(
            fold_rec[
                "fold"
            ]
        )

        for row in fold_rec[
            "selection"
        ]:
            key = (
                int(
                    row[
                        "subject"
                    ]
                ),
                int(
                    row[
                        "task"
                    ]
                ),
                int(
                    row[
                        "trial"
                    ]
                ),
            )

            mapping[
                key
            ].add(
                fold
            )

    return mapping


def falling_parents_by_fold(
    calibration_config,
    dataset_root,
):
    d = json.loads(
        Path(
            calibration_config
        ).read_text()
    )

    result = {}

    for fold_rec in d[
        "folds"
    ]:
        fold = int(
            fold_rec[
                "fold"
            ]
        )

        keys = sorted({
            (
                int(
                    row[
                        "subject"
                    ]
                ),
                int(
                    row[
                        "task"
                    ]
                ),
                int(
                    row[
                        "trial"
                    ]
                ),
            )
            for row
            in fold_rec[
                "selection"
            ]
        })

        falling = []

        for subject, task, trial in keys:
            _, labels = (
                load_trial_segments_and_labels(
                    dataset_root,
                    subject=subject,
                    task=task,
                    trial=trial,
                )
            )

            if np.any(
                labels == 1
            ):
                falling.append(
                    (
                        subject,
                        task,
                        trial,
                    )
                )

        result[
            fold
        ] = falling

    return result


def route_offset_ok(
    *,
    subject,
    fall_start_frame,
    fall_start_position,
):
    expected = (
        1
        if int(
            subject
        ) >= 100
        else 0
    )

    return (
        int(
            fall_start_frame
        )
        - int(
            fall_start_position
        )
        == expected
    )


def run_qualification(
    *,
    config_path,
    output_path,
):
    cfg = json.loads(
        Path(
            config_path
        ).read_text()
    )

    assert_partition(
        "training_calibration",
        execution_stage="qualification",
    )

    phase3_path = (
        ROOT
        / cfg[
            "phase3_route"
        ][
            "manifest_path"
        ]
    )

    if sha256_file(
        phase3_path
    ) != cfg[
        "phase3_route"
    ][
        "manifest_sha256"
    ]:
        raise ValueError(
            "Phase-3 route manifest hash changed"
        )

    risk_path = Path(
        cfg[
            "risk_index"
        ][
            "path"
        ]
    )

    if sha256_file(
        risk_path
    ) != cfg[
        "risk_index"
    ][
        "sha256"
    ]:
        raise ValueError(
            "risk-index hash changed"
        )

    calibration_path = (
        ROOT
        / cfg[
            "qualification"
        ][
            "calibration_identity_path"
        ]
    )

    dataset_root = Path(
        cfg[
            "data"
        ][
            "dataset_root"
        ]
    )

    severity_path = (
        ROOT
        / cfg[
            "data"
        ][
            "severity_protocol"
        ]
    )

    severity_protocol = json.loads(
        severity_path.read_text()
    )

    severity_audit = Path(
        cfg[
            "data"
        ][
            "severity_audit"
        ]
    )

    risk_frame, risk_by_trial = (
        load_risk_index(
            risk_path
        )
    )

    parent_folds = calibration_parent_folds(
        calibration_path
    )

    route_digest = hashlib.sha256()

    dataset_counts = defaultdict(
        int
    )

    exhaustive_count = 0

    exact_count = 0

    position_exact_count = 0

    route_failures = []

    falling_parent_records = []

    source_cache = {}

    for key, folds in sorted(
        parent_folds.items()
    ):
        subject, task, trial = key

        stored, labels = (
            load_trial_segments_and_labels(
                dataset_root,
                subject=subject,
                task=task,
                trial=trial,
            )
        )

        if not np.any(
            labels == 1
        ):
            continue

        row = risk_row_for_falling_trial(
            risk_by_trial,
            subject=subject,
            task=task,
            trial=trial,
        )

        source_path = source_trial_path(
            subject=subject,
            task=task,
            trial=trial,
            univr_root=cfg[
                "data"
            ][
                "univr_source_root"
            ],
            kfall_root=cfg[
                "data"
            ][
                "kfall_source_root"
            ],
        )

        source = load_source_trial(
            source_path
        )

        source_cache[
            key
        ] = (
            source,
            stored,
            labels,
            row,
            source_path,
        )

        rebuilt = rewindow_sequence(
            source,
            labels,
            fall_start_frame=int(
                row.fall_start_frame
            ),
        )

        exact = bool(
            rebuilt.shape
            == stored.shape
            and np.array_equal(
                rebuilt,
                stored,
            )
        )

        # Diagnostic only: confirms why v1's unexercised generic
        # "position" interpretation was unsafe for KFall.
        position_rebuilt = V1.rewindow_sequence(
            source,
            labels,
            fall_start_position=int(
                row.fall_start_position
            ),
        )

        position_exact = bool(
            position_rebuilt.shape
            == stored.shape
            and np.array_equal(
                position_rebuilt,
                stored,
            )
        )

        offset_ok = route_offset_ok(
            subject=subject,
            fall_start_frame=row.fall_start_frame,
            fall_start_position=row.fall_start_position,
        )

        exhaustive_count += 1
        exact_count += int(
            exact
        )
        position_exact_count += int(
            position_exact
        )

        dataset_name = (
            "KFALL"
            if subject >= 100
            else "UNIVR"
        )

        dataset_counts[
            dataset_name
        ] += 1

        payload = (
            f"{subject}|{task}|{trial}|"
            f"{','.join(str(x) for x in sorted(folds))}|"
            f"{int(row.fall_start_frame)}|"
            f"{int(row.fall_start_position)}"
        )

        route_digest.update(
            payload.encode(
                "utf-8"
            )
        )

        if not (
            exact
            and offset_ok
        ):
            route_failures.append({
                "subject":
                    subject,

                "task":
                    task,

                "trial":
                    trial,

                "folds":
                    sorted(
                        folds
                    ),

                "frame_exact":
                    exact,

                "offset_ok":
                    offset_ok,

                "fall_start_frame":
                    int(
                        row.fall_start_frame
                    ),

                "fall_start_position":
                    int(
                        row.fall_start_position
                    ),
            })

        falling_parent_records.append(
            (
                key,
                sorted(
                    folds
                ),
            )
        )

    expected_count = int(
        cfg[
            "qualification"
        ][
            "expected_unique_falling_parent_count"
        ]
    )

    if exhaustive_count != expected_count:
        raise ValueError(
            f"Falling-parent count changed: {exhaustive_count} != {expected_count}"
        )

    if exact_count != exhaustive_count:
        raise ValueError(
            "Not every calibration Falling parent reconstructed exactly"
        )

    if route_failures:
        raise ValueError(
            f"Phase-3 route failures: {route_failures[:5]}"
        )

    expected_dataset_counts = cfg[
        "qualification"
    ][
        "expected_dataset_parent_counts"
    ]

    if dict(
        dataset_counts
    ) != expected_dataset_counts:
        raise ValueError(
            "Dataset-specific Falling-parent counts changed"
        )

    # Verify the sole frozen historical route exception remains
    # Activity-only and therefore does not enter Falling routing.
    exception = (
        json.loads(
            phase3_path.read_text()
        )[
            "historical_route_exception"
        ]
    )

    if exception[
        "event_id"
    ] != "KFALL_106_T27_R05":
        raise ValueError(
            "Unexpected Phase-3 route exception"
        )

    exception_key = (
        106,
        27,
        5,
    )

    if exception_key in {
        key
        for key, _
        in falling_parent_records
    }:
        raise ValueError(
            "Historical Activity-only exception appeared as Falling parent"
        )

    # --------------------------------------------------------
    # Load seed-42 paired models for all five folds.
    # --------------------------------------------------------

    fp32_freeze = (
        ROOT
        / cfg[
            "models"
        ][
            "fp32_freeze"
        ]
    )

    ptq_all15 = Path(
        cfg[
            "models"
        ][
            "ptq_all15_manifest"
        ]
    )

    fp32_members = load_fp32_members(
        fp32_freeze
    )

    ptq_members = load_ptq_members(
        ptq_all15
    )

    loaded_fp32 = {}
    loaded_ptq = {}

    for fold in range(
        1,
        6
    ):
        fp32_rec = fp32_members[
            (
                42,
                fold,
            )
        ]

        ptq_rec = ptq_members[
            (
                42,
                fold,
            )
        ]

        fp32_path = Path(
            fp32_rec[
                "checkpoint"
            ]
        )

        ptq_path = Path(
            ptq_rec[
                "artifacts"
            ][
                "torchscript"
            ][
                "path"
            ]
        )

        if sha256_file(
            fp32_path
        ) != fp32_rec[
            "sha256"
        ]:
            raise ValueError(
                f"FP32 hash mismatch fold {fold}"
            )

        if sha256_file(
            ptq_path
        ) != ptq_rec[
            "artifacts"
        ][
            "torchscript"
        ][
            "sha256"
        ]:
            raise ValueError(
                f"PTQ hash mismatch fold {fold}"
            )

        loaded_fp32[
            fold
        ] = load_fp32_model(
            fp32_path
        )

        loaded_ptq[
            fold
        ] = load_ptq_model(
            ptq_path
        )

    # --------------------------------------------------------
    # Falling-path fault/model qualification:
    # every fold x all seven sequence families.
    #
    # Parent selection is geometry-only. No model output participates.
    # --------------------------------------------------------

    fold_parents = falling_parents_by_fold(
        calibration_path,
        dataset_root,
    )

    smoke_checks = []

    gates = {}

    for fold in range(
        1,
        6
    ):
        references = load_reference_scales(
            severity_audit,
            fold,
        )

        for family in SEQUENCE_FAMILIES:
            selected = None
            attempts = 0

            for key in fold_parents[
                fold
            ]:
                attempts += 1

                if key in source_cache:
                    (
                        source,
                        stored,
                        labels,
                        risk_row,
                        source_path,
                    ) = source_cache[
                        key
                    ]

                else:
                    subject, task, trial = key

                    stored, labels = (
                        load_trial_segments_and_labels(
                            dataset_root,
                            subject=subject,
                            task=task,
                            trial=trial,
                        )
                    )

                    risk_row = (
                        risk_row_for_falling_trial(
                            risk_by_trial,
                            subject=subject,
                            task=task,
                            trial=trial,
                        )
                    )

                    source_path = (
                        source_trial_path(
                            subject=subject,
                            task=task,
                            trial=trial,
                            univr_root=cfg[
                                "data"
                            ][
                                "univr_source_root"
                            ],
                            kfall_root=cfg[
                                "data"
                            ][
                                "kfall_source_root"
                            ],
                        )
                    )

                    source = load_source_trial(
                        source_path
                    )

                    source_cache[
                        key
                    ] = (
                        source,
                        stored,
                        labels,
                        risk_row,
                        source_path,
                    )

                subject, task, trial = key

                parent_id = (
                    f"subject={subject}"
                    f"|task={task}"
                    f"|trial={trial}"
                )

                instances = (
                    SAMPLING.generate_sequence_instances(
                        fold=fold,
                        partition="training_calibration",
                        parent_sequence_id=parent_id,
                        parent_length=len(
                            source
                        ),
                        severity_protocol=severity_protocol,
                    )
                )

                instance = representative_instance(
                    instances,
                    family=family,
                    level=cfg[
                        "qualification"
                    ][
                        "fault_level"
                    ],
                )

                kwargs = {}

                if family == "drift":
                    kwargs[
                        "reference_scales"
                    ] = references

                source_before = source.copy()

                corrupted, audit = (
                    OPERATORS.apply_fault(
                        source,
                        instance,
                        **kwargs,
                    )
                )

                if not np.array_equal(
                    source,
                    source_before,
                ):
                    raise ValueError(
                        "Operator mutated source input"
                    )

                clean_windows = rewindow_sequence(
                    source,
                    labels,
                    fall_start_frame=int(
                        risk_row.fall_start_frame
                    ),
                )

                faulted_windows = rewindow_sequence(
                    corrupted,
                    labels,
                    fall_start_frame=int(
                        risk_row.fall_start_frame
                    ),
                )

                if not np.array_equal(
                    clean_windows,
                    stored,
                ):
                    raise ValueError(
                        "Candidate parent failed clean v2 route reconstruction"
                    )

                effective_delta = np.max(
                    np.abs(
                        faulted_windows[
                            :,
                            :,
                            0:6
                        ]
                        - clean_windows[
                            :,
                            :,
                            0:6
                        ]
                    ),
                    axis=(
                        1,
                        2,
                    ),
                )

                changed = np.flatnonzero(
                    effective_delta
                    > 0.0
                )

                if len(
                    changed
                ) == 0:
                    continue

                selected = (
                    key,
                    source_path,
                    risk_row,
                    instance,
                    audit,
                    clean_windows,
                    faulted_windows,
                    int(
                        changed[
                            0
                        ]
                    ),
                    attempts,
                )

                break

            if selected is None:
                raise ValueError(
                    f"No propagating calibration Falling parent found "
                    f"for fold={fold} family={family}"
                )

            (
                key,
                source_path,
                risk_row,
                instance,
                audit,
                clean_windows,
                faulted_windows,
                changed_index,
                attempts,
            ) = selected

            corrupted_window = (
                faulted_windows[
                    changed_index
                ].copy()
            )

            pair_a = (
                corrupted_window.copy()
            )

            pair_b = (
                corrupted_window.copy()
            )

            if not np.array_equal(
                pair_a,
                pair_b,
            ):
                raise ValueError(
                    "Paired model inputs differ"
                )

            fp32_logits, _ = (
                logits_and_positive_probability(
                    loaded_fp32[
                        fold
                    ],
                    pair_a,
                )
            )

            ptq_logits, _ = (
                logits_and_positive_probability(
                    loaded_ptq[
                        fold
                    ],
                    pair_b,
                )
            )

            passed = bool(
                fp32_logits.shape
                == (
                    1,
                    2,
                )
                and ptq_logits.shape
                == (
                    1,
                    2,
                )
                and np.isfinite(
                    fp32_logits
                ).all()
                and np.isfinite(
                    ptq_logits
                ).all()
                and audit[
                    "Euler_preserved"
                ]
            )

            gates[
                f"falling_route_{family}_fold{fold}"
            ] = passed

            smoke_checks.append({
                "fold":
                    fold,

                "family":
                    family,

                "subject":
                    int(
                        key[
                            0
                        ]
                    ),

                "task":
                    int(
                        key[
                            1
                        ]
                    ),

                "trial":
                    int(
                        key[
                            2
                        ]
                    ),

                "source_path":
                    str(
                        source_path
                    ),

                "fall_start_frame":
                    int(
                        risk_row.fall_start_frame
                    ),

                "fall_start_position":
                    int(
                        risk_row.fall_start_position
                    ),

                "fault_id":
                    instance[
                        "fault_id"
                    ],

                "replay_id":
                    instance[
                        "replay_id"
                    ],

                "selected_changed_window_index":
                    changed_index,

                "geometry_search_attempts":
                    attempts,

                "same_tensor_to_fp32_ptq":
                    True,

                "performance_metric_computed":
                    False,

                "probability_value_persisted":
                    False,
            })

    # Governance rejection checks.
    for partition in [
        "validation",
        "outer_test",
        "onfield",
    ]:
        rejected = False

        try:
            assert_partition(
                partition,
                execution_stage="qualification",
            )

        except ValueError:
            rejected = True

        gates[
            f"qualification_rejects_{partition}"
        ] = rejected

    gates[
        "all_2269_falling_parents_exact"
    ] = (
        exact_count
        == exhaustive_count
        == expected_count
    )

    gates[
        "dataset_counts_match_probe"
    ] = (
        dict(
            dataset_counts
        )
        == expected_dataset_counts
    )

    gates[
        "phase3_exception_not_falling_parent"
    ] = True

    gates[
        "all_7_sequence_families_each_fold"
    ] = (
        len(
            smoke_checks
        )
        == 35
        and {
            row[
                "family"
            ]
            for row
            in smoke_checks
        }
        == set(
            SEQUENCE_FAMILIES
        )
    )

    gates[
        "performance_acceptance_disabled"
    ] = (
        cfg[
            "qualification"
        ][
            "performance_acceptance_gate"
        ]
        is False
    )

    gates[
        "probability_delta_gate_disabled"
    ] = (
        cfg[
            "qualification"
        ][
            "probability_delta_gate"
        ]
        is False
    )

    gates = {
        key:
            bool(
                value
            )
        for key, value
        in gates.items()
    }

    status = (
        "PASS"
        if all(
            gates.values()
        )
        else "FAIL"
    )

    result = {
        "schema_version":
            "phase4h_sensor_fi_devcal_runner_v2_phase3_falling_route_qualification_result",

        "status":
            status,

        "phase":
            "4H",

        "partition":
            "training_calibration",

        "config_sha256":
            sha256_file(
                config_path
            ),

        "runner_sha256":
            sha256_file(
                Path(
                    __file__
                )
            ),

        "route_conformance": {
            "unique_falling_parent_count":
                exhaustive_count,

            "exact_phase3_frame_route_count":
                exact_count,

            "position_route_exact_count_diagnostic_only":
                position_exact_count,

            "dataset_parent_counts":
                dict(
                    dataset_counts
                ),

            "route_identity_digest_sha256":
                route_digest.hexdigest(),

            "route_failure_count":
                len(
                    route_failures
                ),

            "authoritative_rule":
                "fall_start_frame + 15 * falling_local_index",

            "historical_exception":
                "KFALL_106_T27_R05 -> whole-trial fallback / retained Activity labels",
        },

        "fault_model_smoke": {
            "model_seed":
                42,

            "severity_level":
                cfg[
                    "qualification"
                ][
                    "fault_level"
                ],

            "check_count":
                len(
                    smoke_checks
                ),

            "checks":
                smoke_checks,
        },

        "gates":
            gates,

        "scientific_boundary": {
            "training_calibration_used":
                True,

            "model_predictions_computed":
                True,

            "sensor_fault_injection_executed":
                True,

            "validation_used":
                False,

            "outer_test_used":
                False,

            "onfield_used":
                False,

            "thresholds_applied":
                False,

            "threshold_selection":
                False,

            "performance_metrics_computed":
                False,

            "probabilities_persisted":
                False,

            "robustness_comparison_made":
                False,

            "protocol_tuned_from_model_outputs":
                False,

            "physical_realism_claim":
                False,
        },
    }

    Path(
        output_path
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qualify",
        action="store_true",
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if not args.qualify:
        raise SystemExit(
            "Runner v2 exposes qualification only. "
            "Outer-test execution requires a later frozen execution manifest."
        )

    result = run_qualification(
        config_path=args.config,
        output_path=args.output,
    )

    print(
        "RUNNER_V2_QUALIFICATION_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    print(
        "FALLING_PARENT_COUNT=",
        result[
            "route_conformance"
        ][
            "unique_falling_parent_count"
        ],
        sep="",
    )

    print(
        "EXACT_PHASE3_ROUTE_COUNT=",
        result[
            "route_conformance"
        ][
            "exact_phase3_frame_route_count"
        ],
        sep="",
    )

    print(
        "POSITION_ROUTE_EXACT_DIAGNOSTIC_COUNT=",
        result[
            "route_conformance"
        ][
            "position_route_exact_count_diagnostic_only"
        ],
        sep="",
    )

    print(
        "FAULT_MODEL_SMOKE_COUNT=",
        result[
            "fault_model_smoke"
        ][
            "check_count"
        ],
        sep="",
    )

    print(
        "GATE_COUNT=",
        len(
            result[
                "gates"
            ]
        ),
        sep="",
    )

    print(
        "PASS_COUNT=",
        sum(
            result[
                "gates"
            ].values()
        ),
        sep="",
    )

    if result[
        "status"
    ] != "PASS":
        raise SystemExit(
            "Runner-v2 qualification failed"
        )


if __name__ == "__main__":
    main()
