"""Frozen Phase-4H outer sensor-FI shard executor v1.

The module exposes two modes:

  dry-run
      Metadata/interface verification only. It must never load model
      weights, call a model forward, or call the fault operator.

  execute-shard
      Execute exactly one immutable v2 subject/block shard.

The executor consumes raw 30x9 historical windows. The CNN/PTQ callable
owns normalization internally; no external normalization is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_json(path):
    return json.loads(
        Path(path).read_text()
    )


def repo_path(value):
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


def load_module(
    name,
    path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(path),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_runtime_modules(cfg):
    deps = cfg[
        "frozen_dependencies"
    ]

    runner = load_module(
        "_phase4h_outer_runner_v2",
        repo_path(
            deps[
                "runner_v2"
            ][
                "path"
            ]
        ),
    )

    operators = load_module(
        "_phase4h_outer_operators",
        repo_path(
            deps[
                "operator_engine"
            ][
                "path"
            ]
        ),
    )

    sampling = load_module(
        "_phase4h_outer_sampling",
        repo_path(
            deps[
                "sampling_v3"
            ][
                "path"
            ]
        ),
    )

    return (
        runner,
        operators,
        sampling,
    )


def validate_dependency_hashes(cfg):
    failures = []

    for name, rec in cfg[
        "frozen_dependencies"
    ].items():
        path = repo_path(
            rec[
                "path"
            ]
        )

        if not path.is_file():
            failures.append(
                f"{name}:missing:{path}"
            )
            continue

        observed = sha256_file(
            path
        )

        if observed != rec[
            "sha256"
        ]:
            failures.append(
                f"{name}:hash:{observed}"
            )

    if failures:
        raise ValueError(
            "frozen dependency validation failed: "
            + "; ".join(
                failures
            )
        )


def validate_freeze(
    cfg_path,
    executor_path,
    freeze_path,
):
    freeze = load_json(
        freeze_path
    )

    if freeze[
        "status"
    ] != (
        "FROZEN_PRE_OUTER_EXECUTION_EXECUTOR"
    ):
        raise ValueError(
            "executor freeze status invalid"
        )

    if (
        freeze[
            "config"
        ][
            "sha256"
        ]
        != sha256_file(
            cfg_path
        )
    ):
        raise ValueError(
            "executor config hash differs from freeze"
        )

    if (
        freeze[
            "implementation"
        ][
            "sha256"
        ]
        != sha256_file(
            executor_path
        )
    ):
        raise ValueError(
            "executor implementation hash differs from freeze"
        )

    return freeze


def load_thresholds(path):
    with Path(
        path
    ).open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    expected_ops = {
        "balanced",
        "low_false_alarm",
        "timely_150ms",
    }

    if len(
        rows
    ) != 45:
        raise ValueError(
            "expected 45 frozen threshold rows"
        )

    result = {}

    for row in rows:
        key = (
            int(
                row[
                    "seed"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
            str(
                row[
                    "operating_point"
                ]
            ),
        )

        if (
            key in result
            or key[
                2
            ] not in expected_ops
        ):
            raise ValueError(
                f"invalid threshold key: {key}"
            )

        result[
            key
        ] = {
            "threshold":
                float(
                    row[
                        "threshold"
                    ]
                ),

            "required_consecutive":
                int(
                    row[
                        "required_consecutive"
                    ]
                ),
        }

    expected_keys = {
        (
            seed,
            fold,
            op,
        )
        for seed
        in (
            42,
            123,
            2025,
        )
        for fold
        in range(
            1,
            6,
        )
        for op
        in expected_ops
    }

    if set(
        result
    ) != expected_keys:
        raise ValueError(
            "threshold matrix incomplete"
        )

    return result


def slot_key(instance):
    return (
        str(
            instance[
                "family"
            ]
        ),
        str(
            instance[
                "severity"
            ][
                "level"
            ]
        ),
        str(
            instance[
                "variant_id"
            ]
        ),
        int(
            instance[
                "replicate_index"
            ]
        ),
    )


def severity_order(value):
    return {
        "L1":
            1,
        "L2":
            2,
        "L3":
            3,
    }[
        str(
            value
        )
    ]


def sorted_slots(
    instances,
    *,
    family,
):
    rows = [
        row
        for row
        in instances
        if row[
            "family"
        ]
        == family
    ]

    keys = {
        slot_key(
            row
        )
        for row
        in rows
    }

    if len(
        keys
    ) != len(
        rows
    ):
        raise ValueError(
            "duplicate condition slot in representative metadata"
        )

    return sorted(
        keys,
        key=lambda x: (
            severity_order(
                x[
                    1
                ]
            ),
            x[
                2
            ],
            x[
                3
            ],
        ),
    )


def trigger_episodes(
    probabilities,
    threshold,
    required_consecutive,
):
    episodes = []

    run = 0
    armed = True

    for index, probability in enumerate(
        np.asarray(
            probabilities,
            dtype=float,
        )
    ):
        if probability >= threshold:
            run += 1

            if (
                armed
                and run
                >= required_consecutive
            ):
                episodes.append(
                    index
                    - required_consecutive
                    + 1
                )

                armed = False

        else:
            run = 0
            armed = True

    return episodes


def first_valid_trigger(
    probabilities,
    threshold,
    required_consecutive,
    labels,
):
    for start in trigger_episodes(
        probabilities,
        threshold,
        required_consecutive,
    ):
        if labels is None:
            return start

        stop = (
            start
            + required_consecutive
        )

        if (
            stop
            <= len(
                labels
            )
            and np.all(
                np.asarray(
                    labels
                )[
                    start:stop
                ]
                == 1
            )
        ):
            return start

    return -1


def safe_ratio(
    numerator,
    denominator,
    field,
    undefined,
):
    if denominator <= 0:
        undefined.append(
            field
        )
        return None

    return (
        float(
            numerator
        )
        / float(
            denominator
        )
    )


def historical_window_ends(
    labels,
    *,
    risk_row,
):
    labels = np.asarray(
        labels,
        dtype=np.int8,
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

    ends = np.empty(
        len(
            labels
        ),
        dtype=np.int64,
    )

    if activity_count:
        ends[
            :activity_count
        ] = (
            30
            + np.arange(
                activity_count,
                dtype=np.int64,
            )
            * 15
        )

    fall_start = -1

    if risk_row is not None:
        fall_start = int(
            risk_row.fall_start_position
        )

    if falling_count:
        base = (
            fall_start
            if fall_start >= 0
            else (
                int(
                    ends[
                        activity_count
                        - 1
                    ]
                )
                if activity_count
                else 0
            )
        )

        ends[
            activity_count:
        ] = (
            base
            + 30
            + np.arange(
                falling_count,
                dtype=np.int64,
            )
            * 15
        )

    return ends


def compute_condition_metrics(
    trial_records,
    *,
    threshold,
    required_consecutive,
    quantile_method,
):
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    eligible_events = 0
    detected_events = 0

    false_episodes = 0
    activity_seconds = 0.0

    leads = []

    for row in trial_records:
        probs = np.asarray(
            row[
                "probabilities"
            ],
            dtype=float,
        )

        labels = np.asarray(
            row[
                "labels"
            ],
            dtype=np.int8,
        )

        true_fall = bool(
            row[
                "true_fall"
            ]
        )

        episodes = trigger_episodes(
            probs,
            threshold,
            required_consecutive,
        )

        trigger = first_valid_trigger(
            probs,
            threshold,
            required_consecutive,
            labels
            if true_fall
            else None,
        )

        predicted = (
            trigger
            >= 0
        )

        if true_fall:
            eligible_events += 1

            if predicted:
                tp += 1
                detected_events += 1
            else:
                fn += 1

        else:
            if predicted:
                fp += 1
            else:
                tn += 1

        sampling_rate = float(
            row[
                "sampling_rate_hz"
            ]
        )

        if not true_fall:
            ends = row[
                "window_ends"
            ]

            duration_samples = (
                int(
                    ends[
                        -1
                    ]
                )
                if len(
                    ends
                )
                else 0
            )

            activity_seconds += (
                duration_samples
                / max(
                    sampling_rate,
                    1e-9,
                )
            )

            false_episodes += len(
                episodes
            )

        else:
            fall_start = int(
                row[
                    "fall_start_position"
                ]
            )

            if fall_start > 0:
                activity_seconds += (
                    fall_start
                    / max(
                        sampling_rate,
                        1e-9,
                    )
                )

            for episode in episodes:
                stop = (
                    episode
                    + required_consecutive
                )

                if (
                    stop
                    > len(
                        labels
                    )
                    or not np.all(
                        labels[
                            episode:stop
                        ]
                        == 1
                    )
                ):
                    false_episodes += 1

            if predicted:
                confirmation_index = (
                    trigger
                    + required_consecutive
                    - 1
                )

                end = int(
                    row[
                        "window_ends"
                    ][
                        confirmation_index
                    ]
                )

                impact = int(
                    row[
                        "impact_position"
                    ]
                )

                if impact >= 0:
                    lead = (
                        (
                            impact
                            - end
                        )
                        * 1000.0
                        / max(
                            sampling_rate,
                            1e-9,
                        )
                    )

                    leads.append(
                        float(
                            lead
                        )
                    )

    undefined = []

    recall = safe_ratio(
        tp,
        tp + fn,
        "falling_recall",
        undefined,
    )

    specificity = safe_ratio(
        tn,
        tn + fp,
        "activity_specificity",
        undefined,
    )

    precision = safe_ratio(
        tp,
        tp + fp,
        "precision",
        undefined,
    )

    f1 = safe_ratio(
        2 * tp,
        2 * tp + fp + fn,
        "f1",
        undefined,
    )

    event_recall = safe_ratio(
        detected_events,
        eligible_events,
        "event_recall",
        undefined,
    )

    if (
        recall is None
        or specificity is None
    ):
        balanced = None

        undefined.append(
            "balanced_accuracy"
        )
    else:
        balanced = (
            recall
            + specificity
        ) / 2.0

    if leads:
        q25 = float(
            np.quantile(
                leads,
                0.25,
                method=quantile_method,
            )
        )

        median = float(
            np.quantile(
                leads,
                0.50,
                method=quantile_method,
            )
        )

        q75 = float(
            np.quantile(
                leads,
                0.75,
                method=quantile_method,
            )
        )
    else:
        q25 = None
        median = None
        q75 = None

        undefined.extend([
            "q25_sensor_lead_ms",
            "median_sensor_lead_ms",
            "q75_sensor_lead_ms",
        ])

    return {
        "trial_count":
            int(
                tp
                + fn
                + tn
                + fp
            ),

        "TP":
            int(
                tp
            ),

        "FN":
            int(
                fn
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "falling_recall":
            recall,

        "activity_specificity":
            specificity,

        "balanced_accuracy":
            balanced,

        "precision":
            precision,

        "f1":
            f1,

        "eligible_event_count":
            int(
                eligible_events
            ),

        "detected_event_count":
            int(
                detected_events
            ),

        "missed_event_count":
            int(
                eligible_events
                - detected_events
            ),

        "event_recall":
            event_recall,

        "median_sensor_lead_ms":
            median,

        "q25_sensor_lead_ms":
            q25,

        "q75_sensor_lead_ms":
            q75,

        "false_trigger_episode_count":
            int(
                false_episodes
            ),

        "activity_seconds":
            float(
                activity_seconds
            ),

        "undefined_metric_fields":
            sorted(
                set(
                    undefined
                )
            ),
    }


def subject_trial_keys(
    dataset_root,
    subject,
):
    subject_dir = (
        Path(
            dataset_root
        )
        / str(
            int(
                subject
            )
        )
    )

    keys = []

    for task_dir in sorted(
        p
        for p
        in subject_dir.iterdir()
        if p.is_dir()
    ):
        for trial_dir in sorted(
            p
            for p
            in task_dir.iterdir()
            if p.is_dir()
        ):
            if (
                (
                    trial_dir
                    / "segments.npy"
                ).is_file()
                and (
                    trial_dir
                    / "labels.npy"
                ).is_file()
            ):
                keys.append(
                    (
                        int(
                            task_dir.name
                        ),
                        int(
                            trial_dir.name
                        ),
                    )
                )

    return keys


def risk_row(
    risk_by_trial,
    *,
    subject,
    task,
    trial,
):
    rows = risk_by_trial.get(
        (
            int(
                subject
            ),
            int(
                task
            ),
            int(
                trial
            ),
        ),
        [],
    )

    if len(
        rows
    ) > 1:
        raise ValueError(
            "multiple risk rows for one trial"
        )

    return (
        rows[
            0
        ]
        if rows
        else None
    )


def make_trial_metadata(
    runner,
    *,
    dataset_root,
    risk_by_trial,
    subject,
    task,
    trial,
):
    segments, labels = (
        runner.load_trial_segments_and_labels(
            dataset_root,
            subject=subject,
            task=task,
            trial=trial,
        )
    )

    rr = risk_row(
        risk_by_trial,
        subject=subject,
        task=task,
        trial=trial,
    )

    stored_falling_count = int(
        np.sum(
            labels == 1
        )
    )

    true_fall = bool(
        stored_falling_count
        > 0
        or rr is not None
    )

    if (
        stored_falling_count
        > 0
        and rr is None
    ):
        raise ValueError(
            "Falling-labelled trial lacks risk row"
        )

    return {
        "task":
            int(
                task
            ),

        "trial":
            int(
                trial
            ),

        "segments":
            segments,

        "labels":
            labels,

        "risk_row":
            rr,

        "true_fall":
            true_fall,

        "fall_start_position":
            (
                int(
                    rr.fall_start_position
                )
                if rr is not None
                else -1
            ),

        "fall_start_frame":
            (
                int(
                    rr.fall_start_frame
                )
                if rr is not None
                else None
            ),

        "impact_position":
            (
                int(
                    rr.impact_position
                )
                if rr is not None
                else -1
            ),

        "window_ends":
            historical_window_ends(
                labels,
                risk_row=rr,
            ),
    }


def infer_probability(
    runner,
    model,
    windows,
    *,
    batch_size,
):
    x = np.asarray(
        windows,
        dtype=np.float64,
    )

    parts = []

    for start in range(
        0,
        len(
            x
        ),
        int(
            batch_size
        ),
    ):
        stop = min(
            len(
                x
            ),
            start
            + int(
                batch_size
            ),
        )

        _, probability = (
            runner.logits_and_positive_probability(
                model,
                x[
                    start:stop
                ],
            )
        )

        parts.append(
            np.asarray(
                probability,
                dtype=np.float64,
            )
        )

    return (
        np.concatenate(
            parts
        )
        if parts
        else np.empty(
            0,
            dtype=np.float64,
        )
    )


def split_probabilities(
    probabilities,
    trial_meta,
):
    result = []

    offset = 0

    for trial in trial_meta:
        count = len(
            trial[
                "labels"
            ]
        )

        part = probabilities[
            offset:
            offset
            + count
        ]

        if len(
            part
        ) != count:
            raise ValueError(
                "probability split length mismatch"
            )

        result.append({
            "probabilities":
                part,

            "labels":
                trial[
                    "labels"
                ],

            "true_fall":
                trial[
                    "true_fall"
                ],

            "window_ends":
                trial[
                    "window_ends"
                ],

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                trial[
                    "fall_start_position"
                ],

            "impact_position":
                trial[
                    "impact_position"
                ],
        })

        offset += count

    if offset != len(
        probabilities
    ):
        raise ValueError(
            "unused model probabilities after trial split"
        )

    return result


def build_parent_instance_maps(
    sampling,
    severity_protocol,
    shard,
    trial_meta,
):
    family = shard[
        "block"
    ]

    fold = int(
        shard[
            "fold"
        ]
    )

    subject = int(
        shard[
            "subject"
        ]
    )

    maps = []

    fault_count = 0

    all_fault_ids = set()
    all_replay_ids = set()

    if shard[
        "parent_kind"
    ] == "stored_window":
        for trial in trial_meta:
            task = trial[
                "task"
            ]

            trial_id = trial[
                "trial"
            ]

            for window_index in range(
                len(
                    trial[
                        "labels"
                    ]
                )
            ):
                parent_id = (
                    f"subject={subject}"
                    f"|task={task}"
                    f"|trial={trial_id}"
                    f"|window={window_index}"
                )

                rows = (
                    sampling.generate_window_instances(
                        fold=fold,
                        partition="outer_test",
                        parent_sequence_id=parent_id,
                        severity_protocol=severity_protocol,
                    )
                )

                chosen = {
                    slot_key(
                        row
                    ):
                        row
                    for row
                    in rows
                    if row[
                        "family"
                    ]
                    == family
                }

                maps.append(
                    chosen
                )

                fault_count += len(
                    chosen
                )

                for row in chosen.values():
                    fid = row[
                        "fault_id"
                    ]

                    rid = row[
                        "replay_id"
                    ]

                    if (
                        fid in all_fault_ids
                        or rid in all_replay_ids
                    ):
                        raise ValueError(
                            "fault/replay identity collision"
                        )

                    all_fault_ids.add(
                        fid
                    )

                    all_replay_ids.add(
                        rid
                    )

    elif shard[
        "parent_kind"
    ] == "source_trial":
        for trial in trial_meta:
            task = trial[
                "task"
            ]

            trial_id = trial[
                "trial"
            ]

            parent_id = (
                f"subject={subject}"
                f"|task={task}"
                f"|trial={trial_id}"
            )

            source_length = int(
                trial[
                    "source_length"
                ]
            )

            rows = (
                sampling.generate_sequence_instances(
                    fold=fold,
                    partition="outer_test",
                    parent_sequence_id=parent_id,
                    parent_length=source_length,
                    severity_protocol=severity_protocol,
                )
            )

            chosen = {
                slot_key(
                    row
                ):
                    row
                for row
                in rows
                if row[
                    "family"
                ]
                == family
            }

            maps.append(
                chosen
            )

            fault_count += len(
                chosen
            )

            for row in chosen.values():
                fid = row[
                    "fault_id"
                ]

                rid = row[
                    "replay_id"
                ]

                if (
                    fid in all_fault_ids
                    or rid in all_replay_ids
                ):
                    raise ValueError(
                        "fault/replay identity collision"
                    )

                all_fault_ids.add(
                    fid
                )

                all_replay_ids.add(
                    rid
                )

    else:
        raise ValueError(
            "cannot build FI map for parent_kind "
            + str(
                shard[
                    "parent_kind"
                ]
            )
        )

    if fault_count != int(
        shard[
            "expected_unique_fault_instances"
        ]
    ):
        raise ValueError(
            "fault-instance count differs from frozen shard"
        )

    return maps


def condition_identity_digest(
    parent_maps,
    slot,
):
    h = hashlib.sha256()

    count = 0

    for mapping in parent_maps:
        row = mapping[
            slot
        ]

        payload = (
            str(
                row[
                    "fault_id"
                ]
            )
            + "|"
            + str(
                row[
                    "replay_id"
                ]
            )
            + "\n"
        )

        h.update(
            payload.encode(
                "utf-8"
            )
        )

        count += 1

    return (
        h.hexdigest(),
        count,
    )


def load_models_for_shard(
    cfg,
    runner,
    *,
    fold,
):
    fp32_freeze = repo_path(
        cfg[
            "frozen_dependencies"
        ][
            "fp32_freeze"
        ][
            "path"
        ]
    )

    ptq_all15 = repo_path(
        cfg[
            "frozen_dependencies"
        ][
            "ptq_all15"
        ][
            "path"
        ]
    )

    fp32_members = runner.load_fp32_members(
        fp32_freeze
    )

    ptq_members = runner.load_ptq_members(
        ptq_all15
    )

    models = {}

    for seed in (
        42,
        123,
        2025,
    ):
        fp32 = fp32_members[
            (
                seed,
                fold,
            )
        ]

        fp32_path = Path(
            fp32[
                "checkpoint"
            ]
        )

        if sha256_file(
            fp32_path
        ) != fp32[
            "sha256"
        ]:
            raise ValueError(
                f"FP32 artifact hash mismatch seed={seed} fold={fold}"
            )

        ptq = ptq_members[
            (
                seed,
                fold,
            )
        ]

        ptq_path = Path(
            ptq[
                "artifacts"
            ][
                "torchscript"
            ][
                "path"
            ]
        )

        if sha256_file(
            ptq_path
        ) != ptq[
            "artifacts"
        ][
            "torchscript"
        ][
            "sha256"
        ]:
            raise ValueError(
                f"PTQ artifact hash mismatch seed={seed} fold={fold}"
            )

        models[
            (
                "prospective_fp32_300ms",
                seed,
            )
        ] = runner.load_fp32_model(
            fp32_path
        )

        models[
            (
                "qualified_static_ptq_v7",
                seed,
            )
        ] = runner.load_ptq_model(
            ptq_path
        )

    return models


def condition_windows(
    *,
    shard,
    slot,
    trial_meta,
    parent_maps,
    runner,
    operators,
    reference_scales,
):
    family = shard[
        "block"
    ]

    if shard[
        "parent_kind"
    ] == "stored_window":
        windows = []

        parent_index = 0

        for trial in trial_meta:
            stored = trial[
                "segments"
            ]

            for window_index in range(
                len(
                    stored
                )
            ):
                instance = parent_maps[
                    parent_index
                ][
                    slot
                ]

                corrupted, audit = (
                    operators.apply_fault(
                        stored[
                            window_index
                        ],
                        instance,
                        reference_scales=reference_scales,
                    )
                )

                if audit[
                    "family"
                ] != family:
                    raise ValueError(
                        "operator audit family mismatch"
                    )

                windows.append(
                    corrupted
                )

                parent_index += 1

        if parent_index != len(
            parent_maps
        ):
            raise ValueError(
                "window parent map was not consumed exactly"
            )

        return np.asarray(
            windows,
            dtype=np.float64,
        )

    if shard[
        "parent_kind"
    ] == "source_trial":
        windows = []

        for parent_index, trial in enumerate(
            trial_meta
        ):
            instance = parent_maps[
                parent_index
            ][
                slot
            ]

            corrupted_source, audit = (
                operators.apply_fault(
                    trial[
                        "source"
                    ],
                    instance,
                    reference_scales=reference_scales,
                )
            )

            if audit[
                "family"
            ] != family:
                raise ValueError(
                    "operator audit family mismatch"
                )

            fall_start_frame = (
                trial[
                    "fall_start_frame"
                ]
                if np.any(
                    trial[
                        "labels"
                    ]
                    == 1
                )
                else None
            )

            rebuilt = runner.rewindow_sequence(
                corrupted_source,
                trial[
                    "labels"
                ],
                fall_start_frame=fall_start_frame,
            )

            if rebuilt.shape != trial[
                "segments"
            ].shape:
                raise ValueError(
                    "faulted sequence rewindow shape differs from stored trial"
                )

            windows.extend(
                rebuilt
            )

        return np.asarray(
            windows,
            dtype=np.float64,
        )

    raise ValueError(
        "condition_windows called for unsupported parent kind"
    )


def clean_windows(
    trial_meta,
):
    return np.concatenate(
        [
            np.asarray(
                trial[
                    "segments"
                ],
                dtype=np.float64,
            )
            for trial
            in trial_meta
        ],
        axis=0,
    )


def json_safe_row(row):
    result = {}

    for key, value in row.items():
        if isinstance(
            value,
            (
                np.integer,
            ),
        ):
            value = int(
                value
            )

        elif isinstance(
            value,
            (
                np.floating,
            ),
        ):
            value = float(
                value
            )

        if isinstance(
            value,
            float,
        ) and not math.isfinite(
            value
        ):
            value = None

        result[
            key
        ] = value

    return result


def condition_base_fields(
    shard,
    slot,
):
    if shard[
        "block"
    ] == "C0":
        return {
            "regime":
                "C0",

            "fault_family":
                "C0",

            "severity_level":
                "C0",

            "variant_id":
                "clean",

            "replicate_index":
                0,
        }

    family, level, variant, replicate = slot

    return {
        "regime":
            "CS",

        "fault_family":
            family,

        "severity_level":
            level,

        "variant_id":
            variant,

        "replicate_index":
            int(
                replicate
            ),
    }


def execute_shard(
    *,
    cfg_path,
    plan_path,
    freeze_path,
    shard_id,
    recompute_partial,
):
    cfg = load_json(
        cfg_path
    )

    validate_freeze(
        cfg_path,
        __file__,
        freeze_path,
    )

    validate_dependency_hashes(
        cfg
    )

    plan = load_json(
        plan_path
    )

    if sha256_file(
        plan_path
    ) != cfg[
        "frozen_dependencies"
    ][
        "outer_v2_plan"
    ][
        "sha256"
    ]:
        raise ValueError(
            "plan path/hash differs from frozen executor config"
        )

    matches = [
        row
        for row
        in plan[
            "shards"
        ]
        if row[
            "shard_id"
        ]
        == shard_id
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"expected one shard for id {shard_id}, found {len(matches)}"
        )

    shard = matches[
        0
    ]

    output_root = Path(
        cfg[
            "execution_locations"
        ][
            "output_root"
        ]
    )

    final_dir = (
        output_root
        / shard_id
    )

    temp_dir = (
        output_root
        / (
            shard_id
            + cfg[
                "resume_contract"
            ][
                "temporary_suffix"
            ]
        )
    )

    success_path = (
        final_dir
        / cfg[
            "resume_contract"
        ][
            "success_marker"
        ]
    )

    if success_path.is_file():
        success = load_json(
            success_path
        )

        if (
            success[
                "shard_id"
            ]
            == shard_id
            and success[
                "executor_config_sha256"
            ]
            == sha256_file(
                cfg_path
            )
            and success[
                "executor_implementation_sha256"
            ]
            == sha256_file(
                __file__
            )
            and success[
                "shard_plan_sha256"
            ]
            == sha256_file(
                plan_path
            )
        ):
            for rel, expected_hash in success[
                "output_hashes"
            ].items():
                target = (
                    final_dir
                    / rel
                )

                if (
                    not target.is_file()
                    or sha256_file(
                        target
                    )
                    != expected_hash
                ):
                    raise ValueError(
                        "existing success shard output hash mismatch"
                    )

            print(
                f"SHARD_REUSED={shard_id}"
            )

            return

        raise ValueError(
            "existing _SUCCESS.json does not match frozen executor"
        )

    if final_dir.exists():
        if not recompute_partial:
            raise ValueError(
                "partial final shard exists without valid success marker; "
                "rerun with --recompute-partial"
            )

        shutil.rmtree(
            final_dir
        )

    if temp_dir.exists():
        if not recompute_partial:
            raise ValueError(
                "temporary shard exists; rerun with --recompute-partial"
            )

        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    runner, operators, sampling = (
        load_runtime_modules(
            cfg
        )
    )

    runner.assert_partition(
        "outer_test",
        execution_stage="final_outer_evaluation",
    )

    fold = int(
        shard[
            "fold"
        ]
    )

    subject = int(
        shard[
            "subject"
        ]
    )

    dataset_root = Path(
        cfg[
            "execution_locations"
        ][
            "dataset_root"
        ]
    )

    risk_path = repo_path(
        cfg[
            "frozen_dependencies"
        ][
            "risk_index"
        ][
            "path"
        ]
    )

    _, risk_by_trial = (
        runner.load_risk_index(
            risk_path
        )
    )

    trial_meta = []

    for task, trial in subject_trial_keys(
        dataset_root,
        subject,
    ):
        meta = make_trial_metadata(
            runner,
            dataset_root=dataset_root,
            risk_by_trial=risk_by_trial,
            subject=subject,
            task=task,
            trial=trial,
        )

        trial_meta.append(
            meta
        )

    observed_trials = len(
        trial_meta
    )

    observed_windows = sum(
        len(
            row[
                "labels"
            ]
        )
        for row
        in trial_meta
    )

    if observed_trials != int(
        shard[
            "subject_inventory"
        ][
            "trials"
        ]
    ):
        raise ValueError(
            "subject trial count differs from frozen shard"
        )

    if observed_windows != int(
        shard[
            "subject_inventory"
        ][
            "windows"
        ]
    ):
        raise ValueError(
            "subject window count differs from frozen shard"
        )

    if shard[
        "block"
    ] != "C0":
        severity_protocol = load_json(
            repo_path(
                cfg[
                    "frozen_dependencies"
                ][
                    "severity_protocol"
                ][
                    "path"
                ]
            )
        )

        references = (
            runner.load_reference_scales(
                repo_path(
                    cfg[
                        "frozen_dependencies"
                    ][
                        "severity_audit"
                    ][
                        "path"
                    ]
                ),
                fold,
            )
        )

        if shard[
            "parent_kind"
        ] == "source_trial":
            for trial in trial_meta:
                source_path = (
                    runner.source_trial_path(
                        subject=subject,
                        task=trial[
                            "task"
                        ],
                        trial=trial[
                            "trial"
                        ],
                        univr_root=cfg[
                            "execution_locations"
                        ][
                            "univr_source_root"
                        ],
                        kfall_root=cfg[
                            "execution_locations"
                        ][
                            "kfall_source_root"
                        ],
                    )
                )

                source = runner.load_source_trial(
                    source_path
                )

                trial[
                    "source"
                ] = source

                trial[
                    "source_length"
                ] = len(
                    source
                )

        parent_maps = (
            build_parent_instance_maps(
                sampling,
                severity_protocol,
                shard,
                trial_meta,
            )
        )

        first_map = parent_maps[
            0
        ]

        slots = sorted(
            first_map,
            key=lambda x: (
                severity_order(
                    x[
                        1
                    ]
                ),
                x[
                    2
                ],
                x[
                    3
                ],
            ),
        )

        if len(
            slots
        ) != int(
            shard[
                "family_instances_per_parent"
            ]
        ):
            raise ValueError(
                "condition slot count differs from frozen shard"
            )

        for mapping in parent_maps:
            if set(
                mapping
            ) != set(
                slots
            ):
                raise ValueError(
                    "condition slot set differs across parents"
                )

    else:
        severity_protocol = None
        references = None
        parent_maps = None
        slots = [
            (
                "C0",
                "C0",
                "clean",
                0,
            )
        ]

    thresholds = load_thresholds(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "thresholds"
            ][
                "path"
            ]
        )
    )

    models = load_models_for_shard(
        cfg,
        runner,
        fold=fold,
    )

    rows = []

    actual_model_window_evaluations = 0

    for slot in slots:
        if shard[
            "block"
        ] == "C0":
            windows = clean_windows(
                trial_meta
            )

            identity_digest = None

            fault_instance_count = 0

            parent_count = observed_trials

        else:
            windows = condition_windows(
                shard=shard,
                slot=slot,
                trial_meta=trial_meta,
                parent_maps=parent_maps,
                runner=runner,
                operators=operators,
                reference_scales=references,
            )

            (
                identity_digest,
                fault_instance_count,
            ) = condition_identity_digest(
                parent_maps,
                slot,
            )

            parent_count = len(
                parent_maps
            )

        if len(
            windows
        ) != observed_windows:
            raise ValueError(
                "condition window count differs from subject inventory"
            )

        for model_variant in (
            "prospective_fp32_300ms",
            "qualified_static_ptq_v7",
        ):
            for seed in (
                42,
                123,
                2025,
            ):
                model = models[
                    (
                        model_variant,
                        seed,
                    )
                ]

                probabilities = infer_probability(
                    runner,
                    model,
                    windows,
                    batch_size=cfg[
                        "execution_parameters"
                    ][
                        "model_batch_size"
                    ],
                )

                if len(
                    probabilities
                ) != observed_windows:
                    raise ValueError(
                        "model probability count mismatch"
                    )

                actual_model_window_evaluations += (
                    observed_windows
                )

                trial_records = split_probabilities(
                    probabilities,
                    trial_meta,
                )

                for operating_point in (
                    "balanced",
                    "low_false_alarm",
                    "timely_150ms",
                ):
                    rule = thresholds[
                        (
                            seed,
                            fold,
                            operating_point,
                        )
                    ]

                    metrics = (
                        compute_condition_metrics(
                            trial_records,
                            threshold=rule[
                                "threshold"
                            ],
                            required_consecutive=rule[
                                "required_consecutive"
                            ],
                            quantile_method=cfg[
                                "execution_parameters"
                            ][
                                "timing_quantile_method"
                            ],
                        )
                    )

                    row = {
                        "shard_id":
                            shard_id,

                        "model_variant":
                            model_variant,

                        "checkpoint_seed":
                            seed,

                        "fold":
                            fold,

                        "subject":
                            subject,

                        "dataset":
                            shard[
                                "dataset"
                            ],

                        "operating_point":
                            operating_point,

                        **condition_base_fields(
                            shard,
                            slot,
                        ),

                        "threshold":
                            rule[
                                "threshold"
                            ],

                        "required_consecutive":
                            rule[
                                "required_consecutive"
                            ],

                        **metrics,

                        "parent_count":
                            int(
                                parent_count
                            ),

                        "fault_instance_count":
                            int(
                                fault_instance_count
                            ),

                        "fault_identity_digest_sha256":
                            identity_digest,

                        "model_window_evaluation_count":
                            int(
                                observed_windows
                            ),

                        "probability_reused_across_operating_points":
                            True,

                        "historical_exception":
                            bool(
                                subject
                                == 106
                            ),

                        "valid_trigger_possible_under_retained_labels":
                            (
                                False
                                if subject == 106
                                else None
                            ),
                    }

                    rows.append(
                        json_safe_row(
                            row
                        )
                    )

    expected_rows = int(
        shard[
            "expected_condition_rows"
        ]
    )

    if len(
        rows
    ) != expected_rows:
        raise ValueError(
            f"condition row count {len(rows)} != {expected_rows}"
        )

    if actual_model_window_evaluations != int(
        shard[
            "expected_model_window_evaluations"
        ]
    ):
        raise ValueError(
            "model-window evaluation count differs from frozen shard"
        )

    rows.sort(
        key=lambda row: (
            row[
                "model_variant"
            ],
            int(
                row[
                    "checkpoint_seed"
                ]
            ),
            row[
                "operating_point"
            ],
            (
                0
                if row[
                    "severity_level"
                ]
                == "C0"
                else severity_order(
                    row[
                        "severity_level"
                    ]
                )
            ),
            row[
                "variant_id"
            ],
            int(
                row[
                    "replicate_index"
                ]
            ),
        )
    )

    condition_file = (
        temp_dir
        / "subject_condition_metrics.jsonl"
    )

    with condition_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                canonical_json(
                    row
                )
                + "\n"
            )

    coverage = {
        "schema_version":
            "phase4h_sensor_fi_outer_shard_coverage_v1",

        "shard_id":
            shard_id,

        "fold":
            fold,

        "subject":
            subject,

        "block":
            shard[
                "block"
            ],

        "expected_condition_rows":
            expected_rows,

        "observed_condition_rows":
            len(
                rows
            ),

        "expected_unique_fault_instances":
            int(
                shard[
                    "expected_unique_fault_instances"
                ]
            ),

        "observed_unique_fault_instances":
            (
                0
                if shard[
                    "block"
                ]
                == "C0"
                else sum(
                    len(
                        mapping
                    )
                    for mapping
                    in parent_maps
                )
            ),

        "expected_model_window_evaluations":
            int(
                shard[
                    "expected_model_window_evaluations"
                ]
            ),

        "observed_model_window_evaluations":
            int(
                actual_model_window_evaluations
            ),

        "trial_count":
            observed_trials,

        "window_count":
            observed_windows,

        "raw_probabilities_persisted":
            False,

        "complete":
            True,
    }

    coverage_file = (
        temp_dir
        / "coverage.json"
    )

    coverage_file.write_text(
        json.dumps(
            coverage,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    output_hashes = {
        "subject_condition_metrics.jsonl":
            sha256_file(
                condition_file
            ),

        "coverage.json":
            sha256_file(
                coverage_file
            ),
    }

    success = {
        "schema_version":
            "phase4h_sensor_fi_outer_shard_success_v1",

        "status":
            "PASS",

        "shard_id":
            shard_id,

        "executor_config_sha256":
            sha256_file(
                cfg_path
            ),

        "executor_implementation_sha256":
            sha256_file(
                __file__
            ),

        "shard_plan_sha256":
            sha256_file(
                plan_path
            ),

        "output_hashes":
            output_hashes,

        "coverage":
            coverage,

        "scientific_boundary": {
            "outer_partition":
                True,

            "fault_family_or_threshold_retuned":
                False,

            "OnField_used":
                False,

            "raw_probabilities_persisted":
                False,
        },
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(
        temp_dir,
        final_dir,
    )

    (
        final_dir
        / "_SUCCESS.json"
    ).write_text(
        json.dumps(
            success,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print(
        f"SHARD_EXECUTION_STATUS=PASS"
    )

    print(
        f"SHARD_ID={shard_id}"
    )

    print(
        f"CONDITION_ROWS={len(rows)}"
    )

    print(
        f"MODEL_WINDOW_EVALUATIONS={actual_model_window_evaluations}"
    )


def dry_run(
    *,
    cfg_path,
    plan_path,
    freeze_path,
    output_path,
):
    cfg = load_json(
        cfg_path
    )

    freeze = validate_freeze(
        cfg_path,
        __file__,
        freeze_path,
    )

    validate_dependency_hashes(
        cfg
    )

    plan = load_json(
        plan_path
    )

    if sha256_file(
        plan_path
    ) != cfg[
        "frozen_dependencies"
    ][
        "outer_v2_plan"
    ][
        "sha256"
    ]:
        raise ValueError(
            "v2 plan hash mismatch"
        )

    if plan[
        "status"
    ] != (
        "FROZEN_PRE_OUTER_EXECUTION"
    ):
        raise ValueError(
            "v2 plan is not frozen"
        )

    if plan[
        "shard_count"
    ] != 793:
        raise ValueError(
            "expected 793 v2 shards"
        )

    # Import only metadata helpers.  No model-loading function is called.
    runner, _, sampling = (
        load_runtime_modules(
            cfg
        )
    )

    thresholds = load_thresholds(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "thresholds"
            ][
                "path"
            ]
        )
    )

    if len(
        thresholds
    ) != 45:
        raise ValueError(
            "threshold dry-run matrix changed"
        )

    severity_protocol = load_json(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "severity_protocol"
            ][
                "path"
            ]
        )
    )

    dataset_root = Path(
        cfg[
            "execution_locations"
        ][
            "dataset_root"
        ]
    )

    _, risk_by_trial = runner.load_risk_index(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "risk_index"
            ][
                "path"
            ]
        )
    )

    # Verify all model artifacts by metadata + SHA only.
    fp32_manifest = load_json(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "fp32_freeze"
            ][
                "path"
            ]
        )
    )

    if len(
        fp32_manifest[
            "checkpoints"
        ]
    ) != 15:
        raise ValueError(
            "expected 15 FP32 checkpoint records"
        )

    fp32_artifact_checks = 0

    for row in fp32_manifest[
        "checkpoints"
    ]:
        path = Path(
            row[
                "checkpoint"
            ]
        )

        if sha256_file(
            path
        ) != row[
            "sha256"
        ]:
            raise ValueError(
                f"FP32 artifact hash mismatch: {path}"
            )

        fp32_artifact_checks += 1

    ptq_manifest = load_json(
        repo_path(
            cfg[
                "frozen_dependencies"
            ][
                "ptq_all15"
            ][
                "path"
            ]
        )
    )

    if len(
        ptq_manifest[
            "members"
        ]
    ) != 15:
        raise ValueError(
            "expected 15 PTQ member records"
        )

    ptq_artifact_checks = 0

    for row in ptq_manifest[
        "members"
    ]:
        path = Path(
            row[
                "artifacts"
            ][
                "torchscript"
            ][
                "path"
            ]
        )

        expected = row[
            "artifacts"
        ][
            "torchscript"
        ][
            "sha256"
        ]

        if sha256_file(
            path
        ) != expected:
            raise ValueError(
                f"PTQ artifact hash mismatch: {path}"
            )

        ptq_artifact_checks += 1

    subject_cache = {}

    source_path_checks = 0
    risk_truth_counts = Counter()

    shard_checks = 0
    slot_checks = 0

    window_blocks = {
        "bias",
        "scale_factor",
        "noise",
        "clipping_saturation",
        "axis_loss",
    }

    sequence_blocks = {
        "drift",
        "stuck_channel",
        "dropout",
        "frame_loss",
        "jitter",
        "delay",
        "orientation",
    }

    for shard in plan[
        "shards"
    ]:
        subject = int(
            shard[
                "subject"
            ]
        )

        fold = int(
            shard[
                "fold"
            ]
        )

        if subject not in subject_cache:
            keys = subject_trial_keys(
                dataset_root,
                subject,
            )

            trial_count = 0
            window_count = 0
            stored_falling_trials = 0
            historical_falling_trials = 0

            representative_sequence = None
            representative_window = None

            for task, trial in keys:
                segments, labels = (
                    runner.load_trial_segments_and_labels(
                        dataset_root,
                        subject=subject,
                        task=task,
                        trial=trial,
                    )
                )

                rr = risk_row(
                    risk_by_trial,
                    subject=subject,
                    task=task,
                    trial=trial,
                )

                stored_fall = bool(
                    np.any(
                        labels == 1
                    )
                )

                historical_fall = bool(
                    stored_fall
                    or rr is not None
                )

                trial_count += 1
                window_count += len(
                    labels
                )

                stored_falling_trials += int(
                    stored_fall
                )

                historical_falling_trials += int(
                    historical_fall
                )

                if representative_window is None:
                    representative_window = (
                        task,
                        trial,
                    )

                source_path = (
                    runner.source_trial_path(
                        subject=subject,
                        task=task,
                        trial=trial,
                        univr_root=cfg[
                            "execution_locations"
                        ][
                            "univr_source_root"
                        ],
                        kfall_root=cfg[
                            "execution_locations"
                        ][
                            "kfall_source_root"
                        ],
                    )
                )

                if not source_path.is_file():
                    raise ValueError(
                        f"missing source trial: {source_path}"
                    )

                source_path_checks += 1

                if representative_sequence is None:
                    # Metadata-only source length. No FI is applied.
                    with source_path.open(
                        "rb"
                    ) as f:
                        line_count = sum(
                            1
                            for _
                            in f
                        )

                    source_length = max(
                        0,
                        line_count
                        - 1
                    )

                    representative_sequence = (
                        task,
                        trial,
                        source_length,
                    )

            if trial_count != int(
                shard[
                    "subject_inventory"
                ][
                    "trials"
                ]
            ):
                raise ValueError(
                    f"subject {subject} trial inventory changed"
                )

            if window_count != int(
                shard[
                    "subject_inventory"
                ][
                    "windows"
                ]
            ):
                raise ValueError(
                    f"subject {subject} window inventory changed"
                )

            stored_activity = (
                trial_count
                - stored_falling_trials
            )

            historical_activity = (
                trial_count
                - historical_falling_trials
            )

            expected_stored = shard[
                "stored_label_trial_inventory"
            ]

            expected_historical = shard[
                "historical_trial_truth_inventory"
            ]

            if {
                "Activity":
                    stored_activity,
                "Falling":
                    stored_falling_trials,
                "total":
                    trial_count,
            } != expected_stored:
                raise ValueError(
                    f"subject {subject} stored-label truth changed"
                )

            if {
                "Activity":
                    historical_activity,
                "Falling":
                    historical_falling_trials,
                "total":
                    trial_count,
            } != expected_historical:
                raise ValueError(
                    f"subject {subject} historical truth changed"
                )

            risk_truth_counts[
                "Activity"
            ] += historical_activity

            risk_truth_counts[
                "Falling"
            ] += historical_falling_trials

            subject_cache[
                subject
            ] = {
                "representative_window":
                    representative_window,

                "representative_sequence":
                    representative_sequence,
            }

        if shard[
            "block"
        ] == "C0":
            expected_slots = 1

        elif shard[
            "block"
        ] in window_blocks:
            task, trial = subject_cache[
                subject
            ][
                "representative_window"
            ]

            rows = (
                sampling.generate_window_instances(
                    fold=fold,
                    partition="outer_test",
                    parent_sequence_id=(
                        f"subject={subject}"
                        f"|task={task}"
                        f"|trial={trial}"
                        f"|window=0"
                    ),
                    severity_protocol=severity_protocol,
                )
            )

            slots = sorted_slots(
                rows,
                family=shard[
                    "block"
                ],
            )

            expected_slots = len(
                slots
            )

        elif shard[
            "block"
        ] in sequence_blocks:
            (
                task,
                trial,
                source_length,
            ) = subject_cache[
                subject
            ][
                "representative_sequence"
            ]

            rows = (
                sampling.generate_sequence_instances(
                    fold=fold,
                    partition="outer_test",
                    parent_sequence_id=(
                        f"subject={subject}"
                        f"|task={task}"
                        f"|trial={trial}"
                    ),
                    parent_length=source_length,
                    severity_protocol=severity_protocol,
                )
            )

            slots = sorted_slots(
                rows,
                family=shard[
                    "block"
                ],
            )

            expected_slots = len(
                slots
            )

        else:
            raise ValueError(
                f"unknown shard block: {shard['block']}"
            )

        if expected_slots != (
            1
            if shard[
                "block"
            ]
            == "C0"
            else int(
                shard[
                    "family_instances_per_parent"
                ]
            )
        ):
            raise ValueError(
                f"slot count mismatch shard={shard['shard_id']}"
            )

        expected_rows = (
            expected_slots
            * 18
        )

        if expected_rows != int(
            shard[
                "expected_condition_rows"
            ]
        ):
            raise ValueError(
                f"condition-row arithmetic mismatch shard={shard['shard_id']}"
            )

        expected_model_eval = (
            int(
                shard[
                    "subject_inventory"
                ][
                    "windows"
                ]
            )
            * expected_slots
            * 6
        )

        if expected_model_eval != int(
            shard[
                "expected_model_window_evaluations"
            ]
        ):
            raise ValueError(
                f"model-evaluation arithmetic mismatch shard={shard['shard_id']}"
            )

        shard_checks += 1
        slot_checks += expected_slots

    if len(
        subject_cache
    ) != 61:
        raise ValueError(
            "dry-run did not cover all 61 subjects"
        )

    # Each subject was encountered first through its C0 shard, so truth
    # totals are accumulated once.
    if dict(
        risk_truth_counts
    ) != {
        "Activity":
            3390,
        "Falling":
            2919,
    }:
        raise ValueError(
            f"historical truth total changed: {dict(risk_truth_counts)}"
        )

    if shard_checks != 793:
        raise ValueError(
            "dry-run did not inspect exactly 793 shards"
        )

    report = {
        "schema_version":
            "phase4h_sensor_fi_outer_executor_v1_dry_run",

        "status":
            "PASS",

        "executor_config_sha256":
            sha256_file(
                cfg_path
            ),

        "executor_implementation_sha256":
            sha256_file(
                __file__
            ),

        "executor_freeze_sha256":
            sha256_file(
                freeze_path
            ),

        "outer_v2_plan_sha256":
            sha256_file(
                plan_path
            ),

        "coverage": {
            "shards_checked":
                shard_checks,

            "subjects_checked":
                len(
                    subject_cache
                ),

            "source_paths_checked":
                source_path_checks,

            "condition_slots_checked":
                slot_checks,

            "threshold_rows_checked":
                len(
                    thresholds
                ),

            "fp32_artifacts_hashed":
                fp32_artifact_checks,

            "ptq_artifacts_hashed":
                ptq_artifact_checks,

            "historical_truth":
                dict(
                    risk_truth_counts
                ),

            "planned_unique_fault_instances":
                plan[
                    "expected_totals_from_shards"
                ][
                    "unique_fault_instances"
                ],

            "planned_model_window_evaluations":
                plan[
                    "expected_totals_from_shards"
                ][
                    "model_window_evaluations"
                ],

            "planned_subject_condition_rows":
                plan[
                    "expected_totals_from_shards"
                ][
                    "subject_condition_rows"
                ],
        },

        "scientific_boundary": {
            "model_weights_loaded":
                False,

            "model_forward_called":
                False,

            "fault_operator_called":
                False,

            "outer_performance_computed":
                False,

            "outer_model_outcomes_seen":
                False,

            "outer_fault_outcomes_seen":
                False,

            "OnField_used":
                False,
        },

        "next_action":
            (
                "After post-dry-run qualification, execute immutable v2 "
                "shards without changing this executor/config/plan."
            ),
    }

    Path(
        output_path
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print(
        "OUTER_EXECUTOR_DRY_RUN_STATUS=PASS"
    )

    print(
        f"SHARDS_CHECKED={shard_checks}"
    )

    print(
        f"SUBJECTS_CHECKED={len(subject_cache)}"
    )

    print(
        f"SOURCE_PATHS_CHECKED={source_path_checks}"
    )

    print(
        f"CONDITION_SLOTS_CHECKED={slot_checks}"
    )

    print(
        f"THRESHOLD_ROWS_CHECKED={len(thresholds)}"
    )

    print(
        f"FP32_ARTIFACTS_HASHED={fp32_artifact_checks}"
    )

    print(
        f"PTQ_ARTIFACTS_HASHED={ptq_artifact_checks}"
    )

    print(
        "MODEL_WEIGHTS_LOADED=False"
    )

    print(
        "MODEL_FORWARD_CALLED=False"
    )

    print(
        "FAULT_OPERATOR_CALLED=False"
    )

    print(
        "OUTER_PERFORMANCE_COMPUTED=False"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "dry-run",
            "execute-shard",
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--freeze",
        required=True,
    )

    parser.add_argument(
        "--dry-run-output",
    )

    parser.add_argument(
        "--shard-id",
    )

    parser.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    if args.mode == "dry-run":
        if not args.dry_run_output:
            raise SystemExit(
                "--dry-run-output is required in dry-run mode"
            )

        if args.shard_id is not None:
            raise SystemExit(
                "--shard-id is prohibited in dry-run mode"
            )

        dry_run(
            cfg_path=args.config,
            plan_path=args.plan,
            freeze_path=args.freeze,
            output_path=args.dry_run_output,
        )

        return

    if not args.shard_id:
        raise SystemExit(
            "--shard-id is required in execute-shard mode"
        )

    if args.dry_run_output is not None:
        raise SystemExit(
            "--dry-run-output is prohibited in execute-shard mode"
        )

    execute_shard(
        cfg_path=args.config,
        plan_path=args.plan,
        freeze_path=args.freeze,
        shard_id=args.shard_id,
        recompute_partial=args.recompute_partial,
    )


if __name__ == "__main__":
    main()
