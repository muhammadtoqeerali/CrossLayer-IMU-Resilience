#!/usr/bin/env python3
"""Phase-4I immutable OnField Activity-only external executor v1.

The executor implements the already-frozen Phase-4I protocol.

Important scientific boundary:
- all 15 seed×fold members are evaluated;
- all three frozen validation-selected operating points are reported;
- OnField does not select checkpoints, thresholds, or operating points;
- no probability-level ensemble is created;
- uncertainty is subject-level only;
- fall-side metrics are prohibited because retained OnField is Activity-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]

MODEL_VARIANTS = (
    "prospective_fp32_300ms",
    "qualified_static_ptq_v7",
)

SEEDS = (
    42,
    123,
    2025,
)

FOLDS = (
    1,
    2,
    3,
    4,
    5,
)

OPERATING_POINTS = (
    "balanced",
    "low_false_alarm",
    "timely_150ms",
)

PRIMARY_METRICS = (
    "activity_specificity",
    "false_triggers_per_activity_hour",
)


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text()
    )


def dump_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def resolve_repo_path(value):
    p = Path(value)

    if p.is_absolute():
        return p

    return (
        REPO
        / p
    ).resolve()


def validate_bound_file(record):
    path = resolve_repo_path(
        record["path"]
    )

    if not path.is_file():
        raise RuntimeError(
            f"missing bound file: {path}"
        )

    observed = sha256_file(
        path
    )

    expected = str(
        record["sha256"]
    )

    if observed != expected:
        raise RuntimeError(
            "bound file hash mismatch: "
            f"{path}; expected={expected}; "
            f"observed={observed}"
        )

    return path


def validate_executor_config(cfg):
    if cfg[
        "schema_version"
    ] != "phase4i_onfield_external_executor_v1":
        raise RuntimeError(
            "unexpected executor schema"
        )

    validate_bound_file(
        cfg["protocol"]
    )

    validate_bound_file(
        cfg["protocol_freeze"]
    )

    validate_bound_file(
        cfg["root_binding"]
    )

    for record in cfg[
        "reused_runtime"
    ].values():
        validate_bound_file(
            record
        )

    boundary = cfg[
        "scientific_boundary"
    ]

    if any(
        boundary[key]
        for key in (
            "onfield_training",
            "onfield_validation",
            "onfield_calibration",
            "onfield_threshold_tuning",
            "onfield_checkpoint_selection",
            "onfield_operating_point_selection",
            "onfield_model_weighting",
            "fall_side_claims",
            "fp32_ptq_equivalence_claim",
            "global_binary_robustness_label",
        )
    ):
        raise RuntimeError(
            "scientific boundary violation"
        )


def load_runtime(cfg):
    outer_path = validate_bound_file(
        cfg[
            "reused_runtime"
        ][
            "outer_executor"
        ]
    )

    outer_cfg_path = validate_bound_file(
        cfg[
            "reused_runtime"
        ][
            "outer_executor_config"
        ]
    )

    hist_path = validate_bound_file(
        cfg[
            "reused_runtime"
        ][
            "historical_evaluator"
        ]
    )

    outer = load_module(
        "_phase4i_reused_outer_executor",
        outer_path,
    )

    outer_cfg = load_json(
        outer_cfg_path
    )

    runner_path = outer.repo_path(
        outer_cfg[
            "frozen_dependencies"
        ][
            "runner_v2"
        ][
            "path"
        ]
    )

    runner = outer.load_module(
        "_phase4i_reused_runner",
        runner_path,
    )

    historical = load_module(
        "_phase4i_historical_evaluator",
        hist_path,
    )

    return (
        outer,
        outer_cfg,
        runner,
        historical,
    )


def validate_protocol(protocol):
    if protocol[
        "status"
    ] != "FROZEN_PRE_ONFIELD_MODEL_INFERENCE":
        raise RuntimeError(
            "Phase-4I protocol not frozen"
        )

    dataset = protocol[
        "dataset"
    ]

    if dataset[
        "retained_storage_ids"
    ] != [
        str(x)
        for x in range(
            1001,
            1011,
        )
    ]:
        raise RuntimeError(
            "retained OnField subject IDs changed"
        )

    if dataset[
        "retained_subject_count"
    ] != 10:
        raise RuntimeError(
            "subject count changed"
        )

    if dataset[
        "retained_trial_count"
    ] != 16:
        raise RuntimeError(
            "trial count changed"
        )

    if dataset[
        "retained_window_count"
    ] != 1023337:
        raise RuntimeError(
            "window count changed"
        )

    if dataset[
        "label_inventory"
    ] != {
        "Activity": 1023337,
        "Falling": 0,
    }:
        raise RuntimeError(
            "Activity-only label inventory changed"
        )

    estate = protocol[
        "model_estate"
    ]

    if estate[
        "seeds"
    ] != list(
        SEEDS
    ):
        raise RuntimeError(
            "seed estate changed"
        )

    if estate[
        "folds"
    ] != list(
        FOLDS
    ):
        raise RuntimeError(
            "fold estate changed"
        )

    if estate[
        "checkpoint_count_per_model_variant"
    ] != 15:
        raise RuntimeError(
            "checkpoint estate changed"
        )

    ops = protocol[
        "operating_points"
    ]

    if ops[
        "names"
    ] != list(
        OPERATING_POINTS
    ):
        raise RuntimeError(
            "operating points changed"
        )

    if len(
        ops["rows"]
    ) != 45:
        raise RuntimeError(
            "threshold matrix changed"
        )

    expected = protocol[
        "expected_execution_counts"
    ]

    required_counts = {
        "model_window_evaluations":
            30700110,

        "threshold_applications":
            92100330,

        "subject_checkpoint_operating_point_rows":
            900,

        "trial_checkpoint_operating_point_rows":
            1440,

        "subject_estate_summary_rows":
            60,

        "cohort_summary_rows":
            6,
    }

    if expected != required_counts:
        raise RuntimeError(
            "expected execution cardinality changed"
        )


def threshold_registry(protocol):
    rows = protocol[
        "operating_points"
    ][
        "rows"
    ]

    out = {}

    for row in rows:
        key = (
            int(
                row["seed"]
            ),
            int(
                row["fold"]
            ),
            str(
                row[
                    "operating_point"
                ]
            ),
        )

        if key in out:
            raise RuntimeError(
                f"duplicate threshold key {key}"
            )

        out[
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

    expected = {
        (
            seed,
            fold,
            op,
        )
        for seed in SEEDS
        for fold in FOLDS
        for op in OPERATING_POINTS
    }

    if set(
        out
    ) != expected:
        raise RuntimeError(
            "incomplete threshold registry"
        )

    threshold_path = Path(
        protocol[
            "operating_points"
        ][
            "threshold_registry_path"
        ]
    )

    if not threshold_path.is_file():
        raise RuntimeError(
            "frozen threshold CSV missing"
        )

    observed = sha256_file(
        threshold_path
    )

    expected_sha = protocol[
        "operating_points"
    ][
        "threshold_registry_sha256"
    ]

    if observed != expected_sha:
        raise RuntimeError(
            "frozen threshold CSV hash mismatch"
        )

    return out


def trial_activity_seconds(
    window_count,
    *,
    window_samples=30,
    stride_samples=15,
    sampling_rate_hz=100.0,
):
    n = int(
        window_count
    )

    if n < 0:
        raise ValueError(
            "window count cannot be negative"
        )

    if n == 0:
        return 0.0

    samples = (
        (
            n - 1
        )
        * int(
            stride_samples
        )
        + int(
            window_samples
        )
    )

    return (
        float(samples)
        / float(
            sampling_rate_hz
        )
    )


def discover_trials(protocol):
    dataset = protocol[
        "dataset"
    ]

    root = Path(
        dataset[
            "canonical_root"
        ]
    )

    if not root.is_dir():
        raise RuntimeError(
            f"canonical OnField root missing: {root}"
        )

    records = []

    for subject in dataset[
        "retained_storage_ids"
    ]:
        subject_root = (
            root
            / subject
        )

        if not subject_root.is_dir():
            raise RuntimeError(
                f"missing retained subject {subject}"
            )

        segment_paths = sorted(
            subject_root.rglob(
                "segments.npy"
            )
        )

        for segment_path in segment_paths:
            labels_path = (
                segment_path.parent
                / "labels.npy"
            )

            if not labels_path.is_file():
                raise RuntimeError(
                    f"missing labels: {segment_path}"
                )

            x = np.load(
                segment_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            y = np.load(
                labels_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if x.ndim != 3:
                raise RuntimeError(
                    f"unexpected segment rank: {segment_path}"
                )

            if tuple(
                x.shape[1:]
            ) != (
                30,
                9,
            ):
                raise RuntimeError(
                    f"unexpected segment shape: {x.shape}"
                )

            if len(
                x
            ) != len(
                y
            ):
                raise RuntimeError(
                    "segment/label length mismatch"
                )

            if not np.all(
                np.asarray(
                    y
                ).astype(str)
                == "Activity"
            ):
                raise RuntimeError(
                    "retained OnField contains non-Activity label"
                )

            records.append({
                "subject":
                    str(
                        subject
                    ),

                "trial_id":
                    str(
                        segment_path.parent.relative_to(
                            subject_root
                        )
                    ),

                "segments_path":
                    str(
                        segment_path
                    ),

                "labels_path":
                    str(
                        labels_path
                    ),

                "window_count":
                    int(
                        len(x)
                    ),

                "activity_seconds":
                    trial_activity_seconds(
                        len(x)
                    ),

                "segments_sha256":
                    sha256_file(
                        segment_path
                    ),

                "labels_sha256":
                    sha256_file(
                        labels_path
                    ),
            })

    records.sort(
        key=lambda row: (
            int(
                row["subject"]
            ),
            row["trial_id"],
        )
    )

    if len(
        records
    ) != 16:
        raise RuntimeError(
            f"expected 16 retained trials, observed {len(records)}"
        )

    total_windows = sum(
        row[
            "window_count"
        ]
        for row in records
    )

    if total_windows != 1023337:
        raise RuntimeError(
            f"expected 1023337 windows, observed {total_windows}"
        )

    if {
        row[
            "subject"
        ]
        for row in records
    } != {
        str(x)
        for x in range(
            1001,
            1011,
        )
    }:
        raise RuntimeError(
            "retained subject set changed"
        )

    return records


def trigger_episodes_reference(
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


def trial_metrics(
    probabilities,
    *,
    threshold,
    required_consecutive,
    trigger_fn,
):
    probs = np.asarray(
        probabilities,
        dtype=float,
    )

    n = int(
        len(probs)
    )

    fp = int(
        np.count_nonzero(
            probs
            >= float(
                threshold
            )
        )
    )

    tn = (
        n
        - fp
    )

    specificity = (
        float(tn)
        / float(n)
        if n
        else float("nan")
    )

    episodes = trigger_fn(
        probs,
        float(
            threshold
        ),
        int(
            required_consecutive
        ),
    )

    seconds = trial_activity_seconds(
        n
    )

    false_rate = (
        float(
            len(
                episodes
            )
        )
        / (
            seconds
            / 3600.0
        )
        if seconds > 0
        else float("nan")
    )

    return {
        "activity_window_count":
            n,

        "true_negative_window_count":
            int(
                tn
            ),

        "false_positive_window_count":
            int(
                fp
            ),

        "activity_specificity":
            float(
                specificity
            ),

        "false_trigger_episode_count":
            int(
                len(
                    episodes
                )
            ),

        "activity_seconds":
            float(
                seconds
            ),

        "false_triggers_per_activity_hour":
            float(
                false_rate
            ),
    }


def aggregate_subject_checkpoint(
    trial_rows,
):
    groups = defaultdict(
        list
    )

    for row in trial_rows:
        key = (
            row[
                "subject"
            ],
            row[
                "model_variant"
            ],
            int(
                row[
                    "checkpoint_seed"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
            row[
                "operating_point"
            ],
        )

        groups[
            key
        ].append(
            row
        )

    out = []

    for key in sorted(
        groups,
        key=lambda k: (
            int(
                k[0]
            ),
            k[1],
            k[2],
            k[3],
            k[4],
        ),
    ):
        rows = groups[
            key
        ]

        subject, model, seed, fold, op = key

        windows = sum(
            int(
                row[
                    "activity_window_count"
                ]
            )
            for row in rows
        )

        tn = sum(
            int(
                row[
                    "true_negative_window_count"
                ]
            )
            for row in rows
        )

        fp = sum(
            int(
                row[
                    "false_positive_window_count"
                ]
            )
            for row in rows
        )

        episodes = sum(
            int(
                row[
                    "false_trigger_episode_count"
                ]
            )
            for row in rows
        )

        seconds = sum(
            float(
                row[
                    "activity_seconds"
                ]
            )
            for row in rows
        )

        if windows != (
            tn
            + fp
        ):
            raise RuntimeError(
                "subject Activity denominator mismatch"
            )

        specificity = (
            float(tn)
            / float(windows)
            if windows
            else float("nan")
        )

        false_rate = (
            float(
                episodes
            )
            / (
                seconds
                / 3600.0
            )
            if seconds > 0
            else float("nan")
        )

        thresholds = {
            float(
                row[
                    "threshold"
                ]
            )
            for row in rows
        }

        consecutives = {
            int(
                row[
                    "required_consecutive"
                ]
            )
            for row in rows
        }

        if (
            len(
                thresholds
            ) != 1
            or len(
                consecutives
            ) != 1
        ):
            raise RuntimeError(
                "subject checkpoint rule inconsistent across trials"
            )

        out.append({
            "subject":
                subject,

            "model_variant":
                model,

            "checkpoint_seed":
                seed,

            "fold":
                fold,

            "operating_point":
                op,

            "threshold":
                next(
                    iter(
                        thresholds
                    )
                ),

            "required_consecutive":
                next(
                    iter(
                        consecutives
                    )
                ),

            "trial_count":
                len(
                    rows
                ),

            "activity_window_count":
                int(
                    windows
                ),

            "true_negative_window_count":
                int(
                    tn
                ),

            "false_positive_window_count":
                int(
                    fp
                ),

            "activity_specificity":
                float(
                    specificity
                ),

            "false_trigger_episode_count":
                int(
                    episodes
                ),

            "activity_seconds":
                float(
                    seconds
                ),

            "false_triggers_per_activity_hour":
                float(
                    false_rate
                ),

            "probability_reused_across_operating_points":
                True,
        })

    return out


def equal_fold_then_seed_mean(
    rows,
    *,
    value_key,
):
    by_seed = defaultdict(
        list
    )

    for row in rows:
        by_seed[
            int(
                row[
                    "checkpoint_seed"
                ]
            )
        ].append(
            (
                int(
                    row[
                        "fold"
                    ]
                ),
                float(
                    row[
                        value_key
                    ]
                ),
            )
        )

    if set(
        by_seed
    ) != set(
        SEEDS
    ):
        raise RuntimeError(
            "seed estate incomplete"
        )

    seed_values = {}

    for seed in SEEDS:
        members = by_seed[
            seed
        ]

        if {
            fold
            for fold, _
            in members
        } != set(
            FOLDS
        ):
            raise RuntimeError(
                f"fold estate incomplete for seed {seed}"
            )

        values = np.asarray(
            [
                value
                for _, value
                in members
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"nonfinite estate metric {value_key}"
            )

        seed_values[
            seed
        ] = float(
            np.mean(
                values
            )
        )

    return (
        float(
            np.mean(
                [
                    seed_values[
                        seed
                    ]
                    for seed
                    in SEEDS
                ]
            )
        ),
        seed_values,
    )


def aggregate_subject_estate(
    subject_checkpoint_rows,
):
    groups = defaultdict(
        list
    )

    for row in subject_checkpoint_rows:
        key = (
            row[
                "subject"
            ],
            row[
                "model_variant"
            ],
            row[
                "operating_point"
            ],
        )

        groups[
            key
        ].append(
            row
        )

    out = []

    for key in sorted(
        groups,
        key=lambda k: (
            int(
                k[0]
            ),
            k[1],
            k[2],
        ),
    ):
        rows = groups[
            key
        ]

        if len(
            rows
        ) != 15:
            raise RuntimeError(
                "expected 15 checkpoint-specific subject rows"
            )

        subject, model, op = key

        specificity, spec_seed = (
            equal_fold_then_seed_mean(
                rows,
                value_key=
                    "activity_specificity",
            )
        )

        false_rate, rate_seed = (
            equal_fold_then_seed_mean(
                rows,
                value_key=
                    "false_triggers_per_activity_hour",
            )
        )

        false_count, count_seed = (
            equal_fold_then_seed_mean(
                rows,
                value_key=
                    "false_trigger_episode_count",
            )
        )

        # Activity denominator is a property of the external
        # subject, not of the checkpoint. Assert it is constant.
        activity_windows = {
            int(
                row[
                    "activity_window_count"
                ]
            )
            for row in rows
        }

        activity_seconds = {
            float(
                row[
                    "activity_seconds"
                ]
            )
            for row in rows
        }

        trial_counts = {
            int(
                row[
                    "trial_count"
                ]
            )
            for row in rows
        }

        if (
            len(
                activity_windows
            ) != 1
            or len(
                activity_seconds
            ) != 1
            or len(
                trial_counts
            ) != 1
        ):
            raise RuntimeError(
                "checkpoint changed external subject denominator"
            )

        out.append({
            "subject":
                subject,

            "model_variant":
                model,

            "operating_point":
                op,

            "checkpoint_count":
                15,

            "seed_count":
                3,

            "fold_count_per_seed":
                5,

            "activity_window_count":
                next(
                    iter(
                        activity_windows
                    )
                ),

            "activity_seconds":
                next(
                    iter(
                        activity_seconds
                    )
                ),

            "trial_count":
                next(
                    iter(
                        trial_counts
                    )
                ),

            "activity_specificity":
                specificity,

            "false_triggers_per_activity_hour":
                false_rate,

            "mean_false_trigger_episode_count_across_checkpoints":
                false_count,

            "activity_specificity_seed_means":
                {
                    str(k):
                        v
                    for k, v
                    in sorted(
                        spec_seed.items()
                    )
                },

            "false_triggers_per_activity_hour_seed_means":
                {
                    str(k):
                        v
                    for k, v
                    in sorted(
                        rate_seed.items()
                    )
                },

            "false_trigger_episode_count_seed_means":
                {
                    str(k):
                        v
                    for k, v
                    in sorted(
                        count_seed.items()
                    )
                },

            "checkpoint_selection":
                False,

            "checkpoint_performance_weighting":
                False,
        })

    return out


def deterministic_seed(
    *parts,
):
    payload = (
        "|".join(
            map(
                str,
                parts,
            )
        )
    ).encode(
        "utf-8"
    )

    return int.from_bytes(
        hashlib.sha256(
            payload
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )


def bootstrap_subject_mean(
    rows,
    *,
    value_key,
    seed_parts,
    replicates=10000,
):
    if len(
        rows
    ) != 10:
        raise RuntimeError(
            "external bootstrap requires exactly 10 subject rows"
        )

    subjects = [
        str(
            row[
                "subject"
            ]
        )
        for row in rows
    ]

    if len(
        set(
            subjects
        )
    ) != 10:
        raise RuntimeError(
            "duplicate external subjects in bootstrap input"
        )

    values = np.asarray(
        [
            float(
                row[
                    value_key
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise RuntimeError(
            f"nonfinite external subject metric {value_key}"
        )

    rng = np.random.default_rng(
        deterministic_seed(
            *seed_parts
        )
    )

    draws = np.empty(
        int(
            replicates
        ),
        dtype=float,
    )

    n = len(
        values
    )

    for i in range(
        int(
            replicates
        )
    ):
        index = rng.integers(
            0,
            n,
            size=n,
        )

        draws[
            i
        ] = float(
            np.mean(
                values[
                    index
                ]
            )
        )

    low, high = np.percentile(
        draws,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "point_estimate":
            float(
                np.mean(
                    values
                )
            ),

        "ci95_low":
            float(
                low
            ),

        "ci95_high":
            float(
                high
            ),

        "eligible_subject_count":
            10,

        "bootstrap_replicates":
            int(
                replicates
            ),

        "sampling_unit":
            "subject",
    }


def cohort_summary(
    subject_estate_rows,
    *,
    replicates=10000,
):
    groups = defaultdict(
        list
    )

    for row in subject_estate_rows:
        key = (
            row[
                "model_variant"
            ],
            row[
                "operating_point"
            ],
        )

        groups[
            key
        ].append(
            row
        )

    out = []

    for key in sorted(
        groups
    ):
        rows = groups[
            key
        ]

        if len(
            rows
        ) != 10:
            raise RuntimeError(
                "expected 10 subjects in cohort summary"
            )

        model, op = key

        metrics = {}

        for metric in PRIMARY_METRICS:
            metrics[
                metric
            ] = bootstrap_subject_mean(
                rows,
                value_key=
                    metric,

                seed_parts=(
                    "phase4i_onfield_external_evaluation_protocol_v1",
                    model,
                    op,
                    metric,
                ),

                replicates=
                    replicates,
            )

        out.append({
            "model_variant":
                model,

            "operating_point":
                op,

            "subject_count":
                10,

            "checkpoint_count_per_model_variant":
                15,

            "activity_specificity":
                metrics[
                    "activity_specificity"
                ],

            "false_triggers_per_activity_hour":
                metrics[
                    "false_triggers_per_activity_hour"
                ],

            "fall_side_metrics_available":
                False,

            "fp32_ptq_equivalence_test":
                False,

            "global_binary_robustness_label":
                False,
        })

    return out


def validate_member_inventory(
    outer,
    outer_cfg,
    runner,
):
    fp32_freeze = outer.repo_path(
        outer_cfg[
            "frozen_dependencies"
        ][
            "fp32_freeze"
        ][
            "path"
        ]
    )

    ptq_all15 = outer.repo_path(
        outer_cfg[
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

    expected = {
        (
            seed,
            fold,
        )
        for seed in SEEDS
        for fold in FOLDS
    }

    if set(
        fp32_members
    ) != expected:
        raise RuntimeError(
            "FP32 member inventory incomplete"
        )

    if set(
        ptq_members
    ) != expected:
        raise RuntimeError(
            "PTQ member inventory incomplete"
        )

    for key in sorted(
        expected
    ):
        fp32 = fp32_members[
            key
        ]

        fp32_path = Path(
            fp32[
                "checkpoint"
            ]
        )

        if not fp32_path.is_file():
            raise RuntimeError(
                f"missing FP32 artifact {key}"
            )

        if sha256_file(
            fp32_path
        ) != fp32[
            "sha256"
        ]:
            raise RuntimeError(
                f"FP32 artifact hash mismatch {key}"
            )

        ptq = ptq_members[
            key
        ]

        torchscript = Path(
            ptq[
                "artifacts"
            ][
                "torchscript"
            ][
                "path"
            ]
        )

        if not torchscript.is_file():
            raise RuntimeError(
                f"missing PTQ artifact {key}"
            )

        if sha256_file(
            torchscript
        ) != ptq[
            "artifacts"
        ][
            "torchscript"
        ][
            "sha256"
        ]:
            raise RuntimeError(
                f"PTQ artifact hash mismatch {key}"
            )

    return (
        fp32_members,
        ptq_members,
    )


def dry_run(
    cfg,
):
    validate_executor_config(
        cfg
    )

    protocol = load_json(
        resolve_repo_path(
            cfg[
                "protocol"
            ][
                "path"
            ]
        )
    )

    validate_protocol(
        protocol
    )

    thresholds = threshold_registry(
        protocol
    )

    trials = discover_trials(
        protocol
    )

    (
        outer,
        outer_cfg,
        runner,
        historical,
    ) = load_runtime(
        cfg
    )

    validate_member_inventory(
        outer,
        outer_cfg,
        runner,
    )

    # Verify historical trigger semantics against a fixed structural
    # example without reading any model output.
    synthetic = np.asarray([
        0.1,
        0.9,
        0.9,
        0.9,
        0.1,
        0.9,
        0.9,
    ])

    expected_episodes = (
        trigger_episodes_reference(
            synthetic,
            0.5,
            2,
        )
    )

    observed_episodes = (
        historical.trigger_episodes(
            synthetic,
            0.5,
            2,
        )
    )

    if observed_episodes != expected_episodes:
        raise RuntimeError(
            "historical trigger semantics changed"
        )

    expected_counts = protocol[
        "expected_execution_counts"
    ]

    observed_window_count = sum(
        trial[
            "window_count"
        ]
        for trial in trials
    )

    if (
        observed_window_count
        * 15
        * 2
        != expected_counts[
            "model_window_evaluations"
        ]
    ):
        raise RuntimeError(
            "model-window execution count mismatch"
        )

    if (
        observed_window_count
        * 15
        * 2
        * 3
        != expected_counts[
            "threshold_applications"
        ]
    ):
        raise RuntimeError(
            "threshold application count mismatch"
        )

    report = {
        "status":
            "PASS",

        "mode":
            "DRY_RUN_NO_MODEL_LOADING_NO_FORWARD",

        "retained_subject_count":
            10,

        "retained_trial_count":
            len(
                trials
            ),

        "retained_window_count":
            observed_window_count,

        "threshold_rows_checked":
            len(
                thresholds
            ),

        "fp32_checkpoint_artifacts_checked":
            15,

        "ptq_checkpoint_artifacts_checked":
            15,

        "historical_trigger_semantics_checked":
            True,

        "expected_model_window_evaluations":
            expected_counts[
                "model_window_evaluations"
            ],

        "expected_threshold_applications":
            expected_counts[
                "threshold_applications"
            ],

        "onfield_model_forward_passes":
            0,

        "onfield_performance_values_read":
            False,

        "probabilities_stored":
            False,

        "checkpoint_selection":
            False,

        "threshold_tuning":
            False,

        "operating_point_selection":
            False,
    }

    return report


def execute(
    cfg,
):
    validate_executor_config(
        cfg
    )

    protocol_path = resolve_repo_path(
        cfg[
            "protocol"
        ][
            "path"
        ]
    )

    protocol = load_json(
        protocol_path
    )

    validate_protocol(
        protocol
    )

    thresholds = threshold_registry(
        protocol
    )

    trials = discover_trials(
        protocol
    )

    (
        outer,
        outer_cfg,
        runner,
        historical,
    ) = load_runtime(
        cfg
    )

    validate_member_inventory(
        outer,
        outer_cfg,
        runner,
    )

    output_root = Path(
        cfg[
            "execution"
        ][
            "output_root"
        ]
    )

    temp_root = Path(
        str(
            output_root
        )
        + cfg[
            "execution"
        ][
            "temporary_suffix"
        ]
    )

    if output_root.exists():
        raise RuntimeError(
            f"immutable final output already exists: {output_root}"
        )

    if temp_root.exists():
        raise RuntimeError(
            f"partial output exists: {temp_root}"
        )

    temp_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    trial_rows = []

    batch_size = int(
        cfg[
            "execution"
        ][
            "batch_size"
        ]
    )

    model_window_evaluations = 0
    threshold_applications = 0

    try:
        for fold in FOLDS:
            models = outer.load_models_for_shard(
                outer_cfg,
                runner,
                fold=fold,
            )

            expected_model_keys = {
                (
                    variant,
                    seed,
                )
                for variant
                in MODEL_VARIANTS
                for seed
                in SEEDS
            }

            if set(
                models
            ) != expected_model_keys:
                raise RuntimeError(
                    f"model key mismatch fold={fold}"
                )

            for variant in MODEL_VARIANTS:
                for seed in SEEDS:
                    model = models[
                        (
                            variant,
                            seed,
                        )
                    ]

                    for trial in trials:
                        x = np.load(
                            trial[
                                "segments_path"
                            ],
                            mmap_mode="r",
                            allow_pickle=False,
                        )

                        probabilities = (
                            outer.infer_probability(
                                runner,
                                model,
                                x,
                                batch_size=
                                    batch_size,
                            )
                        )

                        if len(
                            probabilities
                        ) != trial[
                            "window_count"
                        ]:
                            raise RuntimeError(
                                "probability length mismatch"
                            )

                        if not np.all(
                            np.isfinite(
                                probabilities
                            )
                        ):
                            raise RuntimeError(
                                "nonfinite model probability"
                            )

                        model_window_evaluations += int(
                            trial[
                                "window_count"
                            ]
                        )

                        for op in OPERATING_POINTS:
                            rule = thresholds[
                                (
                                    seed,
                                    fold,
                                    op,
                                )
                            ]

                            metric = trial_metrics(
                                probabilities,
                                threshold=
                                    rule[
                                        "threshold"
                                    ],
                                required_consecutive=
                                    rule[
                                        "required_consecutive"
                                    ],
                                trigger_fn=
                                    historical.trigger_episodes,
                            )

                            threshold_applications += int(
                                trial[
                                    "window_count"
                                ]
                            )

                            trial_rows.append({
                                "subject":
                                    trial[
                                        "subject"
                                    ],

                                "trial_id":
                                    trial[
                                        "trial_id"
                                    ],

                                "model_variant":
                                    variant,

                                "checkpoint_seed":
                                    seed,

                                "fold":
                                    fold,

                                "operating_point":
                                    op,

                                "threshold":
                                    float(
                                        rule[
                                            "threshold"
                                        ]
                                    ),

                                "required_consecutive":
                                    int(
                                        rule[
                                            "required_consecutive"
                                        ]
                                    ),

                                **metric,

                                "probability_reused_across_operating_points":
                                    True,

                                "raw_probabilities_stored":
                                    False,
                            })

                    del model
                    gc.collect()

            models.clear()
            gc.collect()

        expected = protocol[
            "expected_execution_counts"
        ]

        if len(
            trial_rows
        ) != expected[
            "trial_checkpoint_operating_point_rows"
        ]:
            raise RuntimeError(
                "trial result row count mismatch"
            )

        if model_window_evaluations != expected[
            "model_window_evaluations"
        ]:
            raise RuntimeError(
                "model-window evaluation count mismatch"
            )

        if threshold_applications != expected[
            "threshold_applications"
        ]:
            raise RuntimeError(
                "threshold application count mismatch"
            )

        subject_checkpoint_rows = (
            aggregate_subject_checkpoint(
                trial_rows
            )
        )

        if len(
            subject_checkpoint_rows
        ) != expected[
            "subject_checkpoint_operating_point_rows"
        ]:
            raise RuntimeError(
                "subject checkpoint row count mismatch"
            )

        subject_estate_rows = (
            aggregate_subject_estate(
                subject_checkpoint_rows
            )
        )

        if len(
            subject_estate_rows
        ) != expected[
            "subject_estate_summary_rows"
        ]:
            raise RuntimeError(
                "subject estate row count mismatch"
            )

        cohort_rows = cohort_summary(
            subject_estate_rows,
            replicates=
                protocol[
                    "uncertainty"
                ][
                    "bootstrap_replicates"
                ],
        )

        if len(
            cohort_rows
        ) != expected[
            "cohort_summary_rows"
        ]:
            raise RuntimeError(
                "cohort summary row count mismatch"
            )

        trial_file = (
            temp_root
            / "trial_checkpoint_operating_point.jsonl"
        )

        subject_checkpoint_file = (
            temp_root
            / "subject_checkpoint_operating_point.jsonl"
        )

        subject_estate_file = (
            temp_root
            / "subject_estate_summary.jsonl"
        )

        cohort_file = (
            temp_root
            / "cohort_summary.jsonl"
        )

        write_jsonl(
            trial_file,
            trial_rows,
        )

        write_jsonl(
            subject_checkpoint_file,
            subject_checkpoint_rows,
        )

        write_jsonl(
            subject_estate_file,
            subject_estate_rows,
        )

        write_jsonl(
            cohort_file,
            cohort_rows,
        )

        coverage = {
            "status":
                "PASS",

            "external_role":
                "ACTIVITY_ONLY",

            "retained_subject_count":
                10,

            "retained_trial_count":
                16,

            "retained_window_count":
                1023337,

            "record_counts": {
                "trial_checkpoint_operating_point":
                    len(
                        trial_rows
                    ),

                "subject_checkpoint_operating_point":
                    len(
                        subject_checkpoint_rows
                    ),

                "subject_estate_summary":
                    len(
                        subject_estate_rows
                    ),

                "cohort_summary":
                    len(
                        cohort_rows
                    ),
            },

            "model_window_evaluations":
                model_window_evaluations,

            "threshold_applications":
                threshold_applications,

            "all_15_checkpoints_evaluated":
                True,

            "all_3_operating_points_evaluated":
                True,

            "raw_probabilities_stored":
                False,

            "fall_side_metrics_generated":
                False,

            "checkpoint_selection":
                False,

            "threshold_tuning":
                False,

            "operating_point_selection":
                False,

            "subject_level_bootstrap_replicates":
                10000,
        }

        coverage_file = (
            temp_root
            / "coverage.json"
        )

        dump_json(
            coverage_file,
            coverage,
        )

        manifest = {
            "status":
                "PASS",

            "schema_version":
                "phase4i_onfield_external_executor_v1_artifact_manifest",

            "protocol_sha256":
                sha256_file(
                    protocol_path
                ),

            "executor_config_sha256":
                sha256_file(
                    resolve_repo_path(
                        "configs/evaluation/"
                        "phase4i_onfield_external_executor_v1.json"
                    )
                ),

            "files": {},

            "scientific_boundary": {
                "onfield_used_for_tuning":
                    False,

                "checkpoint_selected":
                    False,

                "operating_point_selected":
                    False,

                "fall_side_claims_generated":
                    False,

                "probabilities_stored":
                    False,

                "global_binary_robustness_label":
                    False,
            },
        }

        for path in (
            trial_file,
            subject_checkpoint_file,
            subject_estate_file,
            cohort_file,
            coverage_file,
        ):
            manifest[
                "files"
            ][
                path.name
            ] = {
                "sha256":
                    sha256_file(
                        path
                    ),

                "bytes":
                    path.stat().st_size,
            }

        manifest_file = (
            temp_root
            / "artifact_manifest.json"
        )

        dump_json(
            manifest_file,
            manifest,
        )

        success = {
            "status":
                "PASS",

            "external_role":
                "ACTIVITY_ONLY",

            "immutable_execution_complete":
                True,

            "all_15_checkpoints_evaluated":
                True,

            "all_3_operating_points_evaluated":
                True,

            "subject_level_bootstrap_complete":
                True,

            "fall_side_metrics_generated":
                False,

            "checkpoint_selection":
                False,

            "threshold_tuning":
                False,

            "operating_point_selection":
                False,

            "raw_probabilities_stored":
                False,

            "global_binary_robustness_label":
                False,
        }

        success_file = (
            temp_root
            / "_SUCCESS.json"
        )

        dump_json(
            success_file,
            success,
        )

        os.replace(
            temp_root,
            output_root,
        )

    except Exception:
        raise

    return {
        "status":
            "PASS",

        "output_root":
            str(
                output_root
            ),

        "model_window_evaluations":
            model_window_evaluations,

        "threshold_applications":
            threshold_applications,

        "trial_rows":
            len(
                trial_rows
            ),

        "subject_checkpoint_rows":
            len(
                subject_checkpoint_rows
            ),

        "subject_estate_rows":
            len(
                subject_estate_rows
            ),

        "cohort_rows":
            len(
                cohort_rows
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    cfg_path = Path(
        args.config
    )

    cfg = load_json(
        cfg_path
    )

    if args.dry_run:
        report = dry_run(
            cfg
        )

        print(
            "ONFIELD_EXECUTOR_DRY_RUN=PASS"
        )

        for key, value in report.items():
            print(
                f"{key}: {value}"
            )

        return

    result = execute(
        cfg
    )

    print(
        "ONFIELD_EXTERNAL_EXECUTION=PASS"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()
