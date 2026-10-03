"""Static Phase-4H outer execution planner.

No model loading, no model forward pass, no sensor mutation, and no
robustness result computation occur in this module.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
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


def parse_subject(token):
    text = str(
        token
    ).strip()

    upper = text.upper()

    match = re.search(
        r"(\d+)$",
        upper,
    )

    if not match:
        raise ValueError(
            f"cannot parse subject token: {token!r}"
        )

    value = int(
        match.group(1)
    )

    if upper.startswith(
        "KFALL"
    ):
        return 100 + value

    if upper.startswith(
        "UNIVR"
    ):
        return value

    if upper.isdigit():
        return int(
            upper
        )

    raise ValueError(
        f"unknown subject token: {token!r}"
    )


def dataset_name(subject):
    return (
        "KFALL"
        if int(
            subject
        ) >= 100
        else "UNIVR"
    )


def load_fold_subjects(checkpoint_manifest):
    with Path(
        checkpoint_manifest
    ).open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    result = {}

    for fold in range(
        1,
        6,
    ):
        members = [
            row
            for row in rows
            if int(
                row["fold"]
            )
            == fold
        ]

        if len(
            members
        ) != 3:
            raise ValueError(
                f"expected 3 checkpoint seeds for fold {fold}"
            )

        sets = {
            tuple(
                sorted(
                    parse_subject(
                        token
                    )
                    for token
                    in row[
                        "test_subjects"
                    ].split("|")
                    if token
                )
            )
            for row
            in members
        }

        if len(
            sets
        ) != 1:
            raise ValueError(
                f"test-subject sets differ across seeds for fold {fold}"
            )

        result[
            fold
        ] = next(
            iter(
                sets
            )
        )

    all_subjects = [
        subject
        for fold
        in result.values()
        for subject
        in fold
    ]

    if (
        len(
            all_subjects
        )
        != 61
        or len(
            set(
                all_subjects
            )
        )
        != 61
    ):
        raise ValueError(
            "outer subject assignment is not exactly 61 unique subjects"
        )

    return result


def subject_inventory(
    dataset_root,
    subject,
):
    root = (
        Path(
            dataset_root
        )
        / str(
            int(
                subject
            )
        )
    )

    stats = Counter()

    if not root.is_dir():
        raise ValueError(
            f"missing subject directory: {root}"
        )

    stats[
        "subject"
    ] = int(
        subject
    )

    for task_dir in sorted(
        p
        for p
        in root.iterdir()
        if p.is_dir()
    ):
        for trial_dir in sorted(
            p
            for p
            in task_dir.iterdir()
            if p.is_dir()
        ):
            segment_file = (
                trial_dir
                / "segments.npy"
            )

            label_file = (
                trial_dir
                / "labels.npy"
            )

            if not (
                segment_file.is_file()
                and label_file.is_file()
            ):
                continue

            segments = np.load(
                segment_file,
                mmap_mode="r",
                allow_pickle=False,
            )

            raw = np.load(
                label_file,
                allow_pickle=True,
            )

            if (
                segments.ndim != 3
                or tuple(
                    segments.shape[1:]
                )
                != (
                    30,
                    9,
                )
                or len(
                    segments
                )
                != len(
                    raw
                )
            ):
                raise ValueError(
                    f"invalid stored trial: {trial_dir}"
                )

            labels = np.asarray([
                1
                if "FALL" in (
                    value.decode(
                        "utf-8",
                        errors="replace",
                    )
                    if isinstance(
                        value,
                        bytes,
                    )
                    else str(
                        value
                    )
                ).upper()
                else 0
                for value
                in raw
            ], dtype=np.int8)

            stats[
                "trials"
            ] += 1

            stats[
                "windows"
            ] += len(
                labels
            )

            activity = int(
                np.sum(
                    labels == 0
                )
            )

            falling = int(
                np.sum(
                    labels == 1
                )
            )

            stats[
                "activity_windows"
            ] += activity

            stats[
                "falling_windows"
            ] += falling

            if falling:
                stats[
                    "falling_trials"
                ] += 1
            else:
                stats[
                    "activity_only_trials"
                ] += 1

    return {
        key:
            int(
                value
            )
        for key, value
        in stats.items()
    }


def load_sampling_counts(
    sampling_impl,
    severity_protocol,
):
    spec = importlib.util.spec_from_file_location(
        "_outer_plan_sampling",
        sampling_impl,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    severity = json.loads(
        Path(
            severity_protocol
        ).read_text()
    )

    window = module.generate_window_instances(
        fold=1,
        partition="outer_test",
        parent_sequence_id=(
            "subject=9|task=20|trial=1|window=0"
        ),
        severity_protocol=severity,
    )

    sequence = module.generate_sequence_instances(
        fold=1,
        partition="outer_test",
        parent_sequence_id=(
            "subject=9|task=20|trial=1"
        ),
        parent_length=1000,
        severity_protocol=severity,
    )

    counts = {}

    for rows in [
        window,
        sequence,
    ]:
        family_counts = Counter(
            row[
                "family"
            ]
            for row
            in rows
        )

        for family, count in (
            family_counts.items()
        ):
            if family in counts:
                raise ValueError(
                    f"family appears in both domains: {family}"
                )

            counts[
                family
            ] = int(
                count
            )

    if len(
        window
    ) != 153:
        raise ValueError(
            "window-family instance cardinality changed"
        )

    if len(
        sequence
    ) != 138:
        raise ValueError(
            "sequence-family instance cardinality changed"
        )

    return counts


def validate_thresholds(
    threshold_path,
):
    with Path(
        threshold_path
    ).open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(
        rows
    ) != 45:
        raise ValueError(
            "expected 45 frozen threshold rows"
        )

    expected_names = {
        "balanced",
        "low_false_alarm",
        "timely_150ms",
    }

    keys = set()

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

        if key in keys:
            raise ValueError(
                f"duplicate threshold key: {key}"
            )

        keys.add(
            key
        )

        if key[
            2
        ] not in expected_names:
            raise ValueError(
                f"unexpected operating point: {key[2]}"
            )

        threshold = float(
            row[
                "threshold"
            ]
        )

        consecutive = int(
            row[
                "required_consecutive"
            ]
        )

        if not (
            0.0
            <= threshold
            <= 1.0
        ):
            raise ValueError(
                f"invalid threshold: {threshold}"
            )

        if consecutive < 1:
            raise ValueError(
                f"invalid required_consecutive: {consecutive}"
            )

    expected_keys = {
        (
            seed,
            fold,
            op,
        )
        for seed
        in [
            42,
            123,
            2025,
        ]
        for fold
        in range(
            1,
            6,
        )
        for op
        in expected_names
    }

    if keys != expected_keys:
        raise ValueError(
            "threshold seed/fold/operating-point matrix incomplete"
        )

    return rows


def shard_id(
    namespace,
    *,
    fold,
    subject,
    block,
):
    payload = (
        f"{namespace}|"
        f"fold={int(fold)}|"
        f"subject={int(subject)}|"
        f"block={block}"
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[
        :12
    ]

    return (
        f"p4h-o1-"
        f"f{int(fold)}-"
        f"s{int(subject):03d}-"
        f"{block}-"
        f"{digest}"
    )


def build_plan(
    config_path,
):
    cfg = json.loads(
        Path(
            config_path
        ).read_text()
    )

    load_thresholds = validate_thresholds(
        cfg[
            "operating_points"
        ][
            "source"
        ]
    )

    _ = load_thresholds

    checkpoint_path = Path(
        cfg[
            "lineage"
        ][
            "checkpoint_manifest"
        ][
            "path"
        ]
    )

    fold_subjects = load_fold_subjects(
        checkpoint_path
    )

    family_counts = load_sampling_counts(
        ROOT
        / cfg[
            "lineage"
        ][
            "sampling_impl"
        ][
            "path"
        ],
        ROOT
        / cfg[
            "lineage"
        ][
            "severity_protocol"
        ][
            "path"
        ],
    )

    family_order = cfg[
        "fault_matrix"
    ][
        "fault_families_in_order"
    ]

    if set(
        family_counts
    ) != set(
        family_order
    ):
        raise ValueError(
            "sampling family set differs from frozen outer family set"
        )

    window_families = set(
        cfg[
            "fault_matrix"
        ][
            "window_value_families"
        ]
    )

    sequence_families = set(
        cfg[
            "fault_matrix"
        ][
            "sequence_families"
        ]
    )

    namespace = cfg[
        "sharding"
    ][
        "namespace"
    ]

    dataset_root = cfg[
        "lineage"
    ].get(
        "dataset_root",
        cfg.get(
            "dataset_root"
        ),
    )

    # Config intentionally keeps the absolute stored-window root outside
    # lineage because it is an execution location rather than source code.
    if dataset_root is None:
        dataset_root = cfg[
            "_runtime_dataset_root"
        ]

    shards = []

    global_inventory = Counter()

    subject_inventory_rows = {}

    for fold in range(
        1,
        6,
    ):
        for subject in fold_subjects[
            fold
        ]:
            inv = subject_inventory(
                dataset_root,
                subject,
            )

            subject_inventory_rows[
                int(
                    subject
                )
            ] = inv

            global_inventory.update({
                "subjects":
                    1,

                "trials":
                    inv[
                        "trials"
                    ],

                "windows":
                    inv[
                        "windows"
                    ],

                "activity_windows":
                    inv[
                        "activity_windows"
                    ],

                "falling_windows":
                    inv[
                        "falling_windows"
                    ],

                "activity_only_trials":
                    inv[
                        "activity_only_trials"
                    ],

                "falling_trials":
                    inv[
                        "falling_trials"
                    ],
            })

            blocks = [
                "C0",
                *family_order,
            ]

            for block in blocks:
                if block == "C0":
                    parent_kind = (
                        "stored_or_reconstructed_clean_trial_windows"
                    )

                    per_parent = 0

                    fault_instances = 0

                    model_evaluations = (
                        inv[
                            "windows"
                        ]
                        * 6
                    )

                    condition_rows = 18

                else:
                    per_parent = (
                        family_counts[
                            block
                        ]
                    )

                    if block in window_families:
                        parent_kind = (
                            "stored_window"
                        )

                        parent_count = (
                            inv[
                                "windows"
                            ]
                        )

                        fault_instances = (
                            parent_count
                            * per_parent
                        )

                    elif block in sequence_families:
                        parent_kind = (
                            "source_trial"
                        )

                        parent_count = (
                            inv[
                                "trials"
                            ]
                        )

                        fault_instances = (
                            parent_count
                            * per_parent
                        )

                    else:
                        raise ValueError(
                            f"family domain missing: {block}"
                        )

                    model_evaluations = (
                        inv[
                            "windows"
                        ]
                        * per_parent
                        * 6
                    )

                    condition_rows = (
                        per_parent
                        * 18
                    )

                shards.append({
                    "shard_id":
                        shard_id(
                            namespace,
                            fold=fold,
                            subject=subject,
                            block=block,
                        ),

                    "fold":
                        int(
                            fold
                        ),

                    "subject":
                        int(
                            subject
                        ),

                    "dataset":
                        dataset_name(
                            subject
                        ),

                    "block":
                        block,

                    "regime":
                        (
                            "C0"
                            if block == "C0"
                            else "CS"
                        ),

                    "parent_kind":
                        parent_kind,

                    "subject_inventory":
                        inv,

                    "family_instances_per_parent":
                        int(
                            per_parent
                        ),

                    "expected_unique_fault_instances":
                        int(
                            fault_instances
                        ),

                    "expected_model_window_evaluations":
                        int(
                            model_evaluations
                        ),

                    "expected_condition_rows":
                        int(
                            condition_rows
                        ),

                    "checkpoint_seeds":
                        [
                            42,
                            123,
                            2025,
                        ],

                    "model_variants":
                        [
                            "prospective_fp32_300ms",
                            "qualified_static_ptq_v7",
                        ],

                    "operating_points":
                        [
                            "balanced",
                            "low_false_alarm",
                            "timely_150ms",
                        ],
                })

    return (
        shards,
        subject_inventory_rows,
        dict(
            global_inventory
        ),
        family_counts,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    cfg = json.loads(
        config_path.read_text()
    )

    # Bind runtime dataset location without changing the frozen file.
    cfg[
        "_runtime_dataset_root"
    ] = (
        "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
        "Protechto-master_ori/data/"
        "UniVrFall_KFall_NoOF/segments/"
        "300ms_50ov_npseg_filt_binary"
    )

    # build_plan reads config from disk; use a local implementation of the
    # one runtime-only path rather than writing any temporary protocol file.
    original = json.loads(
        config_path.read_text()
    )

    if "dataset_root" not in original:
        original[
            "dataset_root"
        ] = cfg[
            "_runtime_dataset_root"
        ]

    # Replicate build_plan with immutable in-memory config by using the
    # dataset path as an explicit execution-location override.
    fold_subjects = load_fold_subjects(
        original[
            "lineage"
        ][
            "checkpoint_manifest"
        ][
            "path"
        ]
    )

    validate_thresholds(
        original[
            "operating_points"
        ][
            "source"
        ]
    )

    family_counts = load_sampling_counts(
        ROOT
        / original[
            "lineage"
        ][
            "sampling_impl"
        ][
            "path"
        ],
        ROOT
        / original[
            "lineage"
        ][
            "severity_protocol"
        ][
            "path"
        ],
    )

    family_order = original[
        "fault_matrix"
    ][
        "fault_families_in_order"
    ]

    window_families = set(
        original[
            "fault_matrix"
        ][
            "window_value_families"
        ]
    )

    sequence_families = set(
        original[
            "fault_matrix"
        ][
            "sequence_families"
        ]
    )

    namespace = original[
        "sharding"
    ][
        "namespace"
    ]

    dataset_root = original[
        "dataset_root"
    ]

    shards = []
    subject_inventory_rows = {}
    global_inventory = Counter()

    for fold in range(
        1,
        6,
    ):
        for subject in fold_subjects[
            fold
        ]:
            inv = subject_inventory(
                dataset_root,
                subject,
            )

            subject_inventory_rows[
                str(
                    int(
                        subject
                    )
                )
            ] = inv

            global_inventory.update({
                "subjects":
                    1,
                "trials":
                    inv[
                        "trials"
                    ],
                "windows":
                    inv[
                        "windows"
                    ],
                "activity_windows":
                    inv[
                        "activity_windows"
                    ],
                "falling_windows":
                    inv[
                        "falling_windows"
                    ],
                "activity_only_trials":
                    inv[
                        "activity_only_trials"
                    ],
                "falling_trials":
                    inv[
                        "falling_trials"
                    ],
            })

            for block in [
                "C0",
                *family_order,
            ]:
                if block == "C0":
                    parent_kind = (
                        "stored_or_reconstructed_clean_trial_windows"
                    )

                    per_parent = 0
                    fault_instances = 0

                    model_evaluations = (
                        inv[
                            "windows"
                        ]
                        * 6
                    )

                    condition_rows = 18

                elif block in window_families:
                    parent_kind = (
                        "stored_window"
                    )

                    per_parent = int(
                        family_counts[
                            block
                        ]
                    )

                    fault_instances = (
                        inv[
                            "windows"
                        ]
                        * per_parent
                    )

                    model_evaluations = (
                        inv[
                            "windows"
                        ]
                        * per_parent
                        * 6
                    )

                    condition_rows = (
                        per_parent
                        * 18
                    )

                elif block in sequence_families:
                    parent_kind = (
                        "source_trial"
                    )

                    per_parent = int(
                        family_counts[
                            block
                        ]
                    )

                    fault_instances = (
                        inv[
                            "trials"
                        ]
                        * per_parent
                    )

                    model_evaluations = (
                        inv[
                            "windows"
                        ]
                        * per_parent
                        * 6
                    )

                    condition_rows = (
                        per_parent
                        * 18
                    )

                else:
                    raise ValueError(
                        f"unknown block: {block}"
                    )

                shards.append({
                    "shard_id":
                        shard_id(
                            namespace,
                            fold=fold,
                            subject=subject,
                            block=block,
                        ),

                    "fold":
                        fold,

                    "subject":
                        int(
                            subject
                        ),

                    "dataset":
                        dataset_name(
                            subject
                        ),

                    "block":
                        block,

                    "regime":
                        (
                            "C0"
                            if block == "C0"
                            else "CS"
                        ),

                    "parent_kind":
                        parent_kind,

                    "subject_inventory":
                        inv,

                    "family_instances_per_parent":
                        per_parent,

                    "expected_unique_fault_instances":
                        int(
                            fault_instances
                        ),

                    "expected_model_window_evaluations":
                        int(
                            model_evaluations
                        ),

                    "expected_condition_rows":
                        int(
                            condition_rows
                        ),

                    "checkpoint_seeds":
                        [
                            42,
                            123,
                            2025,
                        ],

                    "model_variants":
                        [
                            "prospective_fp32_300ms",
                            "qualified_static_ptq_v7",
                        ],

                    "operating_points":
                        [
                            "balanced",
                            "low_false_alarm",
                            "timely_150ms",
                        ],
                })

    plan = {
        "schema_version":
            "phase4h_sensor_fi_outer_execution_shards_v1",

        "status":
            "FROZEN_PRE_OUTER_EXECUTION",

        "partition":
            "outer_test",

        "namespace":
            namespace,

        "execution_config": {
            "path":
                str(
                    config_path.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    config_path
                ),
        },

        "shard_count":
            len(
                shards
            ),

        "subject_count":
            len(
                subject_inventory_rows
            ),

        "execution_blocks_per_subject":
            13,

        "family_instance_counts_per_parent":
            dict(
                sorted(
                    family_counts.items()
                )
            ),

        "global_outer_inventory":
            dict(
                global_inventory
            ),

        "expected_totals_from_shards": {
            "unique_fault_instances":
                int(
                    sum(
                        row[
                            "expected_unique_fault_instances"
                        ]
                        for row
                        in shards
                    )
                ),

            "model_window_evaluations":
                int(
                    sum(
                        row[
                            "expected_model_window_evaluations"
                        ]
                        for row
                        in shards
                    )
                ),

            "subject_condition_rows":
                int(
                    sum(
                        row[
                            "expected_condition_rows"
                        ]
                        for row
                        in shards
                    )
                ),
        },

        "subject_inventory":
            subject_inventory_rows,

        "shards":
            shards,

        "scientific_boundary": {
            "model_loaded":
                False,

            "model_predictions_computed":
                False,

            "fault_injection_executed":
                False,

            "outer_labels_used_for_inventory_only":
                True,

            "outer_model_outcomes_seen":
                False,

            "OnField_used":
                False,
        },
    }

    Path(
        args.output
    ).write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print(
        "SHARD_PLAN_STATUS=",
        plan[
            "status"
        ],
        sep="",
    )

    print(
        "SHARD_COUNT=",
        plan[
            "shard_count"
        ],
        sep="",
    )

    print(
        "SUBJECT_COUNT=",
        plan[
            "subject_count"
        ],
        sep="",
    )

    print(
        "UNIQUE_FAULT_INSTANCES=",
        plan[
            "expected_totals_from_shards"
        ][
            "unique_fault_instances"
        ],
        sep="",
    )

    print(
        "MODEL_WINDOW_EVALUATIONS=",
        plan[
            "expected_totals_from_shards"
        ][
            "model_window_evaluations"
        ],
        sep="",
    )

    print(
        "SUBJECT_CONDITION_ROWS=",
        plan[
            "expected_totals_from_shards"
        ][
            "subject_condition_rows"
        ],
        sep="",
    )
