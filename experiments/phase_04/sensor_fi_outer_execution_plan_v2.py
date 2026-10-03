"""Phase-4H outer execution planner v2.

This is a metadata-only transformation/audit of the already-qualified
v1 execution plan. It changes one prospective contract before held-out
execution: trial truth follows the recovered historical evaluator,
including the risk-linked KFALL_106_T27_R05 exception.

No model is loaded. No model forward pass occurs. No fault operator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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


def binary_labels(path):
    raw = np.load(
        path,
        allow_pickle=True,
    )

    return np.asarray([
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
            else str(value)
        ).strip().upper()
        else 0
        for value
        in raw
    ], dtype=np.int8)


def risk_subject(row):
    text = str(
        row["subject_id"]
    )

    match = re.search(
        r"(\d+)$",
        text,
    )

    if not match:
        raise ValueError(
            f"cannot parse risk subject_id: {text!r}"
        )

    return int(
        match.group(1)
    )


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
    ).hexdigest()[:12]

    return (
        f"p4h-o2-"
        f"f{int(fold)}-"
        f"s{int(subject):03d}-"
        f"{block}-"
        f"{digest}"
    )


def build(
    *,
    config_path,
):
    cfg = json.loads(
        Path(
            config_path
        ).read_text()
    )

    supersedes = cfg[
        "supersedes"
    ]

    v1_config = (
        ROOT
        / supersedes[
            "v1_config_path"
        ]
    )

    v1_plan_path = (
        ROOT
        / supersedes[
            "v1_shard_plan_path"
        ]
    )

    v1_qual = (
        ROOT
        / supersedes[
            "v1_qualification_path"
        ]
    )

    for path, expected in [
        (
            v1_config,
            supersedes[
                "v1_config_sha256"
            ],
        ),
        (
            v1_plan_path,
            supersedes[
                "v1_shard_plan_sha256"
            ],
        ),
        (
            v1_qual,
            supersedes[
                "v1_qualification_sha256"
            ],
        ),
    ]:
        if sha256_file(
            path
        ) != expected:
            raise ValueError(
                f"v1 prerequisite hash mismatch: {path}"
            )

    v1_plan = json.loads(
        v1_plan_path.read_text()
    )

    if (
        v1_plan[
            "shard_count"
        ]
        != 793
        or len(
            v1_plan[
                "shards"
            ]
        )
        != 793
    ):
        raise ValueError(
            "v1 shard plan cardinality changed"
        )

    dataset_root = Path(
        cfg[
            "execution_locations"
        ][
            "stored_window_dataset_root"
        ]
    )

    risk_path = Path(
        cfg[
            "ground_truth_contract"
        ][
            "risk_index"
        ][
            "path"
        ]
    )

    if sha256_file(
        risk_path
    ) != cfg[
        "ground_truth_contract"
    ][
        "risk_index"
    ][
        "sha256"
    ]:
        raise ValueError(
            "risk index hash mismatch"
        )

    risk = pd.read_csv(
        risk_path
    )

    risk_by_trial = defaultdict(
        list
    )

    for row in risk.to_dict(
        orient="records"
    ):
        key = (
            risk_subject(
                row
            ),
            int(
                row[
                    "task_id"
                ]
            ),
            int(
                row[
                    "trial_id"
                ]
            ),
        )

        risk_by_trial[
            key
        ].append(
            row
        )

    duplicate_risk = {
        key:
            len(
                rows
            )
        for key, rows
        in risk_by_trial.items()
        if len(
            rows
        )
        != 1
    }

    if duplicate_risk:
        raise ValueError(
            f"duplicate risk rows: {list(duplicate_risk.items())[:10]}"
        )

    subject_folds = {}

    for row in v1_plan[
        "shards"
    ]:
        subject = int(
            row[
                "subject"
            ]
        )

        fold = int(
            row[
                "fold"
            ]
        )

        previous = subject_folds.setdefault(
            subject,
            fold,
        )

        if previous != fold:
            raise ValueError(
                f"subject {subject} appears in multiple folds"
            )

    if len(
        subject_folds
    ) != 61:
        raise ValueError(
            "expected 61 subjects"
        )

    stored_global = Counter()
    historical_global = Counter()

    truth_by_subject = {}

    mismatch_rows = []

    outer_trial_keys = set()

    for subject in sorted(
        subject_folds
    ):
        subject_dir = (
            dataset_root
            / str(
                subject
            )
        )

        stored_subject = Counter()
        historical_subject = Counter()

        for task_dir in sorted(
            p
            for p
            in subject_dir.iterdir()
            if p.is_dir()
        ):
            task = int(
                task_dir.name
            )

            for trial_dir in sorted(
                p
                for p
                in task_dir.iterdir()
                if p.is_dir()
            ):
                trial = int(
                    trial_dir.name
                )

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

                key = (
                    subject,
                    task,
                    trial,
                )

                outer_trial_keys.add(
                    key
                )

                labels = binary_labels(
                    label_file
                )

                stored_fall = bool(
                    np.any(
                        labels == 1
                    )
                )

                has_risk = (
                    key
                    in risk_by_trial
                )

                historical_fall = bool(
                    stored_fall
                    or has_risk
                )

                stored_subject[
                    "Falling"
                    if stored_fall
                    else "Activity"
                ] += 1

                historical_subject[
                    "Falling"
                    if historical_fall
                    else "Activity"
                ] += 1

                if stored_fall != historical_fall:
                    row = risk_by_trial[
                        key
                    ][
                        0
                    ]

                    mismatch_rows.append({
                        "fold":
                            subject_folds[
                                subject
                            ],

                        "subject":
                            subject,

                        "task":
                            task,

                        "trial":
                            trial,

                        "event_id":
                            row.get(
                                "event_id"
                            ),

                        "stored_fall":
                            stored_fall,

                        "historical_true_fall":
                            historical_fall,

                        "retained_falling_window_count":
                            int(
                                np.sum(
                                    labels
                                    == 1
                                )
                            ),

                        "risk_record_present":
                            has_risk,

                        "fall_start_frame":
                            int(
                                row[
                                    "fall_start_frame"
                                ]
                            ),

                        "fall_start_position":
                            int(
                                row[
                                    "fall_start_position"
                                ]
                            ),

                        "impact_position":
                            int(
                                row[
                                    "impact_position"
                                ]
                            ),
                    })

        stored_global.update(
            stored_subject
        )

        historical_global.update(
            historical_subject
        )

        truth_by_subject[
            str(
                subject
            )
        ] = {
            "fold":
                subject_folds[
                    subject
                ],

            "stored_label_trial_inventory": {
                "Activity":
                    int(
                        stored_subject[
                            "Activity"
                        ]
                    ),

                "Falling":
                    int(
                        stored_subject[
                            "Falling"
                        ]
                    ),

                "total":
                    int(
                        stored_subject[
                            "Activity"
                        ]
                        + stored_subject[
                            "Falling"
                        ]
                    ),
            },

            "historical_evaluator_trial_truth_inventory": {
                "Activity":
                    int(
                        historical_subject[
                            "Activity"
                        ]
                    ),

                "Falling":
                    int(
                        historical_subject[
                            "Falling"
                        ]
                    ),

                "total":
                    int(
                        historical_subject[
                            "Activity"
                        ]
                        + historical_subject[
                            "Falling"
                        ]
                    ),
            },
        }

    if (
        stored_global[
            "Activity"
        ],
        stored_global[
            "Falling"
        ],
    ) != (
        3391,
        2918,
    ):
        raise ValueError(
            "stored-label inventory differs from frozen audit"
        )

    if (
        historical_global[
            "Activity"
        ],
        historical_global[
            "Falling"
        ],
    ) != (
        3390,
        2919,
    ):
        raise ValueError(
            "historical truth inventory differs from frozen audit"
        )

    if len(
        mismatch_rows
    ) != 1:
        raise ValueError(
            f"expected one truth mismatch, found {len(mismatch_rows)}"
        )

    mismatch = mismatch_rows[
        0
    ]

    expected_mismatch = {
        "fold":
            4,

        "subject":
            106,

        "task":
            27,

        "trial":
            5,

        "event_id":
            "KFALL_106_T27_R05",

        "stored_fall":
            False,

        "historical_true_fall":
            True,

        "retained_falling_window_count":
            0,

        "risk_record_present":
            True,

        "fall_start_frame":
            12,

        "fall_start_position":
            11,

        "impact_position":
            52,
    }

    if mismatch != expected_mismatch:
        raise ValueError(
            "unexpected truth-mismatch record:\n"
            + json.dumps(
                mismatch,
                indent=2,
                sort_keys=True,
            )
        )

    risk_without_outer_trial = [
        key
        for key
        in risk_by_trial
        if (
            key[
                0
            ]
            in subject_folds
            and key
            not in outer_trial_keys
        )
    ]

    if risk_without_outer_trial:
        raise ValueError(
            "risk event exists without stored outer trial"
        )

    namespace = cfg[
        "sharding"
    ][
        "namespace"
    ]

    shards = []

    for old in v1_plan[
        "shards"
    ]:
        row = json.loads(
            json.dumps(
                old
            )
        )

        subject = int(
            row[
                "subject"
            ]
        )

        row[
            "supersedes_v1_shard_id"
        ] = row[
            "shard_id"
        ]

        row[
            "shard_id"
        ] = shard_id(
            namespace,
            fold=row[
                "fold"
            ],
            subject=subject,
            block=row[
                "block"
            ],
        )

        row[
            "historical_trial_truth_inventory"
        ] = truth_by_subject[
            str(
                subject
            )
        ][
            "historical_evaluator_trial_truth_inventory"
        ]

        row[
            "stored_label_trial_inventory"
        ] = truth_by_subject[
            str(
                subject
            )
        ][
            "stored_label_trial_inventory"
        ]

        row[
            "ground_truth_rule"
        ] = (
            "true_fall = (retained_falling_window_count > 0) "
            "OR risk_record_present"
        )

        row[
            "historical_exception_present"
        ] = (
            subject == 106
        )

        shards.append(
            row
        )

    if len(
        shards
    ) != 793:
        raise ValueError(
            "v2 shard count changed"
        )

    if len({
        row[
            "shard_id"
        ]
        for row
        in shards
    }) != 793:
        raise ValueError(
            "v2 shard IDs are not unique"
        )

    totals = {
        key:
            int(
                sum(
                    int(
                        row[
                            field
                        ]
                    )
                    for row
                    in shards
                )
            )
        for key, field
        in [
            (
                "unique_fault_instances",
                "expected_unique_fault_instances",
            ),
            (
                "model_window_evaluations",
                "expected_model_window_evaluations",
            ),
            (
                "subject_condition_rows",
                "expected_condition_rows",
            ),
        ]
    }

    expected_totals = {
        "unique_fault_instances":
            42766632,

        "model_window_evaluations":
            479750160,

        "subject_condition_rows":
            320616,
    }

    if totals != expected_totals:
        raise ValueError(
            f"v2 workload changed: {totals}"
        )

    plan = {
        "schema_version":
            "phase4h_sensor_fi_outer_execution_shards_v2_historical_truth",

        "status":
            "FROZEN_PRE_OUTER_EXECUTION",

        "partition":
            "outer_test",

        "namespace":
            namespace,

        "execution_config": {
            "path":
                str(
                    Path(
                        config_path
                    ).relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    config_path
                ),
        },

        "supersedes_v1_plan": {
            "path":
                str(
                    v1_plan_path.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    v1_plan_path
                ),

            "preserved_unchanged":
                True,
        },

        "single_scientific_clarification":
            (
                "Trial-level true-event denominator follows recovered "
                "historical evaluator risk-or-Falling-label rule."
            ),

        "shard_count":
            793,

        "subject_count":
            61,

        "execution_blocks_per_subject":
            13,

        "family_instance_counts_per_parent":
            v1_plan[
                "family_instance_counts_per_parent"
            ],

        "global_window_trial_inventory": {
            "subjects":
                61,

            "trials":
                6309,

            "windows":
                273830,

            "activity_windows":
                264024,

            "falling_windows":
                9806,
        },

        "global_stored_label_trial_inventory": {
            "Activity":
                3391,

            "Falling":
                2918,

            "total":
                6309,
        },

        "global_historical_evaluator_trial_truth_inventory": {
            "Activity":
                3390,

            "Falling":
                2919,

            "total":
                6309,
        },

        "historical_exception": {
            **mismatch,

            "valid_falling_trigger_possible_under_retained_labels":
                False,

            "classification_consequence":
                "FN",

            "event_detection_consequence":
                "missed",

            "sensor_lead_ms":
                None,
        },

        "truth_by_subject":
            truth_by_subject,

        "expected_totals_from_shards":
            totals,

        "shards":
            shards,

        "scientific_boundary": {
            "metadata_labels_annotations_only":
                True,

            "model_loaded":
                False,

            "model_predictions_computed":
                False,

            "fault_injection_executed":
                False,

            "outer_model_outcomes_seen":
                False,

            "outer_fault_outcomes_seen":
                False,

            "OnField_used":
                False,
        },
    }

    return plan


def main():
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

    plan = build(
        config_path=args.config,
    )

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
        "V2_PLAN_STATUS=",
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
        "STORED_LABEL_TRUTH=",
        plan[
            "global_stored_label_trial_inventory"
        ],
        sep="",
    )

    print(
        "HISTORICAL_EVALUATOR_TRUTH=",
        plan[
            "global_historical_evaluator_trial_truth_inventory"
        ],
        sep="",
    )

    print(
        "HISTORICAL_EXCEPTION=",
        plan[
            "historical_exception"
        ][
            "event_id"
        ],
        sep="",
    )

    print(
        "MODEL_LOADED=False"
    )

    print(
        "MODEL_FORWARD=False"
    )

    print(
        "FAULT_EXECUTION=False"
    )


if __name__ == "__main__":
    main()
