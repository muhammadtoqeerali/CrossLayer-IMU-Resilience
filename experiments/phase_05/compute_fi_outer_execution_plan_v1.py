"""Phase-5E deterministic outer compute-FI execution planner v1.

This planner is a derived execution artifact for the already-frozen Phase-5D
prospective outer compute-FI protocol.

It reads:
- frozen JSON protocol/split metadata;
- filesystem directory names;
- NumPy .npy HEADER BYTES for segments.npy only.

It does NOT:
- call np.load;
- create a memmap;
- open labels.npy;
- read segment payload bytes;
- materialize signal arrays;
- load a model;
- execute a model;
- inject a fault;
- read outer predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.lib import format as npfmt

from compute_fi_outer_sampling_v1 import (
    canonical_sampling_payload,
    derive_persistent_onset_index,
    sampling_instance_id,
)


SCHEMA = "phase5e_compute_fi_outer_execution_plan_v1"
STATUS = "QUALIFIED_FROZEN_DERIVED_EXECUTION_PLAN"

SHARD_NAMESPACE = (
    "crosslayer-phase5e-compute-fi-outer-shard-v1"
)

CLEAN_CACHE_NAMESPACE = (
    "crosslayer-phase5e-compute-fi-clean-cache-v1"
)

PERSISTENCE_ORDER = (
    "transient_one_inference",
    "persistent_from_onset_until_trial_end",
)


def canonical_json(
    value,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def sha256_file(
    path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_npy_header_only(
    path,
) -> dict:
    """Read only the NPY header and validate payload size arithmetically."""

    path = Path(
        path
    )

    with path.open(
        "rb"
    ) as f:
        version = npfmt.read_magic(
            f
        )

        if version == (
            1,
            0,
        ):
            shape, fortran_order, dtype = (
                npfmt.read_array_header_1_0(
                    f
                )
            )

        elif version == (
            2,
            0,
        ):
            shape, fortran_order, dtype = (
                npfmt.read_array_header_2_0(
                    f
                )
            )

        elif version == (
            3,
            0,
        ):
            reader = getattr(
                npfmt,
                "read_array_header_3_0",
                None,
            )

            if reader is None:
                shape, fortran_order, dtype = (
                    npfmt._read_array_header(
                        f,
                        version,
                    )
                )
            else:
                shape, fortran_order, dtype = (
                    reader(
                        f
                    )
                )

        else:
            raise ValueError(
                f"unsupported NPY version {version}: {path}"
            )

        payload_offset = int(
            f.tell()
        )

        file_size = int(
            os.fstat(
                f.fileno()
            ).st_size
        )

    dtype = np.dtype(
        dtype
    )

    element_count = 1

    for value in shape:
        element_count *= int(
            value
        )

    expected_payload_bytes = (
        element_count
        * int(
            dtype.itemsize
        )
    )

    actual_payload_bytes = (
        file_size
        - payload_offset
    )

    if actual_payload_bytes != expected_payload_bytes:
        raise ValueError(
            "NPY payload size mismatch without reading payload: "
            f"{path}"
        )

    return {
        "version":
            [
                int(x)
                for x in version
            ],

        "shape":
            [
                int(x)
                for x in shape
            ],

        "fortran_order":
            bool(
                fortran_order
            ),

        "dtype":
            dtype.str,

        "payload_offset":
            payload_offset,

        "file_size":
            file_size,

        "payload_bytes":
            int(
                expected_payload_bytes
            ),
    }


def load_subject_fold_binding(
    split,
) -> tuple[
    dict[int, int],
    dict[int, str],
]:
    subject_fold = {}
    subject_canonical = {}

    for row in split[
        "folds"
    ]:
        fold = int(
            row[
                "fold"
            ]
        )

        outer = row[
            "outer_test"
        ]

        storage = [
            int(x)
            for x in outer[
                "storage_ids"
            ]
        ]

        canonical = list(
            outer[
                "canonical_ids"
            ]
        )

        if len(
            storage
        ) != len(
            canonical
        ):
            raise ValueError(
                f"fold {fold} storage/canonical length mismatch"
            )

        for subject, canonical_id in zip(
            storage,
            canonical,
        ):
            if subject in subject_fold:
                raise ValueError(
                    f"subject appears in multiple folds: {subject}"
                )

            subject_fold[
                subject
            ] = fold

            subject_canonical[
                subject
            ] = canonical_id

    if (
        len(
            subject_fold
        )
        != 61
    ):
        raise ValueError(
            "expected exactly 61 outer subjects"
        )

    return (
        subject_fold,
        subject_canonical,
    )


def build_trial_inventory(
    *,
    split,
    dataset_root,
):
    (
        subject_fold,
        subject_canonical,
    ) = load_subject_fold_binding(
        split
    )

    dataset_root = Path(
        dataset_root
    )

    rows = []
    subject_summary = {}

    header_versions = Counter()
    dtype_counts = Counter()

    for subject in sorted(
        subject_fold
    ):
        subject_dir = (
            dataset_root
            / str(
                subject
            )
        )

        if not subject_dir.is_dir():
            raise ValueError(
                f"missing subject directory: {subject_dir}"
            )

        subject_trials = 0
        subject_windows = 0

        for task_dir in sorted(
            (
                path
                for path in subject_dir.iterdir()
                if path.is_dir()
            ),
            key=lambda path: int(
                path.name
            ),
        ):
            task = int(
                task_dir.name
            )

            for trial_dir in sorted(
                (
                    path
                    for path in task_dir.iterdir()
                    if path.is_dir()
                ),
                key=lambda path: int(
                    path.name
                ),
            ):
                trial = int(
                    trial_dir.name
                )

                segments_path = (
                    trial_dir
                    / "segments.npy"
                )

                labels_path = (
                    trial_dir
                    / "labels.npy"
                )

                # labels.npy existence defines retained stored trials.
                # The label file is never opened by this planner.
                if not (
                    segments_path.is_file()
                    and labels_path.is_file()
                ):
                    continue

                info = read_npy_header_only(
                    segments_path
                )

                shape = tuple(
                    info[
                        "shape"
                    ]
                )

                if (
                    len(
                        shape
                    )
                    != 3
                    or shape[
                        1:
                    ]
                    != (
                        30,
                        9,
                    )
                    or int(
                        shape[
                            0
                        ]
                    )
                    <= 0
                ):
                    raise ValueError(
                        "unexpected stored-window shape: "
                        f"path={segments_path} shape={shape}"
                    )

                windows = int(
                    shape[
                        0
                    ]
                )

                rows.append({
                    "fold":
                        int(
                            subject_fold[
                                subject
                            ]
                        ),

                    "canonical_subject":
                        subject_canonical[
                            subject
                        ],

                    "subject":
                        int(
                            subject
                        ),

                    "task":
                        int(
                            task
                        ),

                    "trial":
                        int(
                            trial
                        ),

                    "window_count":
                        windows,
                })

                subject_trials += 1
                subject_windows += windows

                header_versions[
                    str(
                        tuple(
                            info[
                                "version"
                            ]
                        )
                    )
                ] += 1

                dtype_counts[
                    info[
                        "dtype"
                    ]
                ] += 1

        subject_summary[
            int(
                subject
            )
        ] = {
            "fold":
                int(
                    subject_fold[
                        subject
                    ]
                ),

            "canonical_subject":
                subject_canonical[
                    subject
                ],

            "trial_count":
                int(
                    subject_trials
                ),

            "window_count":
                int(
                    subject_windows
                ),
        }

    rows.sort(
        key=lambda row: (
            int(
                row[
                    "fold"
                ]
            ),
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
    )

    return (
        rows,
        subject_summary,
        dict(
            sorted(
                header_versions.items()
            )
        ),
        dict(
            sorted(
                dtype_counts.items()
            )
        ),
    )


def rows_digest(
    rows,
) -> str:
    payload = (
        "\n".join(
            canonical_json(
                row
            )
            for row in rows
        )
        + "\n"
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def subject_rows(
    subject_summary,
):
    return [
        {
            "subject":
                int(
                    subject
                ),

            **subject_summary[
                subject
            ],
        }
        for subject in sorted(
            subject_summary
        )
    ]


def target_rows_for_variant(
    protocol,
    model_variant,
):
    return [
        row
        for row in protocol[
            "target_inventory"
        ]
        if model_variant in row[
            "model_variants"
        ]
    ]


def shard_id(
    *,
    fold,
    subject,
    model_variant,
    checkpoint_seed,
    persistence,
) -> str:
    payload = {
        "namespace":
            SHARD_NAMESPACE,

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "model_variant":
            model_variant,

        "checkpoint_seed":
            int(
                checkpoint_seed
            ),

        "persistence":
            persistence,
    }

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]

    mode = (
        "transient"
        if persistence
        == "transient_one_inference"
        else "persistent"
    )

    return (
        f"p5e-o1-"
        f"f{int(fold)}-"
        f"s{int(subject):03d}-"
        f"{model_variant}-"
        f"seed{int(checkpoint_seed)}-"
        f"{mode}-"
        f"{digest}"
    )


def clean_cache_id(
    *,
    fold,
    subject,
    model_variant,
    checkpoint_seed,
) -> str:
    payload = {
        "namespace":
            CLEAN_CACHE_NAMESPACE,

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "model_variant":
            model_variant,

        "checkpoint_seed":
            int(
                checkpoint_seed
            ),
    }

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]

    return (
        f"p5e-c0-"
        f"f{int(fold)}-"
        f"s{int(subject):03d}-"
        f"{model_variant}-"
        f"seed{int(checkpoint_seed)}-"
        f"{digest}"
    )


def persistent_target_exposure(
    *,
    protocol_target,
    fold,
    subject,
    trials,
):
    exposure = 0
    instances = 0

    onset_digest = hashlib.sha256()

    for trial_row in trials:
        payload = canonical_sampling_payload(
            partition="outer_test",
            fold=fold,
            subject=subject,
            task=int(
                trial_row[
                    "task"
                ]
            ),
            trial=int(
                trial_row[
                    "trial"
                ]
            ),
            parent_kind="trial",
            window_index=None,
            representation_class=protocol_target[
                "representation_class"
            ],
            target_name=protocol_target[
                "target_name"
            ],
            target_role=protocol_target[
                "target_role"
            ],
            persistence="persistent_from_onset_until_trial_end",
            replicate_index=0,
        )

        trial_windows = int(
            trial_row[
                "window_count"
            ]
        )

        onset = derive_persistent_onset_index(
            payload,
            trial_window_count=trial_windows,
        )

        sid = sampling_instance_id(
            payload
        )

        onset_digest.update(
            (
                f"{sid}|"
                f"{trial_windows}|"
                f"{onset}\n"
            ).encode(
                "utf-8"
            )
        )

        exposure += (
            trial_windows
            - onset
        )

        instances += 1

    return {
        "outer_instance_count_per_seed":
            int(
                instances
            ),

        "faulted_model_window_evaluations_per_seed":
            int(
                exposure
            ),

        "onset_binding_sha256":
            onset_digest.hexdigest(),
    }


def build_plan(
    *,
    protocol_path,
    split_path,
    dataset_root,
):
    protocol_path = Path(
        protocol_path
    )

    split_path = Path(
        split_path
    )

    protocol = json.loads(
        protocol_path.read_text()
    )

    split = json.loads(
        split_path.read_text()
    )

    if (
        protocol[
            "status"
        ]
        != "FROZEN_PROSPECTIVE_OUTER_COMPUTE_FI_PROTOCOL"
    ):
        raise ValueError(
            "Phase-5D protocol is not frozen"
        )

    (
        trials,
        subject_summary,
        header_versions,
        dtype_counts,
    ) = build_trial_inventory(
        split=split,
        dataset_root=dataset_root,
    )

    if len(
        trials
    ) != 6309:
        raise ValueError(
            f"trial inventory changed: {len(trials)}"
        )

    total_windows = sum(
        int(
            row[
                "window_count"
            ]
        )
        for row in trials
    )

    if total_windows != 273830:
        raise ValueError(
            f"window inventory changed: {total_windows}"
        )

    inventory_sha = rows_digest(
        trials
    )

    subjects_as_rows = subject_rows(
        subject_summary
    )

    subject_sha = rows_digest(
        subjects_as_rows
    )

    trials_by_subject = {
        subject:
            []
        for subject in subject_summary
    }

    for row in trials:
        trials_by_subject[
            int(
                row[
                    "subject"
                ]
            )
        ].append(
            row
        )

    model_variants = list(
        protocol[
            "model_estate"
        ][
            "model_variants"
        ]
    )

    checkpoint_seeds = [
        int(
            seed
        )
        for seed in protocol[
            "model_estate"
        ][
            "checkpoint_seeds"
        ]
    ]

    if model_variants != [
        "fp32",
        "ptq_v7",
    ]:
        raise ValueError(
            "model-variant order changed"
        )

    if checkpoint_seeds != [
        42,
        123,
        2025,
    ]:
        raise ValueError(
            "checkpoint seed order changed"
        )

    clean_caches = []

    shards = []

    aggregate_persistent_onset = hashlib.sha256()

    persistent_target_cache = {}

    for subject in sorted(
        subject_summary
    ):
        summary = subject_summary[
            subject
        ]

        fold = int(
            summary[
                "fold"
            ]
        )

        subject_trials = trials_by_subject[
            subject
        ]

        if len(
            subject_trials
        ) != int(
            summary[
                "trial_count"
            ]
        ):
            raise ValueError(
                f"subject trial-count mismatch: {subject}"
            )

        if (
            sum(
                int(
                    row[
                        "window_count"
                    ]
                )
                for row in subject_trials
            )
            != int(
                summary[
                    "window_count"
                ]
            )
        ):
            raise ValueError(
                f"subject window-count mismatch: {subject}"
            )

        # Compute persistent onset/exposure once per model-independent target.
        for target in protocol[
            "target_inventory"
        ]:
            key = (
                int(
                    subject
                ),
                target[
                    "representation_class"
                ],
                target[
                    "target_name"
                ],
                target[
                    "target_role"
                ],
            )

            result = persistent_target_exposure(
                protocol_target=target,
                fold=fold,
                subject=subject,
                trials=subject_trials,
            )

            persistent_target_cache[
                key
            ] = result

            # Reproduce the previously qualified global persistent-onset
            # binding in exact trial/target lexical order later below.

        for model_variant in model_variants:
            targets = target_rows_for_variant(
                protocol,
                model_variant,
            )

            expected_target_count = (
                10
                if model_variant
                == "fp32"
                else 14
            )

            if len(
                targets
            ) != expected_target_count:
                raise ValueError(
                    f"target count changed variant={model_variant}"
                )

            for seed in checkpoint_seeds:
                clean_caches.append({
                    "clean_cache_id":
                        clean_cache_id(
                            fold=fold,
                            subject=subject,
                            model_variant=model_variant,
                            checkpoint_seed=seed,
                        ),

                    "fold":
                        fold,

                    "subject":
                        int(
                            subject
                        ),

                    "canonical_subject":
                        summary[
                            "canonical_subject"
                        ],

                    "model_variant":
                        model_variant,

                    "checkpoint_seed":
                        seed,

                    "window_count":
                        int(
                            summary[
                                "window_count"
                            ]
                        ),

                    "expected_clean_model_window_evaluations":
                        int(
                            summary[
                                "window_count"
                            ]
                        ),

                    "shared_by_persistence_shards":
                        True,
                })

                for persistence in PERSISTENCE_ORDER:
                    per_target = []

                    if (
                        persistence
                        == "transient_one_inference"
                    ):
                        for target in targets:
                            per_target.append({
                                "fault_family":
                                    target[
                                        "fault_family"
                                    ],

                                "representation_class":
                                    target[
                                        "representation_class"
                                    ],

                                "target_name":
                                    target[
                                        "target_name"
                                    ],

                                "target_role":
                                    target[
                                        "target_role"
                                    ],

                                "outer_instance_count":
                                    int(
                                        summary[
                                            "window_count"
                                        ]
                                    ),

                                "faulted_model_window_evaluations":
                                    int(
                                        summary[
                                            "window_count"
                                        ]
                                    ),
                            })

                    else:
                        for target in targets:
                            key = (
                                int(
                                    subject
                                ),
                                target[
                                    "representation_class"
                                ],
                                target[
                                    "target_name"
                                ],
                                target[
                                    "target_role"
                                ],
                            )

                            derived = persistent_target_cache[
                                key
                            ]

                            per_target.append({
                                "fault_family":
                                    target[
                                        "fault_family"
                                    ],

                                "representation_class":
                                    target[
                                        "representation_class"
                                    ],

                                "target_name":
                                    target[
                                        "target_name"
                                    ],

                                "target_role":
                                    target[
                                        "target_role"
                                    ],

                                "outer_instance_count":
                                    int(
                                        derived[
                                            "outer_instance_count_per_seed"
                                        ]
                                    ),

                                "faulted_model_window_evaluations":
                                    int(
                                        derived[
                                            "faulted_model_window_evaluations_per_seed"
                                        ]
                                    ),

                                "subject_target_onset_binding_sha256":
                                    derived[
                                        "onset_binding_sha256"
                                    ],
                            })

                    instance_count = sum(
                        int(
                            row[
                                "outer_instance_count"
                            ]
                        )
                        for row in per_target
                    )

                    faulted_evaluations = sum(
                        int(
                            row[
                                "faulted_model_window_evaluations"
                            ]
                        )
                        for row in per_target
                    )

                    shards.append({
                        "shard_id":
                            shard_id(
                                fold=fold,
                                subject=subject,
                                model_variant=model_variant,
                                checkpoint_seed=seed,
                                persistence=persistence,
                            ),

                        "fold":
                            fold,

                        "subject":
                            int(
                                subject
                            ),

                        "canonical_subject":
                            summary[
                                "canonical_subject"
                            ],

                        "model_variant":
                            model_variant,

                        "checkpoint_seed":
                            seed,

                        "persistence":
                            persistence,

                        "clean_cache_id":
                            clean_cache_id(
                                fold=fold,
                                subject=subject,
                                model_variant=model_variant,
                                checkpoint_seed=seed,
                            ),

                        "subject_inventory": {
                            "trial_count":
                                int(
                                    summary[
                                        "trial_count"
                                    ]
                                ),

                            "window_count":
                                int(
                                    summary[
                                        "window_count"
                                    ]
                                ),
                        },

                        "target_count":
                            len(
                                targets
                            ),

                        "target_names":
                            [
                                target[
                                    "target_name"
                                ]
                                for target in targets
                            ],

                        "expected_outer_instance_ids":
                            int(
                                instance_count
                            ),

                        "expected_faulted_model_window_evaluations":
                            int(
                                faulted_evaluations
                            ),

                        "target_exposure":
                            per_target,

                        "primary_execution_key":
                            "outer_instance_id",

                        "sampling_coordinates_are_frozen":
                            True,
                    })

    # Reproduce the global onset digest in the exact Phase-5E header
    # qualification order: subject -> task/trial -> sorted unique target tuple.
    persistent_target_tuples = sorted({
        (
            row[
                "representation_class"
            ],
            row[
                "target_name"
            ],
            row[
                "target_role"
            ],
        )
        for row in protocol[
            "target_inventory"
        ]
    })

    for subject in sorted(
        subject_summary
    ):
        fold = int(
            subject_summary[
                subject
            ][
                "fold"
            ]
        )

        for trial_row in trials_by_subject[
            subject
        ]:
            for (
                representation_class,
                target_name,
                target_role,
            ) in persistent_target_tuples:
                payload = canonical_sampling_payload(
                    partition="outer_test",
                    fold=fold,
                    subject=subject,
                    task=int(
                        trial_row[
                            "task"
                        ]
                    ),
                    trial=int(
                        trial_row[
                            "trial"
                        ]
                    ),
                    parent_kind="trial",
                    window_index=None,
                    representation_class=representation_class,
                    target_name=target_name,
                    target_role=target_role,
                    persistence="persistent_from_onset_until_trial_end",
                    replicate_index=0,
                )

                trial_windows = int(
                    trial_row[
                        "window_count"
                    ]
                )

                onset = derive_persistent_onset_index(
                    payload,
                    trial_window_count=trial_windows,
                )

                sid = sampling_instance_id(
                    payload
                )

                aggregate_persistent_onset.update(
                    (
                        f"{sid}|"
                        f"{trial_windows}|"
                        f"{onset}\n"
                    ).encode(
                        "utf-8"
                    )
                )

    shards.sort(
        key=lambda row: (
            int(
                row[
                    "subject"
                ]
            ),
            model_variants.index(
                row[
                    "model_variant"
                ]
            ),
            checkpoint_seeds.index(
                int(
                    row[
                        "checkpoint_seed"
                    ]
                )
            ),
            PERSISTENCE_ORDER.index(
                row[
                    "persistence"
                ]
            ),
        )
    )

    clean_caches.sort(
        key=lambda row: (
            int(
                row[
                    "subject"
                ]
            ),
            model_variants.index(
                row[
                    "model_variant"
                ]
            ),
            checkpoint_seeds.index(
                int(
                    row[
                        "checkpoint_seed"
                    ]
                )
            ),
        )
    )

    if len(
        shards
    ) != 732:
        raise ValueError(
            f"expected 732 shards, got {len(shards)}"
        )

    if len(
        clean_caches
    ) != 366:
        raise ValueError(
            f"expected 366 clean caches, got {len(clean_caches)}"
        )

    shard_ids = [
        row[
            "shard_id"
        ]
        for row in shards
    ]

    if len(
        shard_ids
    ) != len(
        set(
            shard_ids
        )
    ):
        raise ValueError(
            "shard-id collision"
        )

    clean_ids = [
        row[
            "clean_cache_id"
        ]
        for row in clean_caches
    ]

    if len(
        clean_ids
    ) != len(
        set(
            clean_ids
        )
    ):
        raise ValueError(
            "clean-cache-id collision"
        )

    total_outer_instances = sum(
        int(
            row[
                "expected_outer_instance_ids"
            ]
        )
        for row in shards
    )

    transient_instances = sum(
        int(
            row[
                "expected_outer_instance_ids"
            ]
        )
        for row in shards
        if row[
            "persistence"
        ]
        == "transient_one_inference"
    )

    persistent_instances = sum(
        int(
            row[
                "expected_outer_instance_ids"
            ]
        )
        for row in shards
        if row[
            "persistence"
        ]
        == "persistent_from_onset_until_trial_end"
    )

    transient_faulted_evaluations = sum(
        int(
            row[
                "expected_faulted_model_window_evaluations"
            ]
        )
        for row in shards
        if row[
            "persistence"
        ]
        == "transient_one_inference"
    )

    persistent_faulted_evaluations = sum(
        int(
            row[
                "expected_faulted_model_window_evaluations"
            ]
        )
        for row in shards
        if row[
            "persistence"
        ]
        == "persistent_from_onset_until_trial_end"
    )

    clean_evaluations = sum(
        int(
            row[
                "expected_clean_model_window_evaluations"
            ]
        )
        for row in clean_caches
    )

    if transient_instances != 19715760:
        raise ValueError(
            transient_instances
        )

    if persistent_instances != 454248:
        raise ValueError(
            persistent_instances
        )

    if total_outer_instances != 20170008:
        raise ValueError(
            total_outer_instances
        )

    if transient_faulted_evaluations != 19715760:
        raise ValueError(
            transient_faulted_evaluations
        )

    if clean_evaluations != 1642980:
        raise ValueError(
            clean_evaluations
        )

    fold_summary = {}

    for fold in range(
        1,
        6,
    ):
        fold_trials = [
            row
            for row in trials
            if int(
                row[
                    "fold"
                ]
            )
            == fold
        ]

        fold_subjects = {
            int(
                row[
                    "subject"
                ]
            )
            for row in fold_trials
        }

        fold_summary[
            str(
                fold
            )
        ] = {
            "subject_count":
                len(
                    fold_subjects
                ),

            "trial_count":
                len(
                    fold_trials
                ),

            "window_count":
                sum(
                    int(
                        row[
                            "window_count"
                        ]
                    )
                    for row in fold_trials
                ),
        }

    return {
        "schema_version":
            SCHEMA,

        "status":
            STATUS,

        "phase":
            "5E",

        "generated_date":
            "2026-10-05",

        "partition":
            "outer_test",

        "evidence_tier":
            "P0",

        "derivation_boundary": {
            "segments_npy_header_bytes_read":
                True,

            "segments_npy_payload_bytes_read":
                False,

            "labels_npy_opened":
                False,

            "np_load_used":
                False,

            "memmap_used":
                False,

            "signal_array_materialized":
                False,

            "model_loaded":
                False,

            "model_forward_executed":
                False,

            "fault_injection_executed":
                False,

            "outer_prediction_read":
                False,

            "onfield_used":
                False,
        },

        "frozen_dependencies": {
            "phase5d_protocol": {
                "path":
                    str(
                        protocol_path
                    ),

                "sha256":
                    sha256_file(
                        protocol_path
                    ),
            },

            "primary_split": {
                "path":
                    str(
                        split_path
                    ),

                "sha256":
                    sha256_file(
                        split_path
                    ),
            },

            "phase5d_sampler": {
                "path":
                    protocol[
                        "source_hashes"
                    ][
                        "sampling_implementation"
                    ][
                        "path"
                    ],

                "sha256":
                    protocol[
                        "source_hashes"
                    ][
                        "sampling_implementation"
                    ][
                        "sha256"
                    ],
            },
        },

        "inventory": {
            "subject_count":
                len(
                    subject_summary
                ),

            "trial_count":
                len(
                    trials
                ),

            "window_count":
                total_windows,

            "header_version_counts":
                header_versions,

            "segment_dtype_counts":
                dtype_counts,

            "trial_inventory_sha256":
                inventory_sha,

            "subject_inventory_sha256":
                subject_sha,

            "persistent_onset_binding_sha256":
                aggregate_persistent_onset.hexdigest(),

            "fold_summary":
                fold_summary,
        },

        "sharding": {
            "namespace":
                SHARD_NAMESPACE,

            "shard_unit":
                "subject_x_model_variant_x_checkpoint_seed_x_persistence",

            "shard_count":
                len(
                    shards
                ),

            "ordering":
                "subject_numeric_then_model_variant_fp32_ptq_v7_then_seed_42_123_2025_then_transient_persistent",

            "primary_execution_key":
                "outer_instance_id",

            "resume_key":
                "shard_id",

            "atomic_success_marker_required":
                True,

            "partial_output_reuse_allowed_without_valid_success_marker":
                False,
        },

        "clean_cache": {
            "namespace":
                CLEAN_CACHE_NAMESPACE,

            "cache_unit":
                "subject_x_model_variant_x_checkpoint_seed",

            "cache_count":
                len(
                    clean_caches
                ),

            "shared_between_transient_and_persistent_shards":
                True,

            "total_clean_model_window_evaluations":
                clean_evaluations,
        },

        "expected_totals": {
            "unique_sampling_ids":
                3921946,

            "transient_outer_instance_ids":
                transient_instances,

            "persistent_outer_instance_ids":
                persistent_instances,

            "total_outer_instance_ids":
                total_outer_instances,

            "clean_model_window_evaluations":
                clean_evaluations,

            "transient_faulted_model_window_evaluations":
                transient_faulted_evaluations,

            "persistent_faulted_model_window_evaluations":
                persistent_faulted_evaluations,

            "total_faulted_model_window_evaluations":
                (
                    transient_faulted_evaluations
                    + persistent_faulted_evaluations
                ),

            "total_model_window_evaluations_including_clean":
                (
                    clean_evaluations
                    + transient_faulted_evaluations
                    + persistent_faulted_evaluations
                ),
        },

        "execution_contract": {
            "all_732_fault_shards_required":
                True,

            "all_366_clean_caches_required":
                True,

            "clean_cache_must_be_verified_before_fault_shard":
                True,

            "common_fp32_sampling_coordinates_reused_across_variants":
                True,

            "sampling_coordinates_reused_across_checkpoint_seeds":
                True,

            "outer_instance_id_required_for_every_fault_execution_record":
                True,

            "phase5a_fault_id_retained_as_mutation_provenance":
                True,

            "faulted_output_nonfinite_must_be_recorded":
                True,

            "faulted_output_nonfinite_must_not_be_silently_coerced":
                True,

            "raw_outer_probabilities_public_persistence_required":
                False,

            "sampling_change_allowed":
                False,

            "target_change_allowed":
                False,

            "bit_resampling_allowed":
                False,

            "persistent_onset_resampling_allowed":
                False,

            "threshold_retuning_allowed":
                False,
        },

        "trial_inventory":
            trials,

        "subject_inventory":
            subjects_as_rows,

        "clean_caches":
            clean_caches,

        "shards":
            shards,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        required=True,
    )

    parser.add_argument(
        "--split",
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    plan = build_plan(
        protocol_path=args.protocol,
        split_path=args.split,
        dataset_root=args.dataset_root,
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE5E_PLAN_STATUS=",
        plan[
            "status"
        ],
        sep="",
    )

    print(
        "SUBJECT_COUNT=",
        plan[
            "inventory"
        ][
            "subject_count"
        ],
        sep="",
    )

    print(
        "TRIAL_COUNT=",
        plan[
            "inventory"
        ][
            "trial_count"
        ],
        sep="",
    )

    print(
        "WINDOW_COUNT=",
        plan[
            "inventory"
        ][
            "window_count"
        ],
        sep="",
    )

    print(
        "TRIAL_INVENTORY_SHA256=",
        plan[
            "inventory"
        ][
            "trial_inventory_sha256"
        ],
        sep="",
    )

    print(
        "SUBJECT_INVENTORY_SHA256=",
        plan[
            "inventory"
        ][
            "subject_inventory_sha256"
        ],
        sep="",
    )

    print(
        "PERSISTENT_ONSET_BINDING_SHA256=",
        plan[
            "inventory"
        ][
            "persistent_onset_binding_sha256"
        ],
        sep="",
    )

    print(
        "CLEAN_CACHE_COUNT=",
        plan[
            "clean_cache"
        ][
            "cache_count"
        ],
        sep="",
    )

    print(
        "SHARD_COUNT=",
        plan[
            "sharding"
        ][
            "shard_count"
        ],
        sep="",
    )

    for key, value in plan[
        "expected_totals"
    ].items():
        print(
            f"{key.upper()}={value}"
        )


if __name__ == "__main__":
    main()
