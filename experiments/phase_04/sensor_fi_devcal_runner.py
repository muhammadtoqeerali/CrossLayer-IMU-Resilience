"""Phase-4H frozen development/calibration FI execution runner.

Qualification mode is restricted to training_calibration.

This runner deliberately does not compute or gate on classification
performance. Its qualification gates only execution correctness,
deterministic pairing, clean reconstruction, finite model interfaces,
and frozen protocol preservation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from models.CNN import CNN
from preprocessing.helper import low_pass_filter


ROOT = Path(__file__).resolve().parents[2]

SAMPLING_IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling_v3.py"
)

OPERATOR_IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_operators.py"
)

WINDOW_FAMILIES = (
    "bias",
    "scale_factor",
    "noise",
    "clipping_saturation",
    "axis_loss",
)

SEQUENCE_FAMILIES = (
    "drift",
    "stuck_channel",
    "dropout",
    "frame_loss",
    "jitter",
    "delay",
    "orientation",
)

ALL_FAMILIES = (
    WINDOW_FAMILIES
    + SEQUENCE_FAMILIES
)


def _load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


SAMPLING = _load_module(
    "_phase4h_sampling_v3",
    SAMPLING_IMPL,
)

OPERATORS = _load_module(
    "_phase4h_sensor_fi_operators",
    OPERATOR_IMPL,
)


def sha256_file(
    path: str | Path,
) -> str:
    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def assert_partition(
    partition: str,
    *,
    execution_stage: str,
) -> None:
    partition = str(
        partition
    )

    execution_stage = str(
        execution_stage
    )

    if partition == "onfield":
        raise ValueError(
            "OnField fault execution is prohibited"
        )

    if execution_stage == "qualification":
        if partition != "training_calibration":
            raise ValueError(
                "runner qualification permits training_calibration only"
            )

        return

    if execution_stage == "final_outer_evaluation":
        if partition != "outer_test":
            raise ValueError(
                "final outer evaluation permits outer_test only"
            )

        return

    raise ValueError(
        f"unknown execution_stage: {execution_stage}"
    )


def normalise_binary_label(
    value,
) -> int:
    if isinstance(
        value,
        bytes,
    ):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    return int(
        "FALL"
        in str(
            value
        ).strip().upper()
    )


def stored_trial_paths(
    dataset_root: str | Path,
    *,
    subject: int,
    task: int,
    trial: int,
) -> tuple[Path, Path]:
    root = Path(
        dataset_root
    )

    trial_dir = (
        root
        / str(
            int(
                subject
            )
        )
        / str(
            int(
                task
            )
        )
        / str(
            int(
                trial
            )
        )
    )

    return (
        trial_dir
        / "segments.npy",

        trial_dir
        / "labels.npy",
    )


def load_stored_window(
    dataset_root: str | Path,
    identity: dict,
) -> tuple[np.ndarray, int]:
    segment_file, label_file = stored_trial_paths(
        dataset_root,
        subject=int(
            identity["subject"]
        ),
        task=int(
            identity["task"]
        ),
        trial=int(
            identity["trial"]
        ),
    )

    segments = np.load(
        segment_file,
        allow_pickle=False,
    )

    labels = np.load(
        label_file,
        allow_pickle=True,
    )

    index = int(
        identity["window_index"]
    )

    if tuple(
        segments.shape[1:]
    ) != (
        30,
        9,
    ):
        raise ValueError(
            f"stored segment shape mismatch: {segments.shape}"
        )

    if not (
        0
        <= index
        < len(
            segments
        )
    ):
        raise ValueError(
            "window_index out of bounds"
        )

    return (
        np.asarray(
            segments[
                index
            ],
            dtype=np.float64,
        ).copy(),

        normalise_binary_label(
            labels[
                index
            ]
        ),
    )


def source_trial_path(
    *,
    subject: int,
    task: int,
    trial: int,
    univr_root: str | Path,
    kfall_root: str | Path,
) -> Path:
    subject = int(
        subject
    )

    task = int(
        task
    )

    trial = int(
        trial
    )

    if subject >= 100:
        source_subject = (
            subject
            - 100
        )

        root = Path(
            kfall_root
        )

    else:
        source_subject = subject

        root = Path(
            univr_root
        )

    return (
        root
        / f"SA{source_subject:02d}"
        / (
            f"S{source_subject:02d}"
            f"T{task:02d}"
            f"R{trial:02d}.csv"
        )
    )


def load_source_trial(
    path: str | Path,
) -> np.ndarray:
    path = Path(
        path
    )

    frame = pd.read_csv(
        path
    )

    channels = [
        "AccX",
        "AccY",
        "AccZ",
        "GyrX",
        "GyrY",
        "GyrZ",
        "EulerX",
        "EulerY",
        "EulerZ",
    ]

    missing = [
        c
        for c in channels
        if c not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"source trial missing channels: {missing}"
        )

    x = frame[
        channels
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if (
        x.ndim != 2
        or x.shape[1] != 9
        or len(
            x
        ) < 30
    ):
        raise ValueError(
            f"invalid source trial shape: {x.shape}"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            "source trial contains non-finite values"
        )

    return x


def load_trial_segments_and_labels(
    dataset_root: str | Path,
    *,
    subject: int,
    task: int,
    trial: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_file, label_file = stored_trial_paths(
        dataset_root,
        subject=subject,
        task=task,
        trial=trial,
    )

    segments = np.load(
        segment_file,
        allow_pickle=False,
    )

    labels_raw = np.load(
        label_file,
        allow_pickle=True,
    )

    labels = np.asarray(
        [
            normalise_binary_label(
                value
            )
            for value
            in labels_raw
        ],
        dtype=np.int8,
    )

    if tuple(
        segments.shape[1:]
    ) != (
        30,
        9,
    ):
        raise ValueError(
            f"stored trial shape mismatch: {segments.shape}"
        )

    if len(
        segments
    ) != len(
        labels
    ):
        raise ValueError(
            "stored segments and labels have different lengths"
        )

    return (
        np.asarray(
            segments,
            dtype=np.float64,
        ),
        labels,
    )


def rewindow_sequence(
    sequence: np.ndarray,
    stored_labels: np.ndarray,
    *,
    fall_start_position: int | None,
) -> np.ndarray:
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

    starts: list[int] = []

    starts.extend(
        15 * i
        for i in range(
            activity_count
        )
    )

    if falling_count:
        if fall_start_position is None:
            raise ValueError(
                "fall_start_position required when Falling windows exist"
            )

        fall_start = int(
            fall_start_position
        )

        if fall_start < 0:
            raise ValueError(
                "fall_start_position must be non-negative"
            )

        starts.extend(
            fall_start
            + 15 * i
            for i in range(
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
                "historical rewindowing exceeds sequence bounds"
            )

        window = x[
            start:stop
        ]

        window = low_pass_filter(
            window,
            5,
            100,
        )

        windows.append(
            window
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


def load_reference_scales(
    severity_audit: str | Path,
    fold: int,
) -> dict[str, np.ndarray]:
    d = json.loads(
        Path(
            severity_audit
        ).read_text()
    )

    records = d[
        "fold_results"
    ]

    if isinstance(
        records,
        dict,
    ):
        records = list(
            records.values()
        )

    matches = [
        record
        for record
        in records
        if int(
            record["fold"]
        )
        == int(
            fold
        )
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"expected exactly one reference-scale record for fold {fold}"
        )

    raw = matches[
        0
    ][
        "filtered_window_raw_unit_stats"
    ]

    q99 = np.asarray(
        raw[
            "q99_abs"
        ],
        dtype=np.float64,
    )

    sigma = np.asarray(
        raw[
            "robust_sigma_mad_1p4826"
        ],
        dtype=np.float64,
    )

    if (
        q99.shape != (6,)
        or sigma.shape != (6,)
        or not np.all(
            q99 > 0
        )
        or not np.all(
            sigma > 0
        )
    ):
        raise ValueError(
            "invalid fold-local reference scale vectors"
        )

    return {
        "q99_abs":
            q99,

        "robust_sigma":
            sigma,
    }


def load_fp32_model(
    checkpoint_path: str | Path,
) -> torch.nn.Module:
    checkpoint_path = Path(
        checkpoint_path
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    hparams = ckpt[
        "hyper_parameters"
    ]

    cfg = dict(
        hparams[
            "cfg"
        ]
    )

    model = CNN(
        n_features=int(
            hparams[
                "n_features"
            ]
        ),
        n_classes=int(
            hparams[
                "n_classes"
            ]
        ),
        config=cfg,
    )

    state = {}

    for key, value in ckpt[
        "state_dict"
    ].items():
        if not key.startswith(
            "model."
        ):
            raise ValueError(
                f"unexpected checkpoint state key: {key}"
            )

        state[
            key[
                len(
                    "model."
                ):
            ]
        ] = value

    result = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise ValueError(
            "strict FP32 state loading failed"
        )

    model.eval()

    return model


def load_ptq_model(
    torchscript_path: str | Path,
):
    model = torch.jit.load(
        str(
            torchscript_path
        ),
        map_location="cpu",
    )

    model.eval()

    return model


def logits_and_positive_probability(
    model,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        windows,
        dtype=np.float32,
    )

    if x.ndim == 2:
        x = x[
            None,
            ...,
        ]

    if (
        x.ndim != 3
        or tuple(
            x.shape[1:]
        ) != (
            30,
            9,
        )
    ):
        raise ValueError(
            f"model windows must have shape N x 30 x 9, got {x.shape}"
        )

    tensor = torch.from_numpy(
        x
    )

    with torch.no_grad():
        logits = model(
            tensor
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )[
            :,
            1
        ]

    logits_np = (
        logits.detach()
        .cpu()
        .numpy()
    )

    probability_np = (
        probability.detach()
        .cpu()
        .numpy()
    )

    if logits_np.shape != (
        len(
            x
        ),
        2,
    ):
        raise ValueError(
            f"unexpected logits shape: {logits_np.shape}"
        )

    if not np.isfinite(
        logits_np
    ).all():
        raise ValueError(
            "model logits are non-finite"
        )

    if not np.isfinite(
        probability_np
    ).all():
        raise ValueError(
            "model probabilities are non-finite"
        )

    if np.any(
        probability_np < 0.0
    ) or np.any(
        probability_np > 1.0
    ):
        raise ValueError(
            "model probabilities outside [0,1]"
        )

    return (
        logits_np,
        probability_np,
    )


def load_fp32_members(
    freeze_path: str | Path,
) -> dict[tuple[int, int], dict]:
    d = json.loads(
        Path(
            freeze_path
        ).read_text()
    )

    members = {}

    for row in d[
        "checkpoints"
    ]:
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
        )

        if key in members:
            raise ValueError(
                f"duplicate FP32 seed/fold: {key}"
            )

        members[
            key
        ] = row

    if len(
        members
    ) != 15:
        raise ValueError(
            "expected 15 FP32 model members"
        )

    return members


def load_ptq_members(
    all15_path: str | Path,
) -> dict[tuple[int, int], dict]:
    d = json.loads(
        Path(
            all15_path
        ).read_text()
    )

    members = {}

    for row in d[
        "members"
    ]:
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
        )

        if key in members:
            raise ValueError(
                f"duplicate PTQ seed/fold: {key}"
            )

        members[
            key
        ] = row

    if len(
        members
    ) != 15:
        raise ValueError(
            "expected 15 PTQ model members"
        )

    return members


def load_threshold_rows(
    path: str | Path,
) -> dict[tuple[int, int, str], dict]:
    path = Path(
        path
    )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(
            csv.DictReader(
                f
            )
        )

    if len(
        rows
    ) != 45:
        raise ValueError(
            "expected exactly 45 frozen threshold rows"
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

        if key in result:
            raise ValueError(
                f"duplicate threshold key: {key}"
            )

        result[
            key
        ] = row

    if len(
        result
    ) != 45:
        raise ValueError(
            "threshold key cardinality mismatch"
        )

    return result


def select_calibration_identity_by_fold(
    calibration_config: str | Path,
) -> dict[int, dict]:
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

        rows = sorted(
            fold_rec[
                "selection"
            ],
            key=lambda row: (
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
                int(
                    row[
                        "window_index"
                    ]
                ),
            ),
        )

        if not rows:
            raise ValueError(
                f"fold {fold} calibration selection is empty"
            )

        result[
            fold
        ] = rows[
            0
        ]

    if set(
        result
    ) != {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(
            "calibration fold coverage mismatch"
        )

    return result


def find_activity_only_sequence_parent(
    calibration_config: str | Path,
    dataset_root: str | Path,
    *,
    fold: int,
    minimum_windows: int = 30,
) -> tuple[int, int, int]:
    d = json.loads(
        Path(
            calibration_config
        ).read_text()
    )

    fold_rec = next(
        record
        for record
        in d[
            "folds"
        ]
        if int(
            record[
                "fold"
            ]
        )
        == int(
            fold
        )
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

    for subject, task, trial in keys:
        segments, labels = (
            load_trial_segments_and_labels(
                dataset_root,
                subject=subject,
                task=task,
                trial=trial,
            )
        )

        if (
            len(
                segments
            )
            >= int(
                minimum_windows
            )
            and np.all(
                labels == 0
            )
        ):
            return (
                subject,
                task,
                trial,
            )

    raise ValueError(
        f"no activity-only sequence parent with >= {minimum_windows} windows in fold {fold}"
    )


def representative_instance(
    rows: Iterable[dict],
    *,
    family: str,
    level: str,
) -> dict:
    matches = [
        row
        for row
        in rows
        if row[
            "family"
        ]
        == family
        and row[
            "severity"
        ][
            "level"
        ]
        == level
    ]

    if not matches:
        raise ValueError(
            f"no {family}/{level} instance"
        )

    return matches[
        0
    ]


def run_qualification(
    *,
    config_path: str | Path,
    output_path: str | Path,
) -> dict:
    config_path = Path(
        config_path
    )

    output_path = Path(
        output_path
    )

    cfg = json.loads(
        config_path.read_text()
    )

    assert_partition(
        cfg[
            "qualification_partition"
        ],
        execution_stage="qualification",
    )

    cross = ROOT

    protech = Path(
        cfg[
            "dataset"
        ][
            "stored_window_root"
        ]
    ).parents[
        3
    ]

    dataset_root = Path(
        cfg[
            "dataset"
        ][
            "stored_window_root"
        ]
    )

    calibration_path = (
        cross
        / cfg[
            "dataset"
        ][
            "calibration_identity"
        ]
    )

    severity_audit = Path(
        cfg[
            "reference_scales"
        ][
            "source"
        ]
    )

    fp32_freeze = (
        cross
        / cfg[
            "models"
        ][
            "fp32"
        ][
            "checkpoint_freeze"
        ]
    )

    ptq_all15 = Path(
        cfg[
            "models"
        ][
            "ptq_v7"
        ][
            "source_all15_manifest"
        ]
    )

    threshold_path = Path(
        cfg[
            "thresholds"
        ][
            "path"
        ]
    )

    severity_protocol = json.loads(
        (
            cross
            / cfg[
                "authoritative_dependencies"
            ][
                "severity_protocol"
            ][
                "path"
            ]
        ).read_text()
    )

    fp32_members = load_fp32_members(
        fp32_freeze
    )

    ptq_members = load_ptq_members(
        ptq_all15
    )

    if set(
        fp32_members
    ) != set(
        ptq_members
    ):
        raise ValueError(
            "FP32 and PTQ seed/fold membership differs"
        )

    thresholds = load_threshold_rows(
        threshold_path
    )

    # Thresholds are loaded only to verify frozen cardinality/schema.
    # They are never applied during qualification.
    threshold_rows_loaded = len(
        thresholds
    )

    identities = select_calibration_identity_by_fold(
        calibration_path
    )

    gates: dict[str, bool] = {}

    clean_model_checks = []

    loaded_fp32 = {}
    loaded_ptq = {}

    # --------------------------------------------------------
    # Clean interface check for every one of the 15 paired models.
    # --------------------------------------------------------

    for seed, fold in sorted(
        fp32_members
    ):
        identity = identities[
            fold
        ]

        window, _ = load_stored_window(
            dataset_root,
            identity,
        )

        fp32_rec = fp32_members[
            (
                seed,
                fold,
            )
        ]

        ptq_rec = ptq_members[
            (
                seed,
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
                f"FP32 hash mismatch seed={seed} fold={fold}"
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
                f"PTQ hash mismatch seed={seed} fold={fold}"
            )

        fp32_model = load_fp32_model(
            fp32_path
        )

        ptq_model = load_ptq_model(
            ptq_path
        )

        loaded_fp32[
            (
                seed,
                fold,
            )
        ] = fp32_model

        loaded_ptq[
            (
                seed,
                fold,
            )
        ] = ptq_model

        fp32_logits, fp32_prob = (
            logits_and_positive_probability(
                fp32_model,
                window,
            )
        )

        ptq_logits, ptq_prob = (
            logits_and_positive_probability(
                ptq_model,
                window,
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
                fp32_prob
            ).all()
            and np.isfinite(
                ptq_prob
            ).all()
        )

        gates[
            f"clean_interface_seed{seed}_fold{fold}"
        ] = passed

        clean_model_checks.append({
            "seed":
                seed,

            "fold":
                fold,

            "fp32_checkpoint":
                str(
                    fp32_path
                ),

            "ptq_torchscript":
                str(
                    ptq_path
                ),

            "identity": {
                "subject":
                    int(
                        identity[
                            "subject"
                        ]
                    ),

                "task":
                    int(
                        identity[
                            "task"
                        ]
                    ),

                "trial":
                    int(
                        identity[
                            "trial"
                        ]
                    ),

                "window_index":
                    int(
                        identity[
                            "window_index"
                        ]
                    ),
            },

            "output_shape_fp32":
                list(
                    fp32_logits.shape
                ),

            "output_shape_ptq":
                list(
                    ptq_logits.shape
                ),

            # Do not persist probabilities: runner qualification
            # must not invite performance or degradation tuning.
            "probabilities_persisted":
                False,
        })

    gates[
        "all_15_fp32_models_loaded"
    ] = (
        len(
            loaded_fp32
        )
        == 15
    )

    gates[
        "all_15_ptq_models_loaded"
    ] = (
        len(
            loaded_ptq
        )
        == 15
    )

    gates[
        "threshold_rows_schema_loaded_but_not_applied"
    ] = (
        threshold_rows_loaded
        == 45
    )

    # --------------------------------------------------------
    # Window-value fault paths: all five families, every fold.
    # Model seed 42 only. Same exact corrupted tensor feeds FP32/PTQ.
    # --------------------------------------------------------

    window_fault_checks = []

    for fold in range(
        1,
        6,
    ):
        identity = identities[
            fold
        ]

        clean_window, _ = load_stored_window(
            dataset_root,
            identity,
        )

        parent_id = (
            f"subject={int(identity['subject'])}"
            f"|task={int(identity['task'])}"
            f"|trial={int(identity['trial'])}"
            f"|window={int(identity['window_index'])}"
        )

        instances = (
            SAMPLING.generate_window_instances(
                fold=fold,
                partition="training_calibration",
                parent_sequence_id=parent_id,
                severity_protocol=severity_protocol,
            )
        )

        references = load_reference_scales(
            severity_audit,
            fold,
        )

        for family in WINDOW_FAMILIES:
            instance = representative_instance(
                instances,
                family=family,
                level="L1",
            )

            kwargs = {}

            if family in {
                "bias",
                "noise",
                "clipping_saturation",
            }:
                kwargs[
                    "reference_scales"
                ] = references

            original = clean_window.copy()

            corrupted, audit = OPERATORS.apply_fault(
                clean_window,
                instance,
                **kwargs,
            )

            input_unchanged = np.array_equal(
                clean_window,
                original,
            )

            corrupted_for_fp32 = corrupted.copy()
            corrupted_for_ptq = corrupted.copy()

            identical_pair_input = np.array_equal(
                corrupted_for_fp32,
                corrupted_for_ptq,
            )

            fp32_logits, _ = (
                logits_and_positive_probability(
                    loaded_fp32[
                        (
                            42,
                            fold,
                        )
                    ],
                    corrupted_for_fp32,
                )
            )

            ptq_logits, _ = (
                logits_and_positive_probability(
                    loaded_ptq[
                        (
                            42,
                            fold,
                        )
                    ],
                    corrupted_for_ptq,
                )
            )

            passed = bool(
                input_unchanged
                and identical_pair_input
                and audit[
                    "Euler_preserved"
                ]
                and fp32_logits.shape
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
            )

            gates[
                f"window_fault_{family}_fold{fold}"
            ] = passed

            window_fault_checks.append({
                "fold":
                    fold,

                "family":
                    family,

                "fault_id":
                    instance[
                        "fault_id"
                    ],

                "replay_id":
                    instance[
                        "replay_id"
                    ],

                "same_tensor_to_both_models":
                    bool(
                        identical_pair_input
                    ),

                "input_unchanged":
                    bool(
                        input_unchanged
                    ),

                "fp32_output_shape":
                    list(
                        fp32_logits.shape
                    ),

                "ptq_output_shape":
                    list(
                        ptq_logits.shape
                    ),

                "performance_metric_computed":
                    False,
            })

    # --------------------------------------------------------
    # Sequence path: real activity-only source parent per fold.
    # First clean reconstruction must be exactly equal to stored
    # windows. Then all seven sequence families execute.
    # --------------------------------------------------------

    sequence_checks = []

    univr_root = cfg[
        "source_trial_roots"
    ][
        "UNIVR"
    ]

    kfall_root = cfg[
        "source_trial_roots"
    ][
        "KFALL"
    ]

    for fold in range(
        1,
        6,
    ):
        (
            subject,
            task,
            trial,
        ) = find_activity_only_sequence_parent(
            calibration_path,
            dataset_root,
            fold=fold,
            minimum_windows=30,
        )

        stored_segments, stored_labels = (
            load_trial_segments_and_labels(
                dataset_root,
                subject=subject,
                task=task,
                trial=trial,
            )
        )

        source_path = source_trial_path(
            subject=subject,
            task=task,
            trial=trial,
            univr_root=univr_root,
            kfall_root=kfall_root,
        )

        source = load_source_trial(
            source_path
        )

        clean_rebuilt = rewindow_sequence(
            source,
            stored_labels,
            fall_start_position=None,
        )

        clean_exact = bool(
            clean_rebuilt.shape
            == stored_segments.shape
            and np.array_equal(
                clean_rebuilt,
                stored_segments,
            )
        )

        gates[
            f"clean_sequence_reconstruction_fold{fold}"
        ] = clean_exact

        if not clean_exact:
            max_diff = (
                float(
                    np.max(
                        np.abs(
                            clean_rebuilt
                            - stored_segments
                        )
                    )
                )
                if clean_rebuilt.shape
                == stored_segments.shape
                else None
            )

            raise ValueError(
                "clean historical reconstruction mismatch "
                f"fold={fold} subject={subject} task={task} trial={trial} "
                f"rebuilt={clean_rebuilt.shape} stored={stored_segments.shape} "
                f"max_diff={max_diff}"
            )

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

        references = load_reference_scales(
            severity_audit,
            fold,
        )

        for family in SEQUENCE_FAMILIES:
            instance = representative_instance(
                instances,
                family=family,
                level="L1",
            )

            kwargs = {}

            if family == "drift":
                kwargs[
                    "reference_scales"
                ] = references

            source_before = source.copy()

            corrupted_source, audit = (
                OPERATORS.apply_fault(
                    source,
                    instance,
                    **kwargs,
                )
            )

            input_unchanged = np.array_equal(
                source,
                source_before,
            )

            faulted_windows = rewindow_sequence(
                corrupted_source,
                stored_labels,
                fall_start_position=None,
            )

            if faulted_windows.shape != clean_rebuilt.shape:
                raise ValueError(
                    f"faulted rewindow shape changed for {family}/fold{fold}"
                )

            effective_delta = np.max(
                np.abs(
                    faulted_windows[
                        :,
                        :,
                        0:6
                    ]
                    - clean_rebuilt[
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

            changed_indices = np.flatnonzero(
                effective_delta
                > 0.0
            )

            propagated = bool(
                len(
                    changed_indices
                )
                > 0
            )

            if not propagated:
                raise ValueError(
                    f"sequence fault did not propagate to any stored-style window: "
                    f"family={family} fold={fold}"
                )

            selected_index = int(
                changed_indices[
                    0
                ]
            )

            selected_window = faulted_windows[
                selected_index
            ].copy()

            paired_a = selected_window.copy()
            paired_b = selected_window.copy()

            same_pair_input = np.array_equal(
                paired_a,
                paired_b,
            )

            fp32_logits, _ = (
                logits_and_positive_probability(
                    loaded_fp32[
                        (
                            42,
                            fold,
                        )
                    ],
                    paired_a,
                )
            )

            ptq_logits, _ = (
                logits_and_positive_probability(
                    loaded_ptq[
                        (
                            42,
                            fold,
                        )
                    ],
                    paired_b,
                )
            )

            passed = bool(
                input_unchanged
                and propagated
                and same_pair_input
                and audit[
                    "Euler_preserved"
                ]
                and fp32_logits.shape
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
            )

            gates[
                f"sequence_fault_{family}_fold{fold}"
            ] = passed

            sequence_checks.append({
                "fold":
                    fold,

                "subject":
                    subject,

                "task":
                    task,

                "trial":
                    trial,

                "source_path":
                    str(
                        source_path
                    ),

                "family":
                    family,

                "fault_id":
                    instance[
                        "fault_id"
                    ],

                "replay_id":
                    instance[
                        "replay_id"
                    ],

                "selected_changed_window_index":
                    selected_index,

                "same_tensor_to_both_models":
                    bool(
                        same_pair_input
                    ),

                "input_unchanged":
                    bool(
                        input_unchanged
                    ),

                "fault_propagated_to_filtered_window":
                    bool(
                        propagated
                    ),

                "performance_metric_computed":
                    False,
            })

    # --------------------------------------------------------
    # Governance gates.
    # --------------------------------------------------------

    validation_rejected = False

    try:
        assert_partition(
            "validation",
            execution_stage="qualification",
        )
    except ValueError:
        validation_rejected = True

    outer_rejected = False

    try:
        assert_partition(
            "outer_test",
            execution_stage="qualification",
        )
    except ValueError:
        outer_rejected = True

    onfield_rejected = False

    try:
        assert_partition(
            "onfield",
            execution_stage="qualification",
        )
    except ValueError:
        onfield_rejected = True

    gates[
        "qualification_rejects_validation"
    ] = validation_rejected

    gates[
        "qualification_rejects_outer_test"
    ] = outer_rejected

    gates[
        "qualification_rejects_onfield"
    ] = onfield_rejected

    gates[
        "all_12_fault_families_exercised"
    ] = (
        {
            row[
                "family"
            ]
            for row
            in window_fault_checks
            + sequence_checks
        }
        == set(
            ALL_FAMILIES
        )
    )

    gates[
        "no_performance_acceptance_gate"
    ] = (
        cfg[
            "qualification_execution"
        ][
            "performance_metric_gate"
        ]
        is False
    )

    gates[
        "no_probability_delta_acceptance_gate"
    ] = (
        cfg[
            "qualification_execution"
        ][
            "probability_delta_gate"
        ]
        is False
    )

    gates[
        "no_threshold_acceptance_gate"
    ] = (
        cfg[
            "qualification_execution"
        ][
            "threshold_gate"
        ]
        is False
    )

    gates = {
        str(
            key
        ):
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
            "phase4h_sensor_fi_devcal_runner_v1_qualification_result",

        "status":
            status,

        "phase":
            "4H",

        "partition":
            "training_calibration",

        "config_path":
            str(
                config_path
            ),

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

        "gates":
            gates,

        "counts": {
            "clean_model_pairs":
                len(
                    clean_model_checks
                ),

            "window_fault_checks":
                len(
                    window_fault_checks
                ),

            "sequence_fault_checks":
                len(
                    sequence_checks
                ),

            "fault_family_count":
                len(
                    {
                        row[
                            "family"
                        ]
                        for row
                        in window_fault_checks
                        + sequence_checks
                    }
                ),

            "threshold_rows_loaded_not_applied":
                threshold_rows_loaded,
        },

        "clean_model_checks":
            clean_model_checks,

        "window_fault_checks":
            window_fault_checks,

        "sequence_fault_checks":
            sequence_checks,

        "scientific_boundary": {
            "model_predictions_computed":
                True,

            "sensor_fault_injection_executed":
                True,

            "training_calibration_used":
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

            "probability_values_persisted":
                False,

            "robustness_comparison_made":
                False,

            "protocol_tuned_from_predictions":
                False,

            "physical_realism_claim":
                False,
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=False,
    ) if not output_path.parent.exists() else None

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    return result


def main() -> None:
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
            "Only --qualify is exposed in runner v1. "
            "Outer-test execution requires a later frozen execution manifest."
        )

    result = run_qualification(
        config_path=args.config,
        output_path=args.output,
    )

    print(
        "DEVCAL_RUNNER_QUALIFICATION_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    print(
        "DEVCAL_RUNNER_GATE_COUNT=",
        len(
            result[
                "gates"
            ]
        ),
        sep="",
    )

    print(
        "DEVCAL_RUNNER_PASS_COUNT=",
        sum(
            result[
                "gates"
            ].values()
        ),
        sep="",
    )

    print(
        "CLEAN_MODEL_PAIR_COUNT=",
        result[
            "counts"
        ][
            "clean_model_pairs"
        ],
        sep="",
    )

    print(
        "WINDOW_FAULT_CHECK_COUNT=",
        result[
            "counts"
        ][
            "window_fault_checks"
        ],
        sep="",
    )

    print(
        "SEQUENCE_FAULT_CHECK_COUNT=",
        result[
            "counts"
        ][
            "sequence_fault_checks"
        ],
        sep="",
    )

    print(
        "FAULT_FAMILY_COUNT=",
        result[
            "counts"
        ][
            "fault_family_count"
        ],
        sep="",
    )

    if result[
        "status"
    ] != "PASS":
        raise SystemExit(
            "Dev/cal runner qualification failed"
        )


if __name__ == "__main__":
    main()
