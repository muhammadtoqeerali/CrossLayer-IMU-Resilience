"""Phase-5Z compute-fault CC outcome-analysis core.

This module contains pure analysis primitives only.

It deliberately contains no prospective result-root path, no JSONL reader,
no NumPy label-array loader, no model loading, and no fault execution.
Prospective I/O must be implemented separately after this core is qualified.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score


REQUIRED_SEEDS = (42, 123, 2025)
OPERATING_POINTS = (
    "balanced",
    "low_false_alarm",
    "timely_150ms",
)

CLEAN_SOFTMAX_FIELD = "clean_softmax_values"
FAULTED_SOFTMAX_FIELD = "faulted_softmax_values"


class AnalysisContractError(ValueError):
    """Raised when an input violates the frozen Phase-5Z contract."""


def normalise_activity_label(value: object) -> int:
    """Historical exact binary normalization: Activity=0, Falling=1."""
    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    return int(
        "FALL"
        in str(value).strip().upper()
    )


def threshold_index(
    protocol: Mapping[str, Any],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    matrix = protocol[
        "threshold_protocol"
    ][
        "matrix"
    ]

    index: dict[
        tuple[int, int, str],
        dict[str, Any],
    ] = {}

    for row in matrix:
        key = (
            int(row["seed"]),
            int(row["fold"]),
            str(row["operating_point"]),
        )

        if key in index:
            raise AnalysisContractError(
                f"duplicate threshold key: {key}"
            )

        index[key] = {
            "seed":
                int(row["seed"]),

            "fold":
                int(row["fold"]),

            "operating_point":
                str(row["operating_point"]),

            "threshold":
                float(row["threshold"]),

            "required_consecutive":
                int(row["required_consecutive"]),
        }

    expected = {
        (
            seed,
            fold,
            operating_point,
        )
        for seed in REQUIRED_SEEDS
        for fold in range(1, 6)
        for operating_point in OPERATING_POINTS
    }

    if set(index) != expected:
        raise AnalysisContractError(
            "threshold matrix does not contain the exact frozen 45 keys"
        )

    return index


def threshold_rule(
    protocol: Mapping[str, Any],
    *,
    seed: int,
    fold: int,
    operating_point: str,
) -> tuple[float, int]:
    row = threshold_index(protocol).get(
        (
            int(seed),
            int(fold),
            str(operating_point),
        )
    )

    if row is None:
        raise AnalysisContractError(
            "missing frozen threshold key"
        )

    return (
        float(row["threshold"]),
        int(row["required_consecutive"]),
    )


def _decode_float_token(
    value: Any,
) -> float:
    """Decode the exact Phase-5M/5R JSON float-token representation.

    Finite values remain ordinary JSON numbers.

    The frozen producers serialize non-finite IEEE values as exactly one-key
    dictionaries with key ``nonfinite`` and token ``nan``, ``+inf`` or
    ``-inf``. Decoding restores the original IEEE value; it does not impute,
    clip, threshold, or otherwise alter it.
    """

    if isinstance(
        value,
        Mapping,
    ):
        if set(
            value
        ) != {
            "nonfinite"
        }:
            raise AnalysisContractError(
                "invalid serialized float-token dictionary shape"
            )

        token = value[
            "nonfinite"
        ]

        if token == "nan":
            return math.nan

        if token == "+inf":
            return math.inf

        if token == "-inf":
            return -math.inf

        raise AnalysisContractError(
            f"unknown serialized nonfinite token: {token!r}"
        )

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AnalysisContractError(
            f"invalid serialized floating value: {value!r}"
        ) from exc


def _softmax_vector(
    record: Mapping[str, Any],
    *,
    field: str,
) -> np.ndarray:
    if field not in record:
        raise AnalysisContractError(
            f"record missing {field}"
        )

    raw = record[
        field
    ]

    if not isinstance(
        raw,
        Sequence,
    ) or isinstance(
        raw,
        (
            str,
            bytes,
        ),
    ):
        raise AnalysisContractError(
            f"{field} must be a two-element sequence"
        )

    if len(
        raw
    ) != 2:
        raise AnalysisContractError(
            f"{field} must have shape (2,), got length {len(raw)}"
        )

    values = np.asarray(
        [
            _decode_float_token(
                value
            )
            for value in raw
        ],
        dtype=float,
    )

    if values.shape != (2,):
        raise AnalysisContractError(
            f"{field} must have shape (2,), got {values.shape}"
        )

    return values


def clean_falling_probability(
    record: Mapping[str, Any],
) -> float:
    return float(
        _softmax_vector(
            record,
            field=CLEAN_SOFTMAX_FIELD,
        )[1]
    )


def faulted_falling_probability(
    record: Mapping[str, Any],
) -> float:
    return float(
        _softmax_vector(
            record,
            field=FAULTED_SOFTMAX_FIELD,
        )[1]
    )


def trigger_episodes(
    probabilities: np.ndarray | Sequence[float],
    threshold: float,
    required_consecutive: int,
) -> list[int]:
    required_consecutive = int(
        required_consecutive
    )

    if required_consecutive < 1:
        raise AnalysisContractError(
            "required_consecutive must be >= 1"
        )

    episodes: list[int] = []
    run = 0
    armed = True

    for index, probability in enumerate(
        np.asarray(
            probabilities,
            dtype=float,
        )
    ):
        if probability >= float(threshold):
            run += 1

            if (
                armed
                and run >= required_consecutive
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
    probabilities: np.ndarray | Sequence[float],
    threshold: float,
    required_consecutive: int,
    labels: np.ndarray | Sequence[int] | None,
) -> int:
    for start in trigger_episodes(
        probabilities,
        threshold,
        required_consecutive,
    ):
        if labels is None:
            return start

        stop = (
            start
            + int(required_consecutive)
        )

        if (
            stop <= len(labels)
            and np.all(
                np.asarray(labels)[
                    start:stop
                ]
                == 1
            )
        ):
            return start

    return -1


def event_metrics(
    rows: list[dict[str, object]],
    threshold: float,
    consecutive: int,
) -> dict[str, float | int]:
    """Exact Phase-4 event metric semantics."""

    y_true: list[int] = []
    y_pred: list[int] = []
    false_episodes = 0
    activity_seconds = 0.0
    leads: list[float] = []
    by_150 = 0
    detected_falls = 0

    for row in rows:
        true_fall = (
            str(row["true_event"])
            == "FALLING"
        )

        probs = np.asarray(
            row["segment_probabilities"],
            dtype=float,
        )

        labels = np.asarray(
            row["segment_labels"],
            dtype=int,
        )

        episodes = trigger_episodes(
            probs,
            threshold,
            consecutive,
        )

        trigger = first_valid_trigger(
            probs,
            threshold,
            consecutive,
            labels
            if true_fall
            else None,
        )

        predicted = trigger >= 0

        y_true.append(
            int(true_fall)
        )

        y_pred.append(
            int(predicted)
        )

        sampling_rate = float(
            row.get(
                "sampling_rate_hz",
                100.0,
            )
        )

        if not true_fall:
            ends = np.asarray(
                row["window_ends"],
                dtype=int,
            )

            duration_samples = (
                int(ends[-1])
                if len(ends)
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
                row.get(
                    "fall_start_position",
                    -1,
                )
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
                    + int(consecutive)
                )

                if (
                    stop > len(labels)
                    or not np.all(
                        labels[
                            episode:stop
                        ]
                        == 1
                    )
                ):
                    false_episodes += 1

            if predicted:
                detected_falls += 1

                confirmation_index = (
                    trigger
                    + int(consecutive)
                    - 1
                )

                end = int(
                    np.asarray(
                        row["window_ends"]
                    )[
                        confirmation_index
                    ]
                )

                impact = int(
                    row.get(
                        "impact_position",
                        -1,
                    )
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
                        float(lead)
                    )

                    if lead >= 150.0:
                        by_150 += 1

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    recall = (
        tp
        / max(
            tp + fn,
            1,
        )
    )

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    fall_total = max(
        tp + fn,
        1,
    )

    return {
        "balanced_accuracy":
            (
                recall
                + specificity
            )
            / 2,

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "fall_recall":
            recall,

        "fall_precision":
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),

        "activity_specificity":
            specificity,

        "recall_by_150ms":
            by_150
            / fall_total,

        "false_triggers_per_activity_hour":
            false_episodes
            / max(
                activity_seconds
                / 3600.0,
                1e-9,
            ),

        "median_trigger_lead_ms":
            (
                float(
                    np.median(
                        leads
                    )
                )
                if leads
                else math.nan
            ),

        "mean_trigger_lead_ms":
            (
                float(
                    np.mean(
                        leads
                    )
                )
                if leads
                else math.nan
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


def _execution_window_index(
    record: Mapping[str, Any],
) -> int:
    if (
        "execution_window_index"
        in record
    ):
        return int(
            record[
                "execution_window_index"
            ]
        )

    parent = record.get(
        "parent"
    )

    if (
        isinstance(
            parent,
            Mapping,
        )
        and parent.get(
            "window_index"
        )
        is not None
    ):
        return int(
            parent[
                "window_index"
            ]
        )

    raise AnalysisContractError(
        "fault record has no execution-window index"
    )


def reconstruct_fault_scenario(
    clean_probabilities: Sequence[float],
    fault_records: Sequence[Mapping[str, Any]],
    *,
    outer_instance_id: str,
    persistence: str,
) -> np.ndarray:
    """Reconstruct one frozen outer fault identity over one clean trial."""

    clean = np.asarray(
        clean_probabilities,
        dtype=float,
    )

    if clean.ndim != 1:
        raise AnalysisContractError(
            "clean probability sequence must be one-dimensional"
        )

    selected = [
        record
        for record in fault_records
        if str(
            record.get(
                "outer_instance_id"
            )
        )
        == str(
            outer_instance_id
        )
    ]

    if not selected:
        raise AnalysisContractError(
            "outer_instance_id has no fault records"
        )

    indices = [
        _execution_window_index(
            record
        )
        for record in selected
    ]

    if len(
        set(indices)
    ) != len(indices):
        raise AnalysisContractError(
            "duplicate execution_window_index within outer_instance_id"
        )

    if any(
        index < 0
        or index >= len(clean)
        for index in indices
    ):
        raise AnalysisContractError(
            "fault execution window outside clean trial"
        )

    output = clean.copy()

    if persistence == "transient_one_inference":
        if len(selected) != 1:
            raise AnalysisContractError(
                "transient identity must have exactly one fault record"
            )

    elif persistence == "persistent_from_onset_to_trial_end":
        onset_values = {
            int(
                record[
                    "onset_or_inference_index"
                ]
            )
            for record in selected
        }

        if len(onset_values) != 1:
            raise AnalysisContractError(
                "persistent identity has inconsistent onset"
            )

        onset = next(
            iter(onset_values)
        )

        expected = list(
            range(
                onset,
                len(clean),
            )
        )

        if sorted(indices) != expected:
            raise AnalysisContractError(
                "persistent fault rows do not cover the exact onset-to-end suffix"
            )

    else:
        raise AnalysisContractError(
            f"unknown persistence mode: {persistence}"
        )

    for record, index in zip(
        selected,
        indices,
        strict=True,
    ):
        output[
            index
        ] = faulted_falling_probability(
            record
        )

    return output


def paired_clean_counterfactual(
    clean_probabilities: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        clean_probabilities,
        dtype=float,
    ).copy()


def paired_metric_degradation(
    metric_name: str,
    *,
    clean_value: float,
    faulted_value: float,
) -> float:
    if metric_name == "false_triggers_per_activity_hour":
        return float(
            faulted_value
            - clean_value
        )

    return float(
        clean_value
        - faulted_value
    )


def seed_equal_subject_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    subject_key: str = "subject",
    seed_key: str = "seed",
    value_key: str = "value",
) -> dict[int, float]:
    """Equal-seed mean within each subject.

    The input must contain exactly one finite scalar per frozen seed for each
    subject. Missing/non-finite timing summaries must therefore be handled by a
    later protocol-qualified timing aggregation layer rather than silently
    dropped here.
    """

    grouped: dict[
        int,
        dict[int, float],
    ] = defaultdict(dict)

    for row in rows:
        subject = int(
            row[
                subject_key
            ]
        )

        seed = int(
            row[
                seed_key
            ]
        )

        value = float(
            row[
                value_key
            ]
        )

        if seed not in REQUIRED_SEEDS:
            raise AnalysisContractError(
                f"unexpected checkpoint seed: {seed}"
            )

        if seed in grouped[
            subject
        ]:
            raise AnalysisContractError(
                "duplicate subject/seed scalar"
            )

        if not math.isfinite(
            value
        ):
            raise AnalysisContractError(
                "non-finite subject/seed scalar requires explicit metric-specific handling"
            )

        grouped[
            subject
        ][
            seed
        ] = value

    output: dict[
        int,
        float,
    ] = {}

    for subject, values in grouped.items():
        if set(
            values
        ) != set(
            REQUIRED_SEEDS
        ):
            raise AnalysisContractError(
                f"subject {subject} does not have exactly the three frozen seeds"
            )

        output[
            subject
        ] = float(
            np.mean([
                values[
                    seed
                ]
                for seed in REQUIRED_SEEDS
            ])
        )

    return output


def equal_subject_mean(
    subject_values: Mapping[int, float],
) -> float:
    if not subject_values:
        raise AnalysisContractError(
            "no subject values"
        )

    values = np.asarray(
        list(
            subject_values.values()
        ),
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise AnalysisContractError(
            "subject aggregation received non-finite value"
        )

    return float(
        np.mean(
            values
        )
    )


def subject_cluster_percentile_ci(
    subject_values: Mapping[int, float],
    *,
    replicates: int = 10000,
    confidence_level: float = 0.95,
    rng_seed: int = 20261006,
) -> dict[str, float | int]:
    if not subject_values:
        raise AnalysisContractError(
            "no subject values"
        )

    if int(replicates) < 1:
        raise AnalysisContractError(
            "bootstrap replicates must be positive"
        )

    if not (
        0.0
        < float(confidence_level)
        < 1.0
    ):
        raise AnalysisContractError(
            "confidence level must be in (0,1)"
        )

    values = np.asarray(
        list(
            subject_values.values()
        ),
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise AnalysisContractError(
            "bootstrap received non-finite subject value"
        )

    rng = np.random.default_rng(
        int(rng_seed)
    )

    draws = np.empty(
        int(replicates),
        dtype=float,
    )

    count = len(
        values
    )

    for index in range(
        int(replicates)
    ):
        sample_indices = rng.integers(
            0,
            count,
            size=count,
        )

        draws[
            index
        ] = float(
            np.mean(
                values[
                    sample_indices
                ]
            )
        )

    alpha = (
        1.0
        - float(
            confidence_level
        )
    )

    lower = float(
        np.quantile(
            draws,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            draws,
            1.0 - alpha / 2.0,
        )
    )

    return {
        "estimate":
            float(
                np.mean(
                    values
                )
            ),

        "lower":
            lower,

        "upper":
            upper,

        "subject_count":
            int(count),

        "bootstrap_replicates":
            int(replicates),

        "rng_seed":
            int(rng_seed),
    }


def count_nonfinite_fault_records(
    records: Iterable[Mapping[str, Any]],
) -> int:
    count = 0

    for record in records:
        if bool(
            record.get(
                "faulted_output_nonfinite",
                False,
            )
        ):
            count += 1

    return count
