"""Phase-4H evaluation aggregation helpers.

No model inference or sensor mutation occurs here.

The helpers encode:
- subject-level primary aggregation,
- nested replicate/variant/seed means,
- paired degradation,
- deterministic dataset-stratified subject bootstrap.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

BOOTSTRAP_NAMESPACE = (
    "crosslayer-phase4h-sensor-fi-eval-bootstrap-v1"
)

HIGHER_IS_BETTER = frozenset({
    "falling_recall",
    "activity_specificity",
    "balanced_accuracy",
    "precision",
    "f1",
    "event_recall",
    "sensor_lead_ms",
})


def safe_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)

    if denominator <= 0:
        return float("nan")

    return numerator / denominator


def confusion_metrics(*, tp, fn, tn, fp):
    recall = safe_ratio(
        tp,
        tp + fn,
    )

    specificity = safe_ratio(
        tn,
        tn + fp,
    )

    precision = safe_ratio(
        tp,
        tp + fp,
    )

    if (
        math.isnan(recall)
        or math.isnan(specificity)
    ):
        balanced = float("nan")
    else:
        balanced = (
            recall
            + specificity
        ) / 2.0

    if (
        math.isnan(recall)
        or math.isnan(precision)
        or recall + precision == 0
    ):
        f1 = (
            0.0
            if (
                not math.isnan(recall)
                and not math.isnan(precision)
                and recall == 0.0
                and precision == 0.0
            )
            else float("nan")
        )
    else:
        f1 = (
            2.0
            * recall
            * precision
            / (
                recall
                + precision
            )
        )

    return {
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
    }


def nanmean_equal(values):
    x = np.asarray(
        list(values),
        dtype=float,
    )

    finite = np.isfinite(
        x
    )

    if not finite.any():
        return float("nan")

    return float(
        np.mean(
            x[finite]
        )
    )


def aggregate_nested_equal(
    rows,
    *,
    value_key,
    replicate_key="replicate_index",
    variant_key="variant_id",
    seed_key="checkpoint_seed",
):
    """Equal-weight replicate -> variant -> seed aggregation.

    Input rows must already belong to one subject/family/severity.
    """

    rows = list(rows)

    if not rows:
        return {
            "subject_value":
                float("nan"),

            "eligible_seed_count":
                0,

            "seed_values":
                {},
        }

    by_seed = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for row in rows:
        by_seed[
            row[seed_key]
        ][
            row[variant_key]
        ][
            row[replicate_key]
        ].append(
            float(
                row[value_key]
            )
        )

    seed_values = {}

    for seed, variants in by_seed.items():
        variant_values = []

        for variant, replicates in variants.items():
            replicate_values = []

            for _, values in sorted(
                replicates.items(),
                key=lambda kv: kv[0],
            ):
                replicate_values.append(
                    nanmean_equal(
                        values
                    )
                )

            variant_values.append(
                nanmean_equal(
                    replicate_values
                )
            )

        seed_values[
            str(seed)
        ] = nanmean_equal(
            variant_values
        )

    return {
        "subject_value":
            nanmean_equal(
                seed_values.values()
            ),

        "eligible_seed_count":
            int(
                sum(
                    np.isfinite(v)
                    for v in seed_values.values()
                )
            ),

        "seed_values":
            seed_values,
    }


def paired_degradation(
    *,
    metric,
    clean,
    fault,
):
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(
            f"metric not registered as higher-is-better: {metric}"
        )

    clean = float(clean)
    fault = float(fault)

    if (
        not math.isfinite(clean)
        or not math.isfinite(fault)
    ):
        return float("nan")

    return clean - fault


def bootstrap_seed(*parts):
    payload = "|".join([
        BOOTSTRAP_NAMESPACE,
        *[
            str(x)
            for x in parts
        ],
    ])

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def stratified_subject_bootstrap(
    records,
    *,
    value_key,
    subject_key="subject",
    dataset_key="dataset",
    replicates=10000,
    seed_parts=(),
):
    """Paired subject bootstrap after one value per subject is formed."""

    records = list(records)

    by_dataset = defaultdict(
        list
    )

    seen_subjects = set()

    for row in records:
        subject = str(
            row[subject_key]
        )

        if subject in seen_subjects:
            raise ValueError(
                "bootstrap input must contain one row per subject"
            )

        seen_subjects.add(
            subject
        )

        dataset = str(
            row[dataset_key]
        )

        if dataset not in {
            "UNIVR",
            "KFALL",
        }:
            raise ValueError(
                f"unexpected dataset stratum: {dataset}"
            )

        value = float(
            row[value_key]
        )

        if math.isfinite(
            value
        ):
            by_dataset[
                dataset
            ].append(
                value
            )

    if len(
        by_dataset["UNIVR"]
    ) != 29:
        raise ValueError(
            "expected 29 finite UniVR subject values"
        )

    if len(
        by_dataset["KFALL"]
    ) != 32:
        raise ValueError(
            "expected 32 finite KFall subject values"
        )

    seed = bootstrap_seed(
        *seed_parts
    )

    rng = np.random.default_rng(
        seed
    )

    u = np.asarray(
        by_dataset["UNIVR"],
        dtype=float,
    )

    k = np.asarray(
        by_dataset["KFALL"],
        dtype=float,
    )

    estimates = np.empty(
        int(replicates),
        dtype=float,
    )

    for i in range(
        int(replicates)
    ):
        ub = rng.choice(
            u,
            size=29,
            replace=True,
        )

        kb = rng.choice(
            k,
            size=32,
            replace=True,
        )

        estimates[i] = float(
            np.mean(
                np.concatenate([
                    ub,
                    kb,
                ])
            )
        )

    point = float(
        np.mean(
            np.concatenate([
                u,
                k,
            ])
        )
    )

    low, high = np.quantile(
        estimates,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "point_estimate":
            point,

        "ci95_low":
            float(
                low
            ),

        "ci95_high":
            float(
                high
            ),

        "bootstrap_replicates":
            int(
                replicates
            ),

        "seed":
            int(
                seed
            ),

        "univr_subject_count":
            29,

        "kfall_subject_count":
            32,

        "total_subject_count":
            61,
    }
