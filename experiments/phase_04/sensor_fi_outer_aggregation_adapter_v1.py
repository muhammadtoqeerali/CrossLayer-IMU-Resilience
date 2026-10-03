"""Phase-4H executable outer aggregation adapter v1.

This module closes execution-only gaps between the already-frozen
aggregation/reporting mathematics and the immutable outer row schema.

It does not select metrics, severities, variants, operating points,
models, datasets, or thresholds from outer outcomes.

Bindings implemented here:

1. The conceptual reporting metric ``sensor_lead_ms`` uses the frozen
   subject-level timing summary ``median_sensor_lead_ms`` for inferential
   replicate -> variant -> seed -> subject aggregation.

2. ``q25_sensor_lead_ms`` and ``q75_sensor_lead_ms`` remain descriptive
   subject-level timing summaries and are not substituted for the
   inferential median.

3. Dataset-specific percentile intervals are the one-stratum
   specialization of the already-qualified subject bootstrap:
   resample subjects with replacement within exactly one frozen dataset
   stratum, using the same frozen bootstrap namespace/seed algorithm.

4. If complete finite subject coverage is not available for a requested
   inferential stratum, no confidence interval is manufactured.
   The finite-subject macro point estimate is retained with explicit
   coverage and the reporting direction status must be UNRESOLVED.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import sensor_fi_aggregation as AGG


TIMING_REPORT_METRIC = "sensor_lead_ms"
TIMING_INFERENTIAL_SOURCE_FIELD = "median_sensor_lead_ms"

TIMING_DESCRIPTIVE_SOURCE_FIELDS = (
    "q25_sensor_lead_ms",
    "median_sensor_lead_ms",
    "q75_sensor_lead_ms",
)

DATASET_SUBJECT_COUNTS = {
    "UNIVR": 29,
    "KFALL": 32,
}

REQUIRED_OVERALL_SUBJECT_COUNT = 61


def source_field_for_metric(metric: str) -> str:
    """Map frozen reporting metric name to immutable outer-row field."""

    if metric == TIMING_REPORT_METRIC:
        return TIMING_INFERENTIAL_SOURCE_FIELD

    return metric


def finite_subject_macro(
    records: Iterable[dict],
    *,
    value_key: str,
) -> dict:
    """Equal subject macro over finite values with explicit coverage."""

    rows = list(records)

    values = np.asarray(
        [
            float(row[value_key])
            for row in rows
        ],
        dtype=float,
    )

    finite = np.isfinite(values)

    if finite.any():
        point = float(
            np.mean(values[finite])
        )
    else:
        point = float("nan")

    return {
        "point_estimate": point,
        "eligible_subject_count": int(
            finite.sum()
        ),
        "total_subject_count": int(
            len(rows)
        ),
        "coverage_complete": bool(
            finite.all()
        ),
    }


def one_stratum_subject_bootstrap(
    records: Iterable[dict],
    *,
    value_key: str,
    dataset: str,
    subject_key: str = "subject",
    dataset_key: str = "dataset",
    replicates: int = 10000,
    seed_parts=(),
) -> dict:
    """One-dataset specialization of the qualified subject bootstrap.

    Requires exactly one row per frozen subject in the requested dataset
    and complete finite metric coverage. The resampling unit remains the
    subject, with replacement, and the interval remains the 2.5th/97.5th
    percentile interval.
    """

    dataset = str(dataset)

    if dataset not in DATASET_SUBJECT_COUNTS:
        raise ValueError(
            f"unexpected dataset stratum: {dataset}"
        )

    rows = list(records)

    seen_subjects = set()
    values = []

    for row in rows:
        row_dataset = str(
            row[dataset_key]
        )

        if row_dataset != dataset:
            raise ValueError(
                "one-stratum bootstrap received row from "
                f"{row_dataset!r}, expected {dataset!r}"
            )

        subject = str(
            row[subject_key]
        )

        if subject in seen_subjects:
            raise ValueError(
                "bootstrap input must contain one row per subject"
            )

        seen_subjects.add(subject)

        value = float(
            row[value_key]
        )

        if not math.isfinite(value):
            raise ValueError(
                "dataset-specific bootstrap requires complete "
                "finite subject coverage"
            )

        values.append(value)

    expected_n = DATASET_SUBJECT_COUNTS[dataset]

    if len(values) != expected_n:
        raise ValueError(
            f"expected {expected_n} {dataset} subject values, "
            f"observed {len(values)}"
        )

    seed = AGG.bootstrap_seed(
        *seed_parts
    )

    rng = np.random.default_rng(
        seed
    )

    x = np.asarray(
        values,
        dtype=float,
    )

    estimates = np.empty(
        int(replicates),
        dtype=float,
    )

    for i in range(
        int(replicates)
    ):
        xb = rng.choice(
            x,
            size=expected_n,
            replace=True,
        )

        estimates[i] = float(
            np.mean(xb)
        )

    point = float(
        np.mean(x)
    )

    low, high = np.quantile(
        estimates,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "point_estimate": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": int(
            replicates
        ),
        "seed": int(seed),
        "dataset": dataset,
        "eligible_subject_count": expected_n,
        "total_subject_count": expected_n,
        "coverage_complete": True,
    }


def inferential_subject_bootstrap(
    records: Iterable[dict],
    *,
    value_key: str,
    stratum: str,
    subject_key: str = "subject",
    dataset_key: str = "dataset",
    replicates: int = 10000,
    seed_parts=(),
) -> dict:
    """Dispatch frozen overall or one-stratum subject bootstrap.

    Incomplete coverage returns the finite-subject macro with NaN interval;
    it does not silently drop missing subjects from an inferential bootstrap.
    """

    rows = list(records)

    if stratum == "overall_61_subject":
        expected_n = REQUIRED_OVERALL_SUBJECT_COUNT
    elif stratum in DATASET_SUBJECT_COUNTS:
        expected_n = DATASET_SUBJECT_COUNTS[
            stratum
        ]
    else:
        raise ValueError(
            f"unexpected reporting stratum: {stratum}"
        )

    macro = finite_subject_macro(
        rows,
        value_key=value_key,
    )

    if (
        macro["total_subject_count"] != expected_n
        or not macro["coverage_complete"]
    ):
        return {
            **macro,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_replicates": 0,
            "seed": None,
            "stratum": stratum,
        }

    if stratum == "overall_61_subject":
        result = AGG.stratified_subject_bootstrap(
            rows,
            value_key=value_key,
            subject_key=subject_key,
            dataset_key=dataset_key,
            replicates=replicates,
            seed_parts=seed_parts,
        )

        return {
            **result,
            "eligible_subject_count":
                REQUIRED_OVERALL_SUBJECT_COUNT,
            "coverage_complete": True,
            "stratum": stratum,
        }

    result = one_stratum_subject_bootstrap(
        rows,
        value_key=value_key,
        dataset=stratum,
        subject_key=subject_key,
        dataset_key=dataset_key,
        replicates=replicates,
        seed_parts=seed_parts,
    )

    return {
        **result,
        "stratum": stratum,
    }


def timing_descriptive_fields() -> tuple[str, ...]:
    """Return the predeclared descriptive timing summaries unchanged."""

    return TIMING_DESCRIPTIVE_SOURCE_FIELDS
