"""Phase-4H immutable outer subject aggregation v1.

The runner consumes already-completed immutable outer condition rows and
materializes the frozen subject-level hierarchy before cross-subject bootstrap.

No result-dependent selection or retuning is performed.

Hierarchy:
    replicate -> variant -> checkpoint seed -> subject

Outputs:
    clean_subject_metrics.jsonl
    subject_family_severity_macro.jsonl
    subject_individual_variant.jsonl
    subject_checkpoint_seed_specific.jsonl
    subject_timing_descriptive_family_macro.jsonl
    subject_denominator_coverage_family_macro.jsonl
    execution_interpretability.json
    coverage.json
    artifact_manifest.json
    _SUCCESS.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sensor_fi_aggregation as AGG
import sensor_fi_reporting_v2 as REPORT
import sensor_fi_outer_aggregation_adapter_v1 as ADAPTER


METRICS = (
    "falling_recall",
    "activity_specificity",
    "balanced_accuracy",
    "precision",
    "f1",
    "event_recall",
    "sensor_lead_ms",
)

TIMING_DESCRIPTIVE_FIELDS = (
    "q25_sensor_lead_ms",
    "median_sensor_lead_ms",
    "q75_sensor_lead_ms",
)

DENOMINATOR_FIELDS = (
    "TP",
    "FN",
    "TN",
    "FP",
    "eligible_event_count",
    "detected_event_count",
    "missed_event_count",
    "false_trigger_episode_count",
    "activity_seconds",
)

EXPECTED_MODELS = {
    "prospective_fp32_300ms",
    "qualified_static_ptq_v7",
}

EXPECTED_SEEDS = {
    42,
    123,
    2025,
}

EXPECTED_OPERATING_POINTS = {
    "balanced",
    "low_false_alarm",
    "timely_150ms",
}

EXPECTED_FAMILIES = {
    "bias",
    "drift",
    "scale_factor",
    "noise",
    "clipping_saturation",
    "stuck_channel",
    "axis_loss",
    "dropout",
    "frame_loss",
    "jitter",
    "delay",
    "orientation",
}

EXPECTED_SEVERITIES = {
    "L1",
    "L2",
    "L3",
}

REQUIRED_DIMENSIONS = {
    "model_variant",
    "checkpoint_seed",
    "fold",
    "subject",
    "dataset",
    "operating_point",
    "regime",
    "fault_family",
    "severity_level",
    "variant_id",
    "replicate_index",
}

METRIC_SOURCE_FIELDS = {
    "falling_recall",
    "activity_specificity",
    "balanced_accuracy",
    "precision",
    "f1",
    "event_recall",
    "median_sensor_lead_ms",
    "q25_sensor_lead_ms",
    "q75_sensor_lead_ms",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_source(metric: str) -> str:
    return ADAPTER.source_field_for_metric(metric)


def metric_subject_aggregate(rows, metric: str) -> dict:
    source = metric_source(metric)

    x = AGG.aggregate_nested_equal(
        rows,
        value_key=source,
    )

    return {
        "value": x["subject_value"],
        "eligible_seed_count":
            x["eligible_seed_count"],
        "seed_values":
            x["seed_values"],
        "source_field":
            source,
        "source_row_total":
            len(rows),
        "source_finite_row_count":
            sum(
                finite(row[source])
                for row in rows
            ),
    }


def timing_subject_aggregate(rows, field: str) -> dict:
    x = AGG.aggregate_nested_equal(
        rows,
        value_key=field,
    )

    return {
        "value":
            x["subject_value"],
        "eligible_seed_count":
            x["eligible_seed_count"],
        "source_row_total":
            len(rows),
        "source_finite_row_count":
            sum(
                finite(row[field])
                for row in rows
            ),
    }


def denominator_summary(rows) -> dict:
    rows = list(rows)

    out = {
        "source_row_count":
            len(rows),
        "noninferential_repeated_source_row_sums":
            {},
    }

    for field in DENOMINATOR_FIELDS:
        values = [
            row[field]
            for row in rows
        ]

        if field == "activity_seconds":
            value = float(
                sum(float(v) for v in values)
            )
        else:
            value = int(
                sum(int(v) for v in values)
            )

        out[
            "noninferential_repeated_source_row_sums"
        ][field] = value

    return out


def load_all_rows(plan, root: Path):
    rows = []

    for index, shard in enumerate(
        plan["shards"]
    ):
        path = (
            root
            / shard["shard_id"]
            / "subject_condition_metrics.jsonl"
        )

        if not path.is_file():
            raise RuntimeError(
                f"missing outer rows at plan index {index}: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                row["_plan_index"] = index
                rows.append(row)

    return rows


def validate_rows(
    rows,
    *,
    plan,
    config,
    lineage,
):
    gates = {}

    gates[
        "frozen_input_contract_hash_matches"
    ] = (
        lineage["input_contract"]
        == "fb7f18ed6d10fc449c32d8e8d458753daaeaa8048ab6faba388566a53cacfe3f"
    )

    gates[
        "frozen_severity_protocol_hash_matches"
    ] = (
        lineage["severity"]
        == "48b8927628f2580642c93ed9b492cf562a4ad40a21c7815bc3d5a93ca970b009"
    )

    gates[
        "qualified_sampling_v3_hash_matches"
    ] = (
        lineage["sampling_qualification"]
        == "18701172ab25147e4dfb54059a9f5e07b0b3d8f89f13025ffb4aaf282e78df7f"
    )

    gates[
        "qualified_aggregation_protocol_hash_matches"
    ] = (
        lineage["aggregation_rebind"]
        == "c0e4016fdcb7ca1bc44c4927fbf33b115b76dbb4067f459abd4a77408ecc0ee6"
    )

    if len(rows) != 320616:
        raise RuntimeError(
            f"expected 320616 rows, observed {len(rows)}"
        )

    models = {
        row["model_variant"]
        for row in rows
    }

    seeds = {
        int(row["checkpoint_seed"])
        for row in rows
    }

    operating_points = {
        row["operating_point"]
        for row in rows
    }

    folds = {
        int(row["fold"])
        for row in rows
    }

    subjects = {
        int(row["subject"])
        for row in rows
    }

    datasets = {
        row["dataset"]
        for row in rows
    }

    families = {
        row["fault_family"]
        for row in rows
        if row["regime"] == "CS"
    }

    severities = {
        row["severity_level"]
        for row in rows
        if row["regime"] == "CS"
    }

    gates[
        "all_required_checkpoints_and_model_variants_accounted_for"
    ] = (
        models == EXPECTED_MODELS
        and seeds == EXPECTED_SEEDS
        and operating_points
            == EXPECTED_OPERATING_POINTS
        and folds == {1, 2, 3, 4, 5}
        and len(subjects) == 61
        and datasets == {"KFALL", "UNIVR"}
        and families == EXPECTED_FAMILIES
        and severities == EXPECTED_SEVERITIES
    )

    digest_re = re.compile(
        r"^[0-9a-f]{64}$"
    )

    replay_ok = True

    for row in rows:
        if row["regime"] == "C0":
            if row[
                "fault_identity_digest_sha256"
            ] is not None:
                replay_ok = False
                break
        else:
            digest = row[
                "fault_identity_digest_sha256"
            ]

            if (
                not isinstance(digest, str)
                or not digest_re.fullmatch(digest)
                or int(row["fault_instance_count"])
                    <= 0
            ):
                replay_ok = False
                break

    gates[
        "all_fault_instances_have_valid_replay_ids"
    ] = replay_ok

    completion = json.loads(
        Path(
            config[
                "outer_execution_completion_path"
            ]
        ).read_text()
    )

    gates[
        "no_outer_test_tuning_occurred"
    ] = (
        completion["scientific_scope"][
            "outer_results_used_for_tuning"
        ]
        is False
        and completion["completion_gate"][
            "scientific_retuning_authorized"
        ]
        is False
    )

    gates[
        "no_OnField_fault_generation_occurred"
    ] = (
        completion["scientific_scope"][
            "onfield_used"
        ]
        is False
    )

    unique_keys = set()
    dimensions_ok = True

    for row in rows:
        if not REQUIRED_DIMENSIONS.issubset(
            row
        ):
            dimensions_ok = False
            break

        key = (
            row["model_variant"],
            int(row["checkpoint_seed"]),
            int(row["fold"]),
            int(row["subject"]),
            row["dataset"],
            row["operating_point"],
            row["regime"],
            row["fault_family"],
            row["severity_level"],
            row["variant_id"],
            int(row["replicate_index"]),
        )

        if key in unique_keys:
            dimensions_ok = False
            break

        unique_keys.add(key)

    gates[
        "all_required_result_dimensions_present"
    ] = dimensions_ok

    missingness_ok = True
    undefined_occurrences = Counter()

    for row in rows:
        undefined = set(
            row.get(
                "undefined_metric_fields",
                [],
            )
        )

        for field in undefined:
            undefined_occurrences[
                field
            ] += 1

        unknown = (
            undefined
            - METRIC_SOURCE_FIELDS
        )

        if unknown:
            missingness_ok = False
            break

        for field in METRIC_SOURCE_FIELDS:
            is_finite = finite(
                row[field]
            )

            if (
                not is_finite
                and field not in undefined
            ):
                missingness_ok = False
                break

            if (
                is_finite
                and field in undefined
            ):
                missingness_ok = False
                break

        if not missingness_ok:
            break

    gates[
        "all_missing_or_undefined_metrics_explicitly_accounted_for"
    ] = missingness_ok

    status = REPORT.execution_interpretability(
        gates
    )

    return {
        "status": status,
        "gates": gates,
        "undefined_metric_field_occurrence_counts":
            dict(
                sorted(
                    undefined_occurrences.items()
                )
            ),
        "row_count": len(rows),
        "model_variants": sorted(models),
        "checkpoint_seeds": sorted(seeds),
        "operating_points":
            sorted(operating_points),
        "folds": sorted(folds),
        "subject_count": len(subjects),
        "datasets": sorted(datasets),
        "fault_families":
            sorted(families),
        "severity_levels":
            sorted(severities),
    }


def write_jsonl(path: Path, records):
    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    allow_nan=True,
                )
                + "\n"
            )
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = json.load(
        open(args.config)
    )

    plan = json.load(
        open(args.plan)
    )

    output = Path(args.output)

    temp = Path(
        str(output)
        + ".tmp"
    )

    success_path = (
        output
        / "_SUCCESS.json"
    )

    if success_path.is_file():
        print(
            "SUBJECT_AGGREGATION_REUSED="
            + str(output)
        )
        return

    if output.exists() or temp.exists():
        if not args.recompute_partial:
            raise RuntimeError(
                "partial output exists; rerun with --recompute-partial"
            )

        if output.exists():
            shutil.rmtree(output)

        if temp.exists():
            shutil.rmtree(temp)

    temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    input_root = Path(
        cfg["outer_execution_root"]
    )

    rows = load_all_rows(
        plan,
        input_root,
    )

    lineage = {
        key: sha256(
            Path(path)
        )
        for key, path in cfg[
            "interpretability_lineage_paths"
        ].items()
    }

    interpretability = validate_rows(
        rows,
        plan=plan,
        config=cfg,
        lineage=lineage,
    )

    if interpretability["status"] != \
       "EXECUTION_INTERPRETABLE":
        raise RuntimeError(
            "execution interpretability gate did not pass"
        )

    (
        temp
        / "execution_interpretability.json"
    ).write_text(
        json.dumps(
            interpretability,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    clean_groups = defaultdict(list)
    family_groups = defaultdict(list)
    variant_groups = defaultdict(list)
    seed_groups = defaultdict(list)

    subject_meta = {}

    for row in rows:
        subject = int(
            row["subject"]
        )

        meta = (
            row["dataset"],
            int(row["fold"]),
        )

        if subject in subject_meta:
            if subject_meta[subject] != meta:
                raise RuntimeError(
                    f"subject metadata changed: {subject}"
                )
        else:
            subject_meta[
                subject
            ] = meta

        model = row["model_variant"]
        op = row["operating_point"]

        if row["regime"] == "C0":
            clean_groups[
                (
                    model,
                    op,
                    subject,
                )
            ].append(row)

            continue

        family = row["fault_family"]
        severity = row[
            "severity_level"
        ]
        variant = row[
            "variant_id"
        ]
        seed = int(
            row["checkpoint_seed"]
        )

        family_groups[
            (
                model,
                op,
                subject,
                family,
                severity,
            )
        ].append(row)

        variant_groups[
            (
                model,
                op,
                subject,
                family,
                severity,
                variant,
            )
        ].append(row)

        seed_groups[
            (
                model,
                op,
                subject,
                family,
                severity,
                seed,
            )
        ].append(row)

    expected_clean_groups = (
        2 * 3 * 61
    )

    if len(clean_groups) != \
       expected_clean_groups:
        raise RuntimeError(
            "unexpected clean group count: "
            f"{len(clean_groups)}"
        )

    expected_family_groups = (
        2 * 3 * 61 * 12 * 3
    )

    if len(family_groups) != \
       expected_family_groups:
        raise RuntimeError(
            "unexpected family group count: "
            f"{len(family_groups)}"
        )

    clean_cache = {}
    clean_seed_cache = {}

    clean_metric_records = []

    for (
        model,
        op,
        subject,
    ), group in sorted(
        clean_groups.items()
    ):
        dataset, fold = subject_meta[
            subject
        ]

        if len(group) != 3:
            raise RuntimeError(
                f"C0 expected 3 seed rows: "
                f"{model} {op} {subject}"
            )

        for metric in METRICS:
            x = metric_subject_aggregate(
                group,
                metric,
            )

            clean_cache[
                (
                    model,
                    op,
                    subject,
                    metric,
                )
            ] = x

            clean_metric_records.append({
                "model_variant": model,
                "operating_point": op,
                "subject": subject,
                "dataset": dataset,
                "fold": fold,
                "metric": metric,
                "source_field":
                    x["source_field"],
                "C0_subject_value":
                    x["value"],
                "eligible_seed_count":
                    x["eligible_seed_count"],
                "source_row_total":
                    x["source_row_total"],
                "source_finite_row_count":
                    x[
                        "source_finite_row_count"
                    ],
            })

        for seed in sorted(
            EXPECTED_SEEDS
        ):
            seed_rows = [
                row
                for row in group
                if int(
                    row["checkpoint_seed"]
                ) == seed
            ]

            if len(seed_rows) != 1:
                raise RuntimeError(
                    "C0 seed cardinality failure"
                )

            for metric in METRICS:
                x = metric_subject_aggregate(
                    seed_rows,
                    metric,
                )

                clean_seed_cache[
                    (
                        model,
                        op,
                        subject,
                        seed,
                        metric,
                    )
                ] = x

    family_records = []

    for key, group in sorted(
        family_groups.items()
    ):
        (
            model,
            op,
            subject,
            family,
            severity,
        ) = key

        dataset, fold = subject_meta[
            subject
        ]

        variants = {
            row["variant_id"]
            for row in group
        }

        replicate_slots = {
            (
                row["variant_id"],
                int(
                    row["replicate_index"]
                ),
            )
            for row in group
        }

        for metric in METRICS:
            fault = metric_subject_aggregate(
                group,
                metric,
            )

            clean = clean_cache[
                (
                    model,
                    op,
                    subject,
                    metric,
                )
            ]

            degradation = AGG.paired_degradation(
                metric=metric,
                clean=clean["value"],
                fault=fault["value"],
            )

            family_records.append({
                "detail_type":
                    "family_severity_macro",
                "model_variant":
                    model,
                "operating_point":
                    op,
                "subject":
                    subject,
                "dataset":
                    dataset,
                "fold":
                    fold,
                "fault_family":
                    family,
                "severity_level":
                    severity,
                "metric":
                    metric,
                "source_field":
                    fault["source_field"],
                "C0_subject_value":
                    clean["value"],
                "CS_subject_value":
                    fault["value"],
                "paired_degradation":
                    degradation,
                "C0_eligible_seed_count":
                    clean[
                        "eligible_seed_count"
                    ],
                "CS_eligible_seed_count":
                    fault[
                        "eligible_seed_count"
                    ],
                "C0_source_row_total":
                    clean[
                        "source_row_total"
                    ],
                "C0_source_finite_row_count":
                    clean[
                        "source_finite_row_count"
                    ],
                "CS_source_row_total":
                    fault[
                        "source_row_total"
                    ],
                "CS_source_finite_row_count":
                    fault[
                        "source_finite_row_count"
                    ],
                "CS_variant_count":
                    len(variants),
                "CS_variant_replicate_slot_count":
                    len(replicate_slots),
            })

    variant_records = []

    for key, group in sorted(
        variant_groups.items()
    ):
        (
            model,
            op,
            subject,
            family,
            severity,
            variant,
        ) = key

        dataset, fold = subject_meta[
            subject
        ]

        for metric in METRICS:
            fault = metric_subject_aggregate(
                group,
                metric,
            )

            clean = clean_cache[
                (
                    model,
                    op,
                    subject,
                    metric,
                )
            ]

            degradation = AGG.paired_degradation(
                metric=metric,
                clean=clean["value"],
                fault=fault["value"],
            )

            variant_records.append({
                "detail_type":
                    "individual_fault_variant",
                "model_variant":
                    model,
                "operating_point":
                    op,
                "subject":
                    subject,
                "dataset":
                    dataset,
                "fold":
                    fold,
                "fault_family":
                    family,
                "severity_level":
                    severity,
                "variant_id":
                    variant,
                "metric":
                    metric,
                "source_field":
                    fault["source_field"],
                "C0_subject_value":
                    clean["value"],
                "CS_subject_value":
                    fault["value"],
                "paired_degradation":
                    degradation,
                "C0_source_row_total":
                    clean[
                        "source_row_total"
                    ],
                "C0_source_finite_row_count":
                    clean[
                        "source_finite_row_count"
                    ],
                "CS_source_row_total":
                    fault[
                        "source_row_total"
                    ],
                "CS_source_finite_row_count":
                    fault[
                        "source_finite_row_count"
                    ],
                "CS_replicate_count":
                    len({
                        int(
                            row[
                                "replicate_index"
                            ]
                        )
                        for row in group
                    }),
            })

    seed_records = []

    for key, group in sorted(
        seed_groups.items()
    ):
        (
            model,
            op,
            subject,
            family,
            severity,
            seed,
        ) = key

        dataset, fold = subject_meta[
            subject
        ]

        for metric in METRICS:
            fault = metric_subject_aggregate(
                group,
                metric,
            )

            clean = clean_seed_cache[
                (
                    model,
                    op,
                    subject,
                    seed,
                    metric,
                )
            ]

            degradation = AGG.paired_degradation(
                metric=metric,
                clean=clean["value"],
                fault=fault["value"],
            )

            seed_records.append({
                "detail_type":
                    "checkpoint_seed_specific",
                "model_variant":
                    model,
                "operating_point":
                    op,
                "subject":
                    subject,
                "dataset":
                    dataset,
                "fold":
                    fold,
                "fault_family":
                    family,
                "severity_level":
                    severity,
                "checkpoint_seed":
                    seed,
                "metric":
                    metric,
                "source_field":
                    fault["source_field"],
                "C0_subject_value":
                    clean["value"],
                "CS_subject_value":
                    fault["value"],
                "paired_degradation":
                    degradation,
                "C0_source_row_total":
                    clean[
                        "source_row_total"
                    ],
                "C0_source_finite_row_count":
                    clean[
                        "source_finite_row_count"
                    ],
                "CS_source_row_total":
                    fault[
                        "source_row_total"
                    ],
                "CS_source_finite_row_count":
                    fault[
                        "source_finite_row_count"
                    ],
            })

    timing_records = []

    for key, group in sorted(
        family_groups.items()
    ):
        (
            model,
            op,
            subject,
            family,
            severity,
        ) = key

        dataset, fold = subject_meta[
            subject
        ]

        clean_group = clean_groups[
            (
                model,
                op,
                subject,
            )
        ]

        for field in (
            TIMING_DESCRIPTIVE_FIELDS
        ):
            c0 = timing_subject_aggregate(
                clean_group,
                field,
            )

            cs = timing_subject_aggregate(
                group,
                field,
            )

            timing_records.append({
                "detail_type":
                    "family_severity_timing_descriptive",
                "model_variant":
                    model,
                "operating_point":
                    op,
                "subject":
                    subject,
                "dataset":
                    dataset,
                "fold":
                    fold,
                "fault_family":
                    family,
                "severity_level":
                    severity,
                "timing_summary_field":
                    field,
                "C0_subject_value":
                    c0["value"],
                "CS_subject_value":
                    cs["value"],
                "C0_source_row_total":
                    c0[
                        "source_row_total"
                    ],
                "C0_source_finite_row_count":
                    c0[
                        "source_finite_row_count"
                    ],
                "CS_source_row_total":
                    cs[
                        "source_row_total"
                    ],
                "CS_source_finite_row_count":
                    cs[
                        "source_finite_row_count"
                    ],
            })

    denominator_records = []

    for key, group in sorted(
        family_groups.items()
    ):
        (
            model,
            op,
            subject,
            family,
            severity,
        ) = key

        dataset, fold = subject_meta[
            subject
        ]

        clean_group = clean_groups[
            (
                model,
                op,
                subject,
            )
        ]

        denominator_records.append({
            "detail_type":
                "family_severity_source_denominators",
            "interpretation":
                "noninferential_raw_repeated_source_condition_totals",
            "model_variant":
                model,
            "operating_point":
                op,
            "subject":
                subject,
            "dataset":
                dataset,
            "fold":
                fold,
            "fault_family":
                family,
            "severity_level":
                severity,
            "C0":
                denominator_summary(
                    clean_group
                ),
            "CS":
                denominator_summary(
                    group
                ),
        })

    counts = {}

    counts[
        "clean_subject_metrics"
    ] = write_jsonl(
        temp
        / "clean_subject_metrics.jsonl",
        clean_metric_records,
    )

    counts[
        "subject_family_severity_macro"
    ] = write_jsonl(
        temp
        / "subject_family_severity_macro.jsonl",
        family_records,
    )

    counts[
        "subject_individual_fault_variant"
    ] = write_jsonl(
        temp
        / "subject_individual_fault_variant.jsonl",
        variant_records,
    )

    counts[
        "subject_checkpoint_seed_specific"
    ] = write_jsonl(
        temp
        / "subject_checkpoint_seed_specific.jsonl",
        seed_records,
    )

    counts[
        "subject_timing_descriptive_family_macro"
    ] = write_jsonl(
        temp
        / "subject_timing_descriptive_family_macro.jsonl",
        timing_records,
    )

    counts[
        "subject_denominator_coverage_family_macro"
    ] = write_jsonl(
        temp
        / "subject_denominator_coverage_family_macro.jsonl",
        denominator_records,
    )

    if counts[
        "clean_subject_metrics"
    ] != 2 * 3 * 61 * 7:
        raise RuntimeError(
            "clean metric record count mismatch"
        )

    if counts[
        "subject_family_severity_macro"
    ] != 2 * 3 * 61 * 12 * 3 * 7:
        raise RuntimeError(
            "family macro record count mismatch"
        )

    if counts[
        "subject_checkpoint_seed_specific"
    ] != 2 * 3 * 61 * 12 * 3 * 3 * 7:
        raise RuntimeError(
            "seed-specific record count mismatch"
        )

    if counts[
        "subject_timing_descriptive_family_macro"
    ] != 2 * 3 * 61 * 12 * 3 * 3:
        raise RuntimeError(
            "timing descriptive record count mismatch"
        )

    if counts[
        "subject_denominator_coverage_family_macro"
    ] != 2 * 3 * 61 * 12 * 3:
        raise RuntimeError(
            "denominator record count mismatch"
        )

    coverage = {
        "schema_version":
            "phase4h_sensor_fi_outer_subject_aggregation_v1_coverage",
        "status": "PASS",
        "input_outer_condition_rows":
            len(rows),
        "subjects":
            len(subject_meta),
        "models":
            sorted(EXPECTED_MODELS),
        "checkpoint_seeds":
            sorted(EXPECTED_SEEDS),
        "operating_points":
            sorted(
                EXPECTED_OPERATING_POINTS
            ),
        "fault_families":
            sorted(EXPECTED_FAMILIES),
        "severity_levels":
            sorted(EXPECTED_SEVERITIES),
        "record_counts":
            counts,
        "interpretability_status":
            interpretability["status"],
        "performance_values_printed":
            False,
        "result_dependent_selection":
            False,
    }

    (
        temp
        / "coverage.json"
    ).write_text(
        json.dumps(
            coverage,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_files = [
        "clean_subject_metrics.jsonl",
        "subject_family_severity_macro.jsonl",
        "subject_individual_fault_variant.jsonl",
        "subject_checkpoint_seed_specific.jsonl",
        "subject_timing_descriptive_family_macro.jsonl",
        "subject_denominator_coverage_family_macro.jsonl",
        "execution_interpretability.json",
        "coverage.json",
    ]

    artifact_manifest = {
        "schema_version":
            "phase4h_sensor_fi_outer_subject_aggregation_v1_artifact_manifest",
        "status":
            "PASS",
        "files": {
            name: {
                "sha256":
                    sha256(
                        temp
                        / name
                    )
            }
            for name in artifact_files
        },
        "scientific_boundary": {
            "outer_metrics_read":
                True,
            "outer_metrics_printed":
                False,
            "outer_metrics_used_for_retuning":
                False,
            "outer_metrics_used_for_selection":
                False,
            "OnField_used":
                False,
            "cross_subject_bootstrap_executed":
                False,
            "direction_status_classification_executed":
                False,
            "global_robustness_label_generated":
                False,
        },
    }

    (
        temp
        / "artifact_manifest.json"
    ).write_text(
        json.dumps(
            artifact_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    success = {
        "status":
            "PASS",
        "schema_version":
            "phase4h_sensor_fi_outer_subject_aggregation_v1_success",
        "artifact_manifest_sha256":
            sha256(
                temp
                / "artifact_manifest.json"
            ),
        "coverage_sha256":
            sha256(
                temp
                / "coverage.json"
            ),
        "interpretability_sha256":
            sha256(
                temp
                / "execution_interpretability.json"
            ),
        "performance_values_printed":
            False,
        "cross_subject_bootstrap_executed":
            False,
    }

    (
        temp
        / "_SUCCESS.json"
    ).write_text(
        json.dumps(
            success,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(
        output
    )

    print(
        "SUBJECT_AGGREGATION_STATUS=PASS"
    )

    print(
        "EXECUTION_INTERPRETABILITY="
        + interpretability["status"]
    )

    print(
        "INPUT_OUTER_ROWS="
        + str(len(rows))
    )

    for key in sorted(counts):
        print(
            f"{key.upper()}_ROWS="
            f"{counts[key]}"
        )

    print(
        "ARTIFACT_MANIFEST_SHA256="
        + sha256(
            output
            / "artifact_manifest.json"
        )
    )

    print(
        "SUCCESS_SHA256="
        + sha256(
            output
            / "_SUCCESS.json"
        )
    )

    print(
        "PERFORMANCE_VALUES_PRINTED_OR_INTERPRETED=False"
    )

    print(
        "CROSS_SUBJECT_BOOTSTRAP_EXECUTED=False"
    )


if __name__ == "__main__":
    main()
