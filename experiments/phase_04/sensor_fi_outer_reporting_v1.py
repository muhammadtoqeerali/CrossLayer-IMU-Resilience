"""Phase-4H frozen outer reporting v1.

Mandatory inferential report:
  model × operating point × fault family × severity × metric

For each such condition, report:
  - C0 subject macro
  - CS subject macro
  - paired C0-CS degradation
  - 95% subject-bootstrap CI for paired degradation
  - frozen direction status
  - subject coverage
  - required overall/UNIVR/KFALL strata
  - descriptive source denominators

Individual fault variants and checkpoint seeds remain required descriptive
detail. They are not promoted into additional inferential hypothesis families.

No global binary robustness label is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sensor_fi_outer_aggregation_adapter_v1 as ADAPTER
import sensor_fi_reporting_v2 as REPORT


STRATA = (
    "overall_61_subject",
    "UNIVR",
    "KFALL",
)

EXPECTED_STRATUM_N = {
    "overall_61_subject": 61,
    "UNIVR": 29,
    "KFALL": 32,
}

METRICS = (
    "falling_recall",
    "activity_specificity",
    "balanced_accuracy",
    "precision",
    "f1",
    "event_recall",
    "sensor_lead_ms",
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path: Path, rows):
    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    allow_nan=True,
                )
                + "\n"
            )
            count += 1

    return count


def select_stratum(rows, stratum):
    if stratum == "overall_61_subject":
        return list(rows)

    if stratum in {
        "UNIVR",
        "KFALL",
    }:
        return [
            row
            for row in rows
            if row["dataset"] == stratum
        ]

    raise ValueError(
        f"unexpected stratum: {stratum}"
    )


def finite_macro(rows, value_key):
    return ADAPTER.finite_subject_macro(
        rows,
        value_key=value_key,
    )


def point_detail_record(
    *,
    rows,
    stratum,
    identity,
):
    selected = select_stratum(
        rows,
        stratum,
    )

    expected_n = EXPECTED_STRATUM_N[
        stratum
    ]

    if len(selected) != expected_n:
        raise ValueError(
            f"unexpected subject count for {stratum}: "
            f"{len(selected)}"
        )

    c0 = finite_macro(
        selected,
        "C0_subject_value",
    )

    cs = finite_macro(
        selected,
        "CS_subject_value",
    )

    degradation = finite_macro(
        selected,
        "paired_degradation",
    )

    return {
        **identity,
        "stratum":
            stratum,

        "C0_point_estimate":
            c0["point_estimate"],

        "C0_eligible_subject_count":
            c0["eligible_subject_count"],

        "CS_point_estimate":
            cs["point_estimate"],

        "CS_eligible_subject_count":
            cs["eligible_subject_count"],

        "paired_degradation_point":
            degradation["point_estimate"],

        "paired_degradation_eligible_subject_count":
            degradation[
                "eligible_subject_count"
            ],

        "total_subject_count":
            expected_n,

        "inferential_ci_applied":
            False,
    }


def family_bootstrap_task(task):
    (
        identity,
        rows,
        denominator_record,
        stratum,
        replicates,
    ) = task

    selected = select_stratum(
        rows,
        stratum,
    )

    expected_n = EXPECTED_STRATUM_N[
        stratum
    ]

    if len(selected) != expected_n:
        raise ValueError(
            f"unexpected subject count for "
            f"{identity} {stratum}: "
            f"{len(selected)}"
        )

    c0 = finite_macro(
        selected,
        "C0_subject_value",
    )

    cs = finite_macro(
        selected,
        "CS_subject_value",
    )

    seed_parts = (
        "outer-report-v1",
        "family_severity_macro",
        identity["model_variant"],
        identity["operating_point"],
        identity["fault_family"],
        identity["severity_level"],
        identity["metric"],
        stratum,
    )

    bootstrap = ADAPTER.inferential_subject_bootstrap(
        selected,
        value_key="paired_degradation",
        stratum=stratum,
        replicates=replicates,
        seed_parts=seed_parts,
    )

    direction = REPORT.classify_direction(
        point=bootstrap[
            "point_estimate"
        ],
        ci95_low=bootstrap[
            "ci95_low"
        ],
        ci95_high=bootstrap[
            "ci95_high"
        ],
        coverage_complete=bootstrap[
            "coverage_complete"
        ],
    )

    out = {
        **identity,
        "detail_type":
            "family_severity_macro",

        "stratum":
            stratum,

        "C0_point_estimate":
            c0["point_estimate"],

        "C0_eligible_subject_count":
            c0["eligible_subject_count"],

        "CS_point_estimate":
            cs["point_estimate"],

        "CS_eligible_subject_count":
            cs["eligible_subject_count"],

        "paired_degradation_point":
            bootstrap[
                "point_estimate"
            ],

        "paired_degradation_ci95_low":
            bootstrap[
                "ci95_low"
            ],

        "paired_degradation_ci95_high":
            bootstrap[
                "ci95_high"
            ],

        "direction_status":
            direction,

        "eligible_subject_count":
            bootstrap[
                "eligible_subject_count"
            ],

        "total_subject_count":
            expected_n,

        "coverage_complete":
            bootstrap[
                "coverage_complete"
            ],

        "bootstrap_replicates":
            bootstrap[
                "bootstrap_replicates"
            ],

        "bootstrap_seed":
            bootstrap[
                "seed"
            ],

        "bootstrap_method":
            (
                "qualified_stratified_subject_bootstrap"
                if stratum
                == "overall_61_subject"
                else
                "qualified_one_stratum_subject_bootstrap_specialization"
            ),

        "source_denominators":
            denominator_record,
    }

    return out


def aggregate_denominator_records(
    rows,
    *,
    stratum,
):
    selected = select_stratum(
        rows,
        stratum,
    )

    expected_n = EXPECTED_STRATUM_N[
        stratum
    ]

    if len(selected) != expected_n:
        raise ValueError(
            f"denominator subject count mismatch: "
            f"{stratum} {len(selected)}"
        )

    result = {
        "interpretation":
            "noninferential_repeated_source_condition_totals",

        "subject_count":
            expected_n,

        "C0": {
            "source_row_count": 0,
            **{
                field: 0.0
                if field == "activity_seconds"
                else 0
                for field in DENOMINATOR_FIELDS
            },
        },

        "CS": {
            "source_row_count": 0,
            **{
                field: 0.0
                if field == "activity_seconds"
                else 0
                for field in DENOMINATOR_FIELDS
            },
        },
    }

    for row in selected:
        for regime in (
            "C0",
            "CS",
        ):
            src = row[regime]

            result[regime][
                "source_row_count"
            ] += int(
                src["source_row_count"]
            )

            sums = src[
                "noninferential_repeated_source_row_sums"
            ]

            for field in DENOMINATOR_FIELDS:
                if field == "activity_seconds":
                    result[regime][field] += float(
                        sums[field]
                    )
                else:
                    result[regime][field] += int(
                        sums[field]
                    )

    return result


def clean_reference_records(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["model_variant"],
            row["operating_point"],
            row["metric"],
        )

        groups[key].append(row)

    output = []

    for key, group in sorted(
        groups.items()
    ):
        model, op, metric = key

        for stratum in STRATA:
            selected = select_stratum(
                group,
                stratum,
            )

            if len(selected) != \
               EXPECTED_STRATUM_N[stratum]:
                raise ValueError(
                    "clean reference subject "
                    "coverage mismatch"
                )

            x = ADAPTER.finite_subject_macro(
                selected,
                value_key="C0_subject_value",
            )

            output.append({
                "detail_type":
                    "clean_reference",

                "model_variant":
                    model,

                "operating_point":
                    op,

                "metric":
                    metric,

                "stratum":
                    stratum,

                "C0_point_estimate":
                    x["point_estimate"],

                "eligible_subject_count":
                    x["eligible_subject_count"],

                "total_subject_count":
                    EXPECTED_STRATUM_N[
                        stratum
                    ],

                "coverage_complete":
                    x["coverage_complete"],
            })

    return output


def descriptive_detail_records(
    rows,
    *,
    identity_fields,
    detail_type,
):
    groups = defaultdict(list)

    for row in rows:
        key = tuple(
            row[field]
            for field in identity_fields
        )

        groups[key].append(row)

    output = []

    for key, group in sorted(
        groups.items()
    ):
        identity = dict(
            zip(
                identity_fields,
                key,
            )
        )

        identity[
            "detail_type"
        ] = detail_type

        for stratum in STRATA:
            output.append(
                point_detail_record(
                    rows=group,
                    stratum=stratum,
                    identity=identity,
                )
            )

    return output


def timing_descriptive_records(rows):
    groups = defaultdict(list)

    identity_fields = (
        "model_variant",
        "operating_point",
        "fault_family",
        "severity_level",
        "timing_summary_field",
    )

    for row in rows:
        key = tuple(
            row[field]
            for field in identity_fields
        )

        groups[key].append(row)

    output = []

    for key, group in sorted(
        groups.items()
    ):
        identity = dict(
            zip(
                identity_fields,
                key,
            )
        )

        for stratum in STRATA:
            selected = select_stratum(
                group,
                stratum,
            )

            if len(selected) != \
               EXPECTED_STRATUM_N[stratum]:
                raise ValueError(
                    "timing descriptive subject "
                    "coverage mismatch"
                )

            c0 = ADAPTER.finite_subject_macro(
                selected,
                value_key="C0_subject_value",
            )

            cs = ADAPTER.finite_subject_macro(
                selected,
                value_key="CS_subject_value",
            )

            output.append({
                **identity,

                "detail_type":
                    "timing_descriptive",

                "stratum":
                    stratum,

                "C0_point_estimate":
                    c0[
                        "point_estimate"
                    ],

                "C0_eligible_subject_count":
                    c0[
                        "eligible_subject_count"
                    ],

                "CS_point_estimate":
                    cs[
                        "point_estimate"
                    ],

                "CS_eligible_subject_count":
                    cs[
                        "eligible_subject_count"
                    ],

                "total_subject_count":
                    EXPECTED_STRATUM_N[
                        stratum
                    ],

                "inferential_ci_applied":
                    False,
            })

    return output


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

    parser.add_argument(
        "--recompute-partial",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = json.load(
        open(args.config)
    )

    subject_root = Path(
        cfg["subject_aggregation_root"]
    )

    output = Path(
        args.output
    )

    temp = Path(
        str(output)
        + ".tmp"
    )

    success = (
        output
        / "_SUCCESS.json"
    )

    if success.is_file():
        print(
            "OUTER_REPORTING_REUSED="
            + str(output)
        )
        return

    if output.exists() or temp.exists():
        if not args.recompute_partial:
            raise RuntimeError(
                "partial output exists; "
                "rerun with --recompute-partial"
            )

        if output.exists():
            shutil.rmtree(
                output
            )

        if temp.exists():
            shutil.rmtree(
                temp
            )

    temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    subject_success = json.loads(
        (
            subject_root
            / "_SUCCESS.json"
        ).read_text()
    )

    if subject_success["status"] != "PASS":
        raise RuntimeError(
            "subject aggregation input not PASS"
        )

    if subject_success[
        "cross_subject_bootstrap_executed"
    ] is not False:
        raise RuntimeError(
            "unexpected prior cross-subject bootstrap"
        )

    clean = load_jsonl(
        subject_root
        / "clean_subject_metrics.jsonl"
    )

    family = load_jsonl(
        subject_root
        / "subject_family_severity_macro.jsonl"
    )

    variants = load_jsonl(
        subject_root
        / "subject_individual_fault_variant.jsonl"
    )

    seeds = load_jsonl(
        subject_root
        / "subject_checkpoint_seed_specific.jsonl"
    )

    timing = load_jsonl(
        subject_root
        / "subject_timing_descriptive_family_macro.jsonl"
    )

    denominators = load_jsonl(
        subject_root
        / "subject_denominator_coverage_family_macro.jsonl"
    )

    interpretability = json.loads(
        (
            subject_root
            / "execution_interpretability.json"
        ).read_text()
    )

    if interpretability["status"] != \
       "EXECUTION_INTERPRETABLE":
        raise RuntimeError(
            "subject aggregation execution "
            "is not interpretable"
        )

    clean_report = clean_reference_records(
        clean
    )

    family_groups = defaultdict(list)

    family_identity_fields = (
        "model_variant",
        "operating_point",
        "fault_family",
        "severity_level",
        "metric",
    )

    for row in family:
        key = tuple(
            row[field]
            for field in family_identity_fields
        )

        family_groups[
            key
        ].append(row)

    denom_groups = defaultdict(list)

    denom_identity_fields = (
        "model_variant",
        "operating_point",
        "fault_family",
        "severity_level",
    )

    for row in denominators:
        key = tuple(
            row[field]
            for field in denom_identity_fields
        )

        denom_groups[
            key
        ].append(row)

    tasks = []

    bootstrap_replicates = int(
        cfg["bootstrap_replicates"]
    )

    for key, group in sorted(
        family_groups.items()
    ):
        identity = dict(
            zip(
                family_identity_fields,
                key,
            )
        )

        denom_key = (
            identity["model_variant"],
            identity["operating_point"],
            identity["fault_family"],
            identity["severity_level"],
        )

        denom_rows = denom_groups[
            denom_key
        ]

        if len(denom_rows) != 61:
            raise RuntimeError(
                f"denominator coverage mismatch: "
                f"{denom_key}"
            )

        for stratum in STRATA:
            denominator_record = \
                aggregate_denominator_records(
                    denom_rows,
                    stratum=stratum,
                )

            tasks.append(
                (
                    identity,
                    group,
                    denominator_record,
                    stratum,
                    bootstrap_replicates,
                )
            )

    expected_family_tasks = (
        2
        * 3
        * 12
        * 3
        * 7
        * 3
    )

    if len(tasks) != \
       expected_family_tasks:
        raise RuntimeError(
            f"family inference task count "
            f"{len(tasks)} != "
            f"{expected_family_tasks}"
        )

    workers = int(
        cfg["bootstrap_workers"]
    )

    context = mp.get_context(
        "fork"
    )

    with context.Pool(
        processes=workers
    ) as pool:
        family_report = list(
            pool.imap(
                family_bootstrap_task,
                tasks,
                chunksize=1,
            )
        )

    variant_report = descriptive_detail_records(
        variants,
        identity_fields=(
            "model_variant",
            "operating_point",
            "fault_family",
            "severity_level",
            "variant_id",
            "metric",
        ),
        detail_type=
            "individual_fault_variant",
    )

    seed_report = descriptive_detail_records(
        seeds,
        identity_fields=(
            "model_variant",
            "operating_point",
            "fault_family",
            "severity_level",
            "checkpoint_seed",
            "metric",
        ),
        detail_type=
            "checkpoint_seed_specific",
    )

    timing_report = timing_descriptive_records(
        timing
    )

    counts = {
        "clean_reference_summary":
            write_jsonl(
                temp
                / "clean_reference_summary.jsonl",
                clean_report,
            ),

        "family_severity_report":
            write_jsonl(
                temp
                / "family_severity_report.jsonl",
                family_report,
            ),

        "individual_variant_detail":
            write_jsonl(
                temp
                / "individual_variant_detail.jsonl",
                variant_report,
            ),

        "checkpoint_seed_detail":
            write_jsonl(
                temp
                / "checkpoint_seed_detail.jsonl",
                seed_report,
            ),

        "timing_descriptive_summary":
            write_jsonl(
                temp
                / "timing_descriptive_summary.jsonl",
                timing_report,
            ),
    }

    expected_counts = {
        "clean_reference_summary":
            2 * 3 * 7 * 3,

        "family_severity_report":
            2 * 3 * 12 * 3 * 7 * 3,

        "individual_variant_detail":
            26082,

        "checkpoint_seed_detail":
            2 * 3 * 12 * 3 * 3 * 7 * 3,

        "timing_descriptive_summary":
            2 * 3 * 12 * 3 * 3 * 3,
    }

    if counts != expected_counts:
        raise RuntimeError(
            "report record count mismatch\n"
            f"observed={counts}\n"
            f"expected={expected_counts}"
        )

    direction_counts = Counter(
        row["direction_status"]
        for row in family_report
    )

    incomplete_family_records = sum(
        not bool(
            row["coverage_complete"]
        )
        for row in family_report
    )

    coverage = {
        "schema_version":
            "phase4h_sensor_fi_outer_reporting_v1_coverage",

        "status":
            "PASS",

        "subject_aggregation_success_sha256":
            sha256(
                subject_root
                / "_SUCCESS.json"
            ),

        "subject_aggregation_artifact_manifest_sha256":
            sha256(
                subject_root
                / "artifact_manifest.json"
            ),

        "execution_interpretability":
            interpretability[
                "status"
            ],

        "bootstrap_replicates":
            bootstrap_replicates,

        "bootstrap_workers":
            workers,

        "required_strata":
            list(STRATA),

        "record_counts":
            counts,

        "direction_status_counts":
            dict(
                sorted(
                    direction_counts.items()
                )
            ),

        "incomplete_family_inferential_records":
            incomplete_family_records,

        "global_robustness_label_generated":
            False,

        "result_dependent_selection":
            False,

        "outer_result_based_retuning":
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

    files = [
        "clean_reference_summary.jsonl",
        "family_severity_report.jsonl",
        "individual_variant_detail.jsonl",
        "checkpoint_seed_detail.jsonl",
        "timing_descriptive_summary.jsonl",
        "coverage.json",
    ]

    manifest = {
        "schema_version":
            "phase4h_sensor_fi_outer_reporting_v1_artifact_manifest",

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
            for name in files
        },

        "scientific_boundary": {
            "outer_metric_values_read":
                True,

            "outer_metric_values_printed_by_runner":
                False,

            "outer_result_based_retuning":
                False,

            "outer_result_based_selection":
                False,

            "family_severity_inference":
                True,

            "individual_variant_inference":
                False,

            "checkpoint_seed_inference":
                False,

            "binary_global_robustness_label":
                False,

            "OnField_used":
                False,
        },
    }

    (
        temp
        / "artifact_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    success_record = {
        "schema_version":
            "phase4h_sensor_fi_outer_reporting_v1_success",

        "status":
            "PASS",

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

        "performance_values_printed_by_runner":
            False,

        "family_severity_bootstrap_complete":
            True,

        "global_robustness_label_generated":
            False,
    }

    (
        temp
        / "_SUCCESS.json"
    ).write_text(
        json.dumps(
            success_record,
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
        "OUTER_REPORTING_STATUS=PASS"
    )

    print(
        "EXECUTION_INTERPRETABILITY="
        + interpretability["status"]
    )

    for key in sorted(counts):
        print(
            f"{key.upper()}_ROWS="
            f"{counts[key]}"
        )

    print(
        "BOOTSTRAP_REPLICATES="
        + str(bootstrap_replicates)
    )

    print(
        "FAMILY_INFERENTIAL_RECORDS="
        + str(len(family_report))
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
        "GLOBAL_ROBUSTNESS_LABEL_GENERATED=False"
    )


if __name__ == "__main__":
    main()
