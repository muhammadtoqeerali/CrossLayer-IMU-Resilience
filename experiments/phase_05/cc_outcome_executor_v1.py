"""Final prospective compute-fault CC outcome executor.

The executor is path-parameterized. Importing it performs no filesystem I/O.

Production execution requires the exact Phase-5AC-bound outer root,
dataset root, Phase-5E plan, Phase-5Z protocol, Phase-5AA analyzer and
Phase-5AB I/O runner. Qualification mode is explicitly synthetic and cannot
authorize production payload access.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import cc_outcome_analyzer_v1 as analyzer
import cc_outcome_io_runner_v1 as io


class OutcomeExecutorError(RuntimeError):
    """Fatal end-to-end outcome executor contract violation."""


PRIMARY_METRICS = (
    "balanced_accuracy",
    "fall_recall",
    "false_triggers_per_activity_hour",
    "recall_by_150ms",
    "median_trigger_lead_ms",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def _module_sha(module) -> str:
    if module.__file__ is None:
        raise OutcomeExecutorError(
            "imported module has no source file"
        )

    return io.sha256_file(
        module.__file__
    )


def validate_lineage(
    *,
    protocol_path: str | Path,
    gate_path: str | Path,
    plan_path: str | Path,
    outer_root: str | Path,
    dataset_root: str | Path,
    qualification_mode: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    protocol_path = Path(protocol_path)
    gate_path = Path(gate_path)
    plan_path = Path(plan_path)

    protocol = _load_json(
        protocol_path
    )

    gate = _load_json(
        gate_path
    )

    plan = _load_json(
        plan_path
    )

    if (
        protocol["status"]
        != "FROZEN_PROSPECTIVE_CC_OUTCOME_ANALYSIS_PROTOCOL"
    ):
        raise OutcomeExecutorError(
            "Phase-5Z protocol is not frozen"
        )

    if (
        gate["status"]
        != "FROZEN_PRE_OUTCOME_CC_EXECUTION_GATE"
    ):
        raise OutcomeExecutorError(
            "Phase-5AC gate is not frozen"
        )

    frozen = gate[
        "frozen_dependencies"
    ]

    if (
        io.sha256_file(
            protocol_path
        )
        != frozen[
            "phase5z_protocol"
        ][
            "sha256"
        ]
    ):
        raise OutcomeExecutorError(
            "Phase-5Z protocol hash mismatch"
        )

    if (
        _module_sha(
            analyzer
        )
        != frozen[
            "phase5aa_analyzer"
        ][
            "sha256"
        ]
    ):
        raise OutcomeExecutorError(
            "Phase-5AA analyzer hash mismatch"
        )

    if (
        _module_sha(
            io
        )
        != frozen[
            "phase5ab_io_runner"
        ][
            "sha256"
        ]
    ):
        raise OutcomeExecutorError(
            "Phase-5AB I/O runner hash mismatch"
        )

    if qualification_mode:
        if str(
            outer_root
        ) == gate[
            "future_input_roots"
        ][
            "accepted_outer_result_root"
        ][
            "path"
        ]:
            raise OutcomeExecutorError(
                "qualification mode cannot use accepted outer root"
            )

        if str(
            dataset_root
        ) == gate[
            "future_input_roots"
        ][
            "outer_dataset_root"
        ][
            "path"
        ]:
            raise OutcomeExecutorError(
                "qualification mode cannot use accepted outer dataset root"
            )

    else:
        if (
            io.sha256_file(
                plan_path
            )
            != frozen[
                "phase5e_execution_plan"
            ][
                "sha256"
            ]
        ):
            raise OutcomeExecutorError(
                "production Phase-5E plan hash mismatch"
            )

        if str(
            outer_root
        ) != gate[
            "future_input_roots"
        ][
            "accepted_outer_result_root"
        ][
            "path"
        ]:
            raise OutcomeExecutorError(
                "production outer root differs from Phase-5AC gate"
            )

        if str(
            dataset_root
        ) != gate[
            "future_input_roots"
        ][
            "outer_dataset_root"
        ][
            "path"
        ]:
            raise OutcomeExecutorError(
                "production dataset root differs from Phase-5AC gate"
            )

    return (
        protocol,
        gate,
        plan,
    )


def _lookup_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    value: object,
    role: str,
) -> dict[str, Any]:
    matches = [
        dict(
            row
        )
        for row in rows
        if row.get(
            key
        )
        == value
    ]

    if len(
        matches
    ) != 1:
        raise OutcomeExecutorError(
            f"{role} lookup expected one row, found {len(matches)}"
        )

    return matches[0]


def _verify_artifact(
    directory: Path,
    *,
    record_filename: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    mode = verification.get(
        "mode"
    )

    if mode == "phase5r_success":
        return io.verify_phase5r_artifact(
            directory,
            record_filename=record_filename,
            expected_plan_sha256=str(
                verification[
                    "expected_plan_sha256"
                ]
            ),
            expected_executor_sha256=str(
                verification[
                    "expected_executor_sha256"
                ]
            ),
        )

    if mode == "frozen_hashes":
        return io.verify_frozen_hash_artifact(
            directory,
            record_filename=record_filename,
            expected_success_sha256=str(
                verification[
                    "expected_success_sha256"
                ]
            ),
            expected_metadata_sha256=str(
                verification[
                    "expected_metadata_sha256"
                ]
            ),
            expected_records_sha256=str(
                verification[
                    "expected_records_sha256"
                ]
            ),
        )

    raise OutcomeExecutorError(
        f"unknown artifact verification mode: {mode}"
    )


def _trial_inventory_for_subject(
    plan: Mapping[str, Any],
    *,
    subject: int,
    fold: int,
) -> list[dict[str, Any]]:
    rows = [
        dict(
            row
        )
        for row in plan[
            "trial_inventory"
        ]
        if (
            int(
                row["subject"]
            )
            == int(
                subject
            )
            and int(
                row["fold"]
            )
            == int(
                fold
            )
        )
    ]

    if not rows:
        raise OutcomeExecutorError(
            "subject has no trial inventory"
        )

    rows.sort(
        key=lambda row: (
            int(
                row["task"]
            ),
            int(
                row["trial"]
            ),
        )
    )

    keys = [
        (
            int(
                row["task"]
            ),
            int(
                row["trial"]
            ),
        )
        for row in rows
    ]

    if len(
        set(
            keys
        )
    ) != len(
        keys
    ):
        raise OutcomeExecutorError(
            "duplicate subject/task/trial inventory row"
        )

    return rows


def _index_clean_by_trial(
    clean_records: Sequence[Mapping[str, Any]],
    *,
    subject: int,
) -> dict[
    tuple[int, int],
    list[dict[str, Any]],
]:
    output: dict[
        tuple[int, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in clean_records:
        parent = record.get(
            "parent"
        )

        if not isinstance(
            parent,
            Mapping,
        ):
            raise OutcomeExecutorError(
                "clean record missing parent"
            )

        if int(
            parent[
                "subject"
            ]
        ) != int(
            subject
        ):
            raise OutcomeExecutorError(
                "clean cache contains a different subject"
            )

        key = (
            int(
                parent[
                    "task"
                ]
            ),
            int(
                parent[
                    "trial"
                ]
            ),
        )

        output[
            key
        ].append(
            dict(
                record
            )
        )

    return dict(
        output
    )


def _parent_trial_key(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    keys = set()

    for record in rows:
        parent = record.get(
            "parent"
        )

        if not isinstance(
            parent,
            Mapping,
        ):
            raise OutcomeExecutorError(
                "fault record missing parent"
            )

        keys.add(
            (
                int(
                    parent[
                        "subject"
                    ]
                ),
                int(
                    parent[
                        "task"
                    ]
                ),
                int(
                    parent[
                        "trial"
                    ]
                ),
            )
        )

    if len(
        keys
    ) != 1:
        raise OutcomeExecutorError(
            "one outer_instance_id spans multiple parent trials"
        )

    return next(
        iter(
            keys
        )
    )


def _stratum(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, str]:
    fields = (
        "fault_family",
        "representation_class",
        "target_name",
        "target_role",
    )

    values = []

    for field in fields:
        unique = {
            str(
                row[
                    field
                ]
            )
            for row in rows
        }

        if len(
            unique
        ) != 1:
            raise OutcomeExecutorError(
                f"outer_instance_id has inconsistent {field}"
            )

        values.append(
            next(
                iter(
                    unique
                )
            )
        )

    return tuple(
        values
    )


def _optional_degradation(
    metric_name: str,
    clean_value: object,
    fault_value: object,
) -> float | None:
    clean = float(
        clean_value
    )

    fault = float(
        fault_value
    )

    if (
        not math.isfinite(
            clean
        )
        or not math.isfinite(
            fault
        )
    ):
        return None

    return analyzer.paired_metric_degradation(
        metric_name,
        clean_value=clean,
        faulted_value=fault,
    )


def run_shard_job(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    qualification_mode = bool(
        spec.get(
            "qualification_mode",
            False,
        )
    )

    protocol, gate, plan = validate_lineage(
        protocol_path=spec[
            "protocol_path"
        ],
        gate_path=spec[
            "gate_path"
        ],
        plan_path=spec[
            "plan_path"
        ],
        outer_root=spec[
            "outer_root"
        ],
        dataset_root=spec[
            "dataset_root"
        ],
        qualification_mode=qualification_mode,
    )

    shard = _lookup_unique(
        plan[
            "shards"
        ],
        key="shard_id",
        value=spec[
            "shard_id"
        ],
        role="fault shard",
    )

    clean_cache = _lookup_unique(
        plan[
            "clean_caches"
        ],
        key="clean_cache_id",
        value=shard[
            "clean_cache_id"
        ],
        role="clean cache",
    )

    subject = int(
        shard[
            "subject"
        ]
    )

    fold = int(
        shard[
            "fold"
        ]
    )

    seed = int(
        shard[
            "checkpoint_seed"
        ]
    )

    model_variant = str(
        shard[
            "model_variant"
        ]
    )

    persistence = str(
        shard[
            "persistence"
        ]
    )

    if (
        int(
            clean_cache[
                "subject"
            ]
        )
        != subject
        or int(
            clean_cache[
                "fold"
            ]
        )
        != fold
        or int(
            clean_cache[
                "checkpoint_seed"
            ]
        )
        != seed
        or str(
            clean_cache[
                "model_variant"
            ]
        )
        != model_variant
    ):
        raise OutcomeExecutorError(
            "clean-cache/shard identity mismatch"
        )

    outer_root = Path(
        spec[
            "outer_root"
        ]
    )

    clean_dir = (
        outer_root
        / str(
            clean_cache[
                "clean_cache_id"
            ]
        )
    )

    fault_dir = (
        outer_root
        / str(
            shard[
                "shard_id"
            ]
        )
    )

    verification = spec[
        "artifact_verification"
    ]

    verified_clean = _verify_artifact(
        clean_dir,
        record_filename="clean_outputs.jsonl",
        verification=verification[
            "clean"
        ],
    )

    verified_fault = _verify_artifact(
        fault_dir,
        record_filename="fault_outputs.jsonl",
        verification=verification[
            "fault"
        ],
    )

    clean_records = io.read_verified_jsonl(
        verified_clean
    )

    fault_records = io.read_verified_jsonl(
        verified_fault
    )

    expected_clean = int(
        clean_cache[
            "expected_clean_model_window_evaluations"
        ]
    )

    expected_fault_rows = int(
        shard[
            "expected_faulted_model_window_evaluations"
        ]
    )

    expected_identities = int(
        shard[
            "expected_outer_instance_ids"
        ]
    )

    if len(
        clean_records
    ) != expected_clean:
        raise OutcomeExecutorError(
            "clean record count differs from plan"
        )

    if len(
        fault_records
    ) != expected_fault_rows:
        raise OutcomeExecutorError(
            "fault record count differs from plan"
        )

    inventory = _trial_inventory_for_subject(
        plan,
        subject=subject,
        fold=fold,
    )

    clean_index = _index_clean_by_trial(
        clean_records,
        subject=subject,
    )

    risk_index = io.load_risk_index_csv(
        spec[
            "risk_csv"
        ]
    )

    trial_context: dict[
        tuple[int, int],
        dict[str, Any],
    ] = {}

    clean_unique_rows = []

    for trial_row in inventory:
        (
            trial_subject,
            task,
            trial,
            window_count,
        ) = io.validate_plan_trial(
            trial_row
        )

        if trial_subject != subject:
            raise OutcomeExecutorError(
                "trial inventory subject mismatch"
            )

        key = (
            task,
            trial,
        )

        if key not in clean_index:
            raise OutcomeExecutorError(
                "missing clean records for trial"
            )

        probabilities = io.clean_sequence(
            clean_index[
                key
            ],
            subject=subject,
            task=task,
            trial=trial,
            expected_window_count=window_count,
        )

        labels = io.load_trial_labels(
            spec[
                "dataset_root"
            ],
            subject=subject,
            task=task,
            trial=trial,
            expected_window_count=window_count,
        )

        risk_row = risk_index.get(
            (
                subject,
                task,
                trial,
            )
        )

        clean_event = io.event_row(
            probabilities=probabilities,
            labels=labels,
            risk_row=risk_row,
        )

        trial_context[
            key
        ] = {
            "clean_probabilities":
                probabilities,

            "labels":
                labels,

            "risk_row":
                risk_row,

            "clean_event":
                clean_event,
        }

        clean_unique_rows.append(
            clean_event
        )

    if set(
        clean_index
    ) != set(
        trial_context
    ):
        raise OutcomeExecutorError(
            "clean cache contains trial keys outside frozen inventory"
        )

    fault_groups = io.fault_identity_groups(
        fault_records
    )

    if len(
        fault_groups
    ) != expected_identities:
        raise OutcomeExecutorError(
            "outer_instance_id count differs from plan"
        )

    scenario_groups: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}

    seen_identity_count = 0

    for outer_instance_id, identity_rows in fault_groups.items():
        (
            parent_subject,
            task,
            trial,
        ) = _parent_trial_key(
            identity_rows
        )

        if parent_subject != subject:
            raise OutcomeExecutorError(
                "fault identity subject mismatch"
            )

        key = (
            task,
            trial,
        )

        if key not in trial_context:
            raise OutcomeExecutorError(
                "fault identity references trial outside frozen inventory"
            )

        for row in identity_rows:
            if (
                int(
                    row[
                        "checkpoint_seed"
                    ]
                )
                != seed
                or int(
                    row[
                        "fold"
                    ]
                )
                != fold
                or str(
                    row[
                        "model_variant"
                    ]
                )
                != model_variant
                or str(
                    row[
                        "persistence"
                    ]
                )
                != persistence
            ):
                raise OutcomeExecutorError(
                    "fault record does not match shard identity"
                )

        stratum = _stratum(
            identity_rows
        )

        context = trial_context[
            key
        ]

        fault_probabilities = io.reconstruct_identity_sequence(
            context[
                "clean_probabilities"
            ],
            identity_rows,
            outer_instance_id=outer_instance_id,
            persistence=persistence,
        )

        fault_event = io.event_row(
            probabilities=fault_probabilities,
            labels=context[
                "labels"
            ],
            risk_row=context[
                "risk_row"
            ],
        )

        bucket = scenario_groups.setdefault(
            stratum,
            {
                "fault_rows":
                    [],

                "paired_clean_rows":
                    [],

                "outer_instance_ids":
                    [],

                "nonfinite_fault_record_count":
                    0,
            },
        )

        bucket[
            "fault_rows"
        ].append(
            fault_event
        )

        bucket[
            "paired_clean_rows"
        ].append(
            context[
                "clean_event"
            ]
        )

        bucket[
            "outer_instance_ids"
        ].append(
            outer_instance_id
        )

        bucket[
            "nonfinite_fault_record_count"
        ] += analyzer.count_nonfinite_fault_records(
            identity_rows
        )

        seen_identity_count += 1

    if seen_identity_count != expected_identities:
        raise OutcomeExecutorError(
            "fault identity accounting mismatch"
        )

    clean_unique_metrics = {}

    for operating_point in protocol[
        "threshold_protocol"
    ][
        "operating_points"
    ]:
        threshold, consecutive = analyzer.threshold_rule(
            protocol,
            seed=seed,
            fold=fold,
            operating_point=operating_point,
        )

        clean_unique_metrics[
            operating_point
        ] = analyzer.event_metrics(
            clean_unique_rows,
            threshold,
            consecutive,
        )

    stratum_results = []

    for stratum in sorted(
        scenario_groups
    ):
        (
            fault_family,
            representation_class,
            target_name,
            target_role,
        ) = stratum

        bucket = scenario_groups[
            stratum
        ]

        operating_point_results = {}

        for operating_point in protocol[
            "threshold_protocol"
        ][
            "operating_points"
        ]:
            threshold, consecutive = analyzer.threshold_rule(
                protocol,
                seed=seed,
                fold=fold,
                operating_point=operating_point,
            )

            paired_clean = analyzer.event_metrics(
                bucket[
                    "paired_clean_rows"
                ],
                threshold,
                consecutive,
            )

            faulted = analyzer.event_metrics(
                bucket[
                    "fault_rows"
                ],
                threshold,
                consecutive,
            )

            degradation = {}

            for metric_name in protocol[
                "metric_contract"
            ][
                "raw_metrics"
            ]:
                if metric_name in {
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                }:
                    continue

                degradation[
                    metric_name
                ] = _optional_degradation(
                    metric_name,
                    paired_clean[
                        metric_name
                    ],
                    faulted[
                        metric_name
                    ],
                )

            operating_point_results[
                operating_point
            ] = {
                "threshold":
                    threshold,

                "required_consecutive":
                    consecutive,

                "paired_clean_metrics":
                    paired_clean,

                "faulted_metrics":
                    faulted,

                "degradation":
                    degradation,
            }

        stratum_results.append({
            "fault_family":
                fault_family,

            "representation_class":
                representation_class,

            "target_name":
                target_name,

            "target_role":
                target_role,

            "outer_instance_id_count":
                len(
                    bucket[
                        "outer_instance_ids"
                    ]
                ),

            "nonfinite_fault_record_count":
                int(
                    bucket[
                        "nonfinite_fault_record_count"
                    ]
                ),

            "operating_points":
                operating_point_results,
        })

    return {
        "schema_version":
            "phase5ad_cc_shard_outcome_v1",

        "qualification_mode":
            qualification_mode,

        "shard_id":
            str(
                shard[
                    "shard_id"
                ]
            ),

        "clean_cache_id":
            str(
                shard[
                    "clean_cache_id"
                ]
            ),

        "subject":
            subject,

        "fold":
            fold,

        "checkpoint_seed":
            seed,

        "model_variant":
            model_variant,

        "persistence":
            persistence,

        "trial_count":
            len(
                inventory
            ),

        "clean_record_count":
            len(
                clean_records
            ),

        "fault_record_count":
            len(
                fault_records
            ),

        "outer_instance_id_count":
            len(
                fault_groups
            ),

        "clean_unique_metrics":
            clean_unique_metrics,

        "strata":
            stratum_results,
    }


def _metric_key(
    result: Mapping[str, Any],
    stratum: Mapping[str, Any],
    operating_point: str,
    metric: str,
) -> tuple[str, ...]:
    return (
        str(
            result[
                "model_variant"
            ]
        ),
        str(
            result[
                "persistence"
            ]
        ),
        str(
            operating_point
        ),
        str(
            stratum[
                "fault_family"
            ]
        ),
        str(
            stratum[
                "representation_class"
            ]
        ),
        str(
            stratum[
                "target_name"
            ]
        ),
        str(
            stratum[
                "target_role"
            ]
        ),
        str(
            metric
        ),
    )


def aggregate_shard_results(
    shard_results: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    expected_subject_count: int,
) -> dict[str, Any]:
    if not shard_results:
        raise OutcomeExecutorError(
            "no shard results"
        )

    degradation_rows: dict[
        tuple[str, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)

    clean_values: dict[
        tuple[str, str, int, int, str],
        float,
    ] = {}

    for result in shard_results:
        subject = int(
            result[
                "subject"
            ]
        )

        seed = int(
            result[
                "checkpoint_seed"
            ]
        )

        model_variant = str(
            result[
                "model_variant"
            ]
        )

        for operating_point, metrics in result[
            "clean_unique_metrics"
        ].items():
            for metric_name in PRIMARY_METRICS:
                value = float(
                    metrics[
                        metric_name
                    ]
                )

                key = (
                    model_variant,
                    operating_point,
                    subject,
                    seed,
                    metric_name,
                )

                if key in clean_values:
                    prior = clean_values[
                        key
                    ]

                    both_nan = (
                        math.isnan(
                            prior
                        )
                        and math.isnan(
                            value
                        )
                    )

                    if (
                        not both_nan
                        and not math.isclose(
                            prior,
                            value,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise OutcomeExecutorError(
                            "duplicate clean baseline differs between persistence shards"
                        )
                else:
                    clean_values[
                        key
                    ] = value

        for stratum in result[
            "strata"
        ]:
            for operating_point, op_result in stratum[
                "operating_points"
            ].items():
                for metric_name in PRIMARY_METRICS:
                    key = _metric_key(
                        result,
                        stratum,
                        operating_point,
                        metric_name,
                    )

                    degradation_rows[
                        key
                    ].append({
                        "subject":
                            subject,

                        "seed":
                            seed,

                        "value":
                            op_result[
                                "degradation"
                            ][
                                metric_name
                            ],
                    })

    stratum_aggregates = []

    bootstrap_replicates = int(
        protocol[
            "uncertainty_contract"
        ][
            "bootstrap_replicates"
        ]
    )

    confidence_level = float(
        protocol[
            "uncertainty_contract"
        ][
            "confidence_level"
        ]
    )

    rng_seed = int(
        protocol[
            "uncertainty_contract"
        ][
            "rng_seed"
        ]
    )

    for key in sorted(
        degradation_rows
    ):
        rows = degradation_rows[
            key
        ]

        subjects = {
            int(
                row[
                    "subject"
                ]
            )
            for row in rows
        }

        if len(
            subjects
        ) != int(
            expected_subject_count
        ):
            raise OutcomeExecutorError(
                "reporting stratum does not cover expected subject count"
            )

        unavailable = any(
            row[
                "value"
            ] is None
            or not math.isfinite(
                float(
                    row[
                        "value"
                    ]
                )
            )
            for row in rows
        )

        if unavailable:
            aggregate = {
                "status":
                    "UNAVAILABLE_WITHOUT_IMPUTATION",

                "estimate":
                    None,

                "lower":
                    None,

                "upper":
                    None,

                "subject_count":
                    len(
                        subjects
                    ),

                "missing_or_nonfinite_subject_seed_values":
                    sum(
                        int(
                            row[
                                "value"
                            ] is None
                            or not math.isfinite(
                                float(
                                    row[
                                        "value"
                                    ]
                                )
                            )
                        )
                        for row in rows
                    ),
            }

        else:
            subject_values = analyzer.seed_equal_subject_values(
                rows
            )

            if len(
                subject_values
            ) != int(
                expected_subject_count
            ):
                raise OutcomeExecutorError(
                    "seed-equal aggregation changed subject count"
                )

            ci = analyzer.subject_cluster_percentile_ci(
                subject_values,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                rng_seed=rng_seed,
            )

            aggregate = {
                "status":
                    "AVAILABLE",

                **ci,

                "missing_or_nonfinite_subject_seed_values":
                    0,
            }

        stratum_aggregates.append({
            "model_variant":
                key[0],

            "persistence":
                key[1],

            "operating_point":
                key[2],

            "fault_family":
                key[3],

            "representation_class":
                key[4],

            "target_name":
                key[5],

            "target_role":
                key[6],

            "metric":
                key[7],

            "aggregate":
                aggregate,
        })

    clean_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for (
        model_variant,
        operating_point,
        subject,
        seed,
        metric_name,
    ), value in clean_values.items():
        clean_groups[
            (
                model_variant,
                operating_point,
                metric_name,
            )
        ].append({
            "subject":
                subject,

            "seed":
                seed,

            "value":
                value,
        })

    clean_aggregates = []

    for key in sorted(
        clean_groups
    ):
        rows = clean_groups[
            key
        ]

        subjects = {
            int(
                row[
                    "subject"
                ]
            )
            for row in rows
        }

        if len(
            subjects
        ) != int(
            expected_subject_count
        ):
            raise OutcomeExecutorError(
                "clean baseline does not cover expected subjects"
            )

        unavailable = any(
            not math.isfinite(
                float(
                    row[
                        "value"
                    ]
                )
            )
            for row in rows
        )

        if unavailable:
            aggregate = {
                "status":
                    "UNAVAILABLE_WITHOUT_IMPUTATION",

                "estimate":
                    None,

                "lower":
                    None,

                "upper":
                    None,

                "subject_count":
                    len(
                        subjects
                    ),
            }
        else:
            subject_values = analyzer.seed_equal_subject_values(
                rows
            )

            ci = analyzer.subject_cluster_percentile_ci(
                subject_values,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                rng_seed=rng_seed,
            )

            aggregate = {
                "status":
                    "AVAILABLE",

                **ci,
            }

        clean_aggregates.append({
            "model_variant":
                key[0],

            "operating_point":
                key[1],

            "metric":
                key[2],

            "aggregate":
                aggregate,
        })

    return {
        "schema_version":
            "phase5ad_cc_aggregate_outcome_v1",

        "expected_subject_count":
            int(
                expected_subject_count
            ),

        "primary_uncertainty_unit":
            "outer subject",

        "bootstrap_replicates":
            bootstrap_replicates,

        "confidence_level":
            confidence_level,

        "rng_seed":
            rng_seed,

        "clean_unique_aggregates":
            clean_aggregates,

        "stratum_degradation_aggregates":
            stratum_aggregates,
    }


def write_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
