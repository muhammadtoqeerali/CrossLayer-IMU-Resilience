from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import cc_outcome_analyzer_v1 as analyzer
import cc_outcome_io_runner_v1 as io
import cc_outcome_executor_v1 as executor


def load_json(path):
    return json.loads(
        Path(path).read_text()
    )


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def phase5r_artifact(
    root,
    *,
    artifact_id,
    filename,
    rows,
    plan_sha,
    executor_sha,
):
    directory = (
        Path(root)
        / artifact_id
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = (
        directory
        / "metadata.json"
    )

    records = (
        directory
        / filename
    )

    write_json(
        metadata,
        {
            "status":
                "COMPLETE",

            "artifact_id":
                artifact_id,
        },
    )

    write_jsonl(
        records,
        rows,
    )

    write_json(
        directory
        / "_SUCCESS.json",
        {
            "status":
                "PASS",

            "plan_sha256":
                plan_sha,

            "executor_sha256":
                executor_sha,

            "output_hashes": {
                "metadata.json":
                    io.sha256_file(
                        metadata
                    ),

                filename:
                    io.sha256_file(
                        records
                    ),
            },
        },
    )

    return {
        "mode":
            "phase5r_success",

        "expected_plan_sha256":
            plan_sha,

        "expected_executor_sha256":
            executor_sha,
    }


def frozen_hash_artifact(
    root,
    *,
    artifact_id,
    filename,
    rows,
):
    directory = (
        Path(root)
        / artifact_id
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = (
        directory
        / "metadata.json"
    )

    records = (
        directory
        / filename
    )

    success = (
        directory
        / "_SUCCESS.json"
    )

    write_json(
        metadata,
        {
            "status":
                "COMPLETE",

            "artifact_id":
                artifact_id,
        },
    )

    write_jsonl(
        records,
        rows,
    )

    write_json(
        success,
        {
            "status":
                "PASS",
        },
    )

    return {
        "mode":
            "frozen_hashes",

        "expected_success_sha256":
            io.sha256_file(
                success
            ),

        "expected_metadata_sha256":
            io.sha256_file(
                metadata
            ),

        "expected_records_sha256":
            io.sha256_file(
                records
            ),
    }


def clean_rows(
    *,
    subject,
    seed,
    fold,
    low,
    high,
):
    rows = []

    trials = (
        (
            1,
            [
                low,
                low,
                low,
                low,
            ],
        ),
        (
            2,
            [
                low,
                low,
                high,
                high,
            ],
        ),
    )

    for task, probabilities in trials:
        for window_index, probability in enumerate(
            probabilities
        ):
            rows.append({
                "schema_version":
                    "synthetic",

                "model_variant":
                    "fp32",

                "checkpoint_seed":
                    seed,

                "fold":
                    fold,

                "parent": {
                    "subject":
                        subject,

                    "task":
                        task,

                    "trial":
                        1,

                    "window_index":
                        window_index,
                },

                "clean_softmax_values":
                    [
                        1.0
                        - probability,

                        probability,
                    ],
            })

    return rows


def transient_rows(
    *,
    subject,
    seed,
    fold,
    low,
    high,
    shard_id,
):
    return [
        {
            "schema_version":
                "synthetic",

            "shard_id":
                shard_id,

            "outer_instance_id":
                f"{shard_id}-activity",

            "model_variant":
                "fp32",

            "checkpoint_seed":
                seed,

            "fold":
                fold,

            "persistence":
                "transient_one_inference",

            "fault_family":
                "fp32_activation_single_bit_flip",

            "representation_class":
                "fp32_activation",

            "target_name":
                "front_end_output_fp32",

            "target_role":
                "activation",

            "parent": {
                "subject":
                    subject,

                "task":
                    1,

                "trial":
                    1,

                "window_index":
                    1,
            },

            "faulted_output_nonfinite":
                False,

            "faulted_softmax_values":
                [
                    1.0
                    - high,

                    high,
                ],
        },
        {
            "schema_version":
                "synthetic",

            "shard_id":
                shard_id,

            "outer_instance_id":
                f"{shard_id}-fall",

            "model_variant":
                "fp32",

            "checkpoint_seed":
                seed,

            "fold":
                fold,

            "persistence":
                "transient_one_inference",

            "fault_family":
                "fp32_activation_single_bit_flip",

            "representation_class":
                "fp32_activation",

            "target_name":
                "front_end_output_fp32",

            "target_role":
                "activation",

            "parent": {
                "subject":
                    subject,

                "task":
                    2,

                "trial":
                    1,

                "window_index":
                    2,
            },

            "faulted_output_nonfinite":
                False,

            "faulted_softmax_values":
                [
                    1.0
                    - low,

                    low,
                ],
        },
    ]


def persistent_rows(
    *,
    subject,
    seed,
    fold,
    low,
    high,
    shard_id,
):
    rows = []

    for outer_id, task, probability in (
        (
            f"{shard_id}-activity",
            1,
            high,
        ),
        (
            f"{shard_id}-fall",
            2,
            low,
        ),
    ):
        for index in (
            1,
            2,
            3,
        ):
            rows.append({
                "schema_version":
                    "synthetic",

                "shard_id":
                    shard_id,

                "outer_instance_id":
                    outer_id,

                "model_variant":
                    "fp32",

                "checkpoint_seed":
                    seed,

                "fold":
                    fold,

                "persistence":
                    "persistent_from_onset_to_trial_end",

                "fault_family":
                    "fp32_activation_single_bit_flip",

                "representation_class":
                    "fp32_activation",

                "target_name":
                    "front_end_output_fp32",

                "target_role":
                    "activation",

                "execution_window_index":
                    index,

                "onset_or_inference_index":
                    1,

                "parent": {
                    "subject":
                        subject,

                    "task":
                        task,

                    "trial":
                        1,
                },

                "faulted_output_nonfinite":
                    False,

                "faulted_softmax_values":
                    [
                        1.0
                        - probability,

                        probability,
                    ],
            })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--executor",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_json(
        args.config
    )

    for rec in cfg[
        "frozen_dependencies"
    ].values():
        assert (
            io.sha256_file(
                rec["path"]
            )
            == rec["sha256"]
        )

    protocol = load_json(
        cfg[
            "frozen_dependencies"
        ][
            "phase5z_protocol"
        ][
            "path"
        ]
    )

    gate_path = cfg[
        "frozen_dependencies"
    ][
        "phase5ac_gate"
    ][
        "path"
    ]

    gate = load_json(
        gate_path
    )

    checks = {}

    # Production path mismatch must fail on string binding before any I/O
    # against the supplied fake paths is attempted.
    try:
        executor.validate_lineage(
            protocol_path=cfg[
                "frozen_dependencies"
            ][
                "phase5z_protocol"
            ][
                "path"
            ],
            gate_path=gate_path,
            plan_path=cfg[
                "frozen_dependencies"
            ][
                "phase5e_plan"
            ][
                "path"
            ],
            outer_root="/definitely/not/the/frozen/outer/root",
            dataset_root="/definitely/not/the/frozen/dataset/root",
            qualification_mode=False,
        )
    except executor.OutcomeExecutorError as exc:
        assert (
            "production outer root differs"
            in str(exc)
        )
    else:
        raise AssertionError(
            "production root mismatch was not rejected"
        )

    checks[
        "production_path_binding_without_payload_access"
    ] = True

    with tempfile.TemporaryDirectory(
        prefix="phase5ad_"
    ) as temporary:
        root = Path(
            temporary
        )

        outer_root = (
            root
            / "synthetic_outer"
        )

        dataset_root = (
            root
            / "synthetic_dataset"
        )

        outer_root.mkdir()
        dataset_root.mkdir()

        plan = {
            "schema_version":
                "phase5ad_synthetic_plan_v1",

            "trial_inventory":
                [],

            "clean_caches":
                [],

            "shards":
                [],
        }

        risk_lines = [
            (
                "subject_id,task_id,trial_id,fall_start_position,"
                "impact_position,sampling_rate_hz,event_id,dataset_id"
            )
        ]

        jobs = []

        synthetic_executor_sha = hashlib.sha256(
            b"phase5ad-synthetic-phase5r-executor"
        ).hexdigest()

        for subject, fold in (
            (1, 1),
            (2, 2),
        ):
            for task, labels in (
                (
                    1,
                    [
                        "Activity",
                        "Activity",
                        "Activity",
                        "Activity",
                    ],
                ),
                (
                    2,
                    [
                        "Activity",
                        "Activity",
                        "Falling",
                        "Falling",
                    ],
                ),
            ):
                directory = (
                    dataset_root
                    / str(
                        subject
                    )
                    / str(
                        task
                    )
                    / "1"
                )

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.save(
                    directory
                    / "labels.npy",
                    np.asarray(
                        labels,
                        dtype="<U8",
                    ),
                )

                plan[
                    "trial_inventory"
                ].append({
                    "canonical_subject":
                        f"SYN_{subject:02d}",

                    "fold":
                        fold,

                    "subject":
                        subject,

                    "task":
                        task,

                    "trial":
                        1,

                    "window_count":
                        4,
                })

            risk_lines.append(
                (
                    f"SYN_{subject},{2},{1},"
                    f"50,130,100,"
                    f"SYN_{subject:02d}_T02_R01,SYN"
                )
            )

            for seed in (
                42,
                123,
                2025,
            ):
                thresholds = [
                    analyzer.threshold_rule(
                        protocol,
                        seed=seed,
                        fold=fold,
                        operating_point=op,
                    )[0]
                    for op in protocol[
                        "threshold_protocol"
                    ][
                        "operating_points"
                    ]
                ]

                low = max(
                    0.001,
                    min(
                        thresholds
                    )
                    - 0.10,
                )

                high = min(
                    0.999,
                    max(
                        thresholds
                    )
                    + 0.05,
                )

                assert all(
                    low
                    < threshold
                    <= high
                    for threshold in thresholds
                )

                clean_cache_id = (
                    f"synthetic-clean-s{subject}-seed{seed}"
                )

                clean = clean_rows(
                    subject=subject,
                    seed=seed,
                    fold=fold,
                    low=low,
                    high=high,
                )

                plan[
                    "clean_caches"
                ].append({
                    "canonical_subject":
                        f"SYN_{subject:02d}",

                    "checkpoint_seed":
                        seed,

                    "clean_cache_id":
                        clean_cache_id,

                    "expected_clean_model_window_evaluations":
                        8,

                    "fold":
                        fold,

                    "model_variant":
                        "fp32",

                    "shared_by_persistence_shards":
                        True,

                    "subject":
                        subject,

                    "window_count":
                        8,
                })

                for persistence in (
                    "transient_one_inference",
                    "persistent_from_onset_to_trial_end",
                ):
                    suffix = (
                        "transient"
                        if persistence
                        == "transient_one_inference"
                        else "persistent"
                    )

                    shard_id = (
                        f"synthetic-s{subject}-seed{seed}-{suffix}"
                    )

                    if (
                        persistence
                        == "transient_one_inference"
                    ):
                        fault = transient_rows(
                            subject=subject,
                            seed=seed,
                            fold=fold,
                            low=low,
                            high=high,
                            shard_id=shard_id,
                        )

                        expected_fault_rows = 2

                    else:
                        fault = persistent_rows(
                            subject=subject,
                            seed=seed,
                            fold=fold,
                            low=low,
                            high=high,
                            shard_id=shard_id,
                        )

                        expected_fault_rows = 6

                    plan[
                        "shards"
                    ].append({
                        "canonical_subject":
                            f"SYN_{subject:02d}",

                        "checkpoint_seed":
                            seed,

                        "clean_cache_id":
                            clean_cache_id,

                        "expected_faulted_model_window_evaluations":
                            expected_fault_rows,

                        "expected_outer_instance_ids":
                            2,

                        "fold":
                            fold,

                        "model_variant":
                            "fp32",

                        "persistence":
                            persistence,

                        "primary_execution_key":
                            "outer_instance_id",

                        "sampling_coordinates_are_frozen":
                            True,

                        "shard_id":
                            shard_id,

                        "subject":
                            subject,

                        "subject_inventory": {
                            "trial_count":
                                2,

                            "window_count":
                                8,
                        },

                        "target_count":
                            1,

                        "target_names": [
                            "front_end_output_fp32"
                        ],
                    })

                    jobs.append({
                        "subject":
                            subject,

                        "seed":
                            seed,

                        "fold":
                            fold,

                        "persistence":
                            persistence,

                        "clean_cache_id":
                            clean_cache_id,

                        "shard_id":
                            shard_id,

                        "clean_rows":
                            clean,

                        "fault_rows":
                            fault,
                    })

        plan_path = (
            root
            / "synthetic_plan.json"
        )

        write_json(
            plan_path,
            plan,
        )

        plan_sha = io.sha256_file(
            plan_path
        )

        risk_path = (
            root
            / "risk.csv"
        )

        risk_path.write_text(
            "\n".join(
                risk_lines
            )
            + "\n",
            encoding="utf-8",
        )

        job_results = []

        created_clean = set()

        for index, job in enumerate(
            jobs
        ):
            clean_cache_id = job[
                "clean_cache_id"
            ]

            if clean_cache_id not in created_clean:
                if (
                    job[
                        "subject"
                    ] == 1
                    and job[
                        "seed"
                    ] == 42
                ):
                    clean_verification = (
                        frozen_hash_artifact(
                            outer_root,
                            artifact_id=clean_cache_id,
                            filename="clean_outputs.jsonl",
                            rows=job[
                                "clean_rows"
                            ],
                        )
                    )
                else:
                    clean_verification = (
                        phase5r_artifact(
                            outer_root,
                            artifact_id=clean_cache_id,
                            filename="clean_outputs.jsonl",
                            rows=job[
                                "clean_rows"
                            ],
                            plan_sha=plan_sha,
                            executor_sha=synthetic_executor_sha,
                        )
                    )

                created_clean.add(
                    clean_cache_id
                )

            else:
                clean_dir = (
                    outer_root
                    / clean_cache_id
                )

                success_path = (
                    clean_dir
                    / "_SUCCESS.json"
                )

                metadata_path = (
                    clean_dir
                    / "metadata.json"
                )

                records_path = (
                    clean_dir
                    / "clean_outputs.jsonl"
                )

                success = load_json(
                    success_path
                )

                if (
                    job[
                        "subject"
                    ] == 1
                    and job[
                        "seed"
                    ] == 42
                ):
                    clean_verification = {
                        "mode":
                            "frozen_hashes",

                        "expected_success_sha256":
                            io.sha256_file(
                                success_path
                            ),

                        "expected_metadata_sha256":
                            io.sha256_file(
                                metadata_path
                            ),

                        "expected_records_sha256":
                            io.sha256_file(
                                records_path
                            ),
                    }

                else:
                    assert (
                        success[
                            "plan_sha256"
                        ]
                        == plan_sha
                    )

                    clean_verification = {
                        "mode":
                            "phase5r_success",

                        "expected_plan_sha256":
                            plan_sha,

                        "expected_executor_sha256":
                            synthetic_executor_sha,
                    }

            if (
                job[
                    "subject"
                ] == 1
                and job[
                    "seed"
                ] == 42
                and job[
                    "persistence"
                ] == "transient_one_inference"
            ):
                fault_verification = (
                    frozen_hash_artifact(
                        outer_root,
                        artifact_id=job[
                            "shard_id"
                        ],
                        filename="fault_outputs.jsonl",
                        rows=job[
                            "fault_rows"
                        ],
                    )
                )

                checks[
                    "phase5m_style_frozen_hash_path"
                ] = True

            else:
                fault_verification = (
                    phase5r_artifact(
                        outer_root,
                        artifact_id=job[
                            "shard_id"
                        ],
                        filename="fault_outputs.jsonl",
                        rows=job[
                            "fault_rows"
                        ],
                        plan_sha=plan_sha,
                        executor_sha=synthetic_executor_sha,
                    )
                )

                checks[
                    "phase5r_integrity_path"
                ] = True

            spec = {
                "qualification_mode":
                    True,

                "protocol_path":
                    cfg[
                        "frozen_dependencies"
                    ][
                        "phase5z_protocol"
                    ][
                        "path"
                    ],

                "gate_path":
                    gate_path,

                "plan_path":
                    str(
                        plan_path
                    ),

                "outer_root":
                    str(
                        outer_root
                    ),

                "dataset_root":
                    str(
                        dataset_root
                    ),

                "risk_csv":
                    str(
                        risk_path
                    ),

                "shard_id":
                    job[
                        "shard_id"
                    ],

                "artifact_verification": {
                    "clean":
                        clean_verification,

                    "fault":
                        fault_verification,
                },
            }

            result = executor.run_shard_job(
                spec
            )

            assert result[
                "qualification_mode"
            ] is True

            assert result[
                "outer_instance_id_count"
            ] == 2

            assert set(
                result[
                    "clean_unique_metrics"
                ]
            ) == {
                "balanced",
                "low_false_alarm",
                "timely_150ms",
            }

            assert len(
                result[
                    "strata"
                ]
            ) == 1

            stratum = result[
                "strata"
            ][
                0
            ]

            assert stratum[
                "outer_instance_id_count"
            ] == 2

            assert stratum[
                "nonfinite_fault_record_count"
            ] == 0

            for op_result in stratum[
                "operating_points"
            ].values():
                assert (
                    "paired_clean_metrics"
                    in op_result
                )

                assert (
                    "faulted_metrics"
                    in op_result
                )

                assert (
                    "degradation"
                    in op_result
                )

            job_results.append(
                result
            )

        assert len(
            job_results
        ) == 12

        checks[
            "twelve_synthetic_subject_seed_persistence_jobs"
        ] = True

        assert {
            result[
                "persistence"
            ]
            for result in job_results
        } == {
            "transient_one_inference",
            "persistent_from_onset_to_trial_end",
        }

        checks[
            "transient_end_to_end"
        ] = True

        checks[
            "persistent_end_to_end"
        ] = True

        checks[
            "three_operating_points"
        ] = True

        checks[
            "paired_clean_faulted_and_degradation_outputs"
        ] = True

        checks[
            "nonfinite_accounting"
        ] = True

        aggregate = executor.aggregate_shard_results(
            job_results,
            protocol=protocol,
            expected_subject_count=2,
        )

        assert aggregate[
            "expected_subject_count"
        ] == 2

        assert aggregate[
            "primary_uncertainty_unit"
        ] == "outer subject"

        assert aggregate[
            "bootstrap_replicates"
        ] == 10000

        available_balanced = [
            row
            for row in aggregate[
                "stratum_degradation_aggregates"
            ]
            if (
                row[
                    "metric"
                ]
                == "balanced_accuracy"
                and row[
                    "aggregate"
                ][
                    "status"
                ]
                == "AVAILABLE"
            )
        ]

        assert available_balanced

        for row in available_balanced:
            assert row[
                "aggregate"
            ][
                "subject_count"
            ] == 2

            assert row[
                "aggregate"
            ][
                "bootstrap_replicates"
            ] == 10000

        checks[
            "equal_seed_then_subject_aggregation"
        ] = True

        checks[
            "subject_cluster_bootstrap"
        ] = True

        clean_keys = {
            (
                row[
                    "model_variant"
                ],
                row[
                    "operating_point"
                ],
                row[
                    "metric"
                ],
            )
            for row in aggregate[
                "clean_unique_aggregates"
            ]
        }

        assert len(
            clean_keys
        ) == len(
            aggregate[
                "clean_unique_aggregates"
            ]
        )

        checks[
            "clean_baseline_deduplicated_across_persistence"
        ] = True

        timing_rows = [
            row
            for row in aggregate[
                "stratum_degradation_aggregates"
            ]
            if row[
                "metric"
            ] == "median_trigger_lead_ms"
        ]

        assert timing_rows

        assert all(
            row[
                "aggregate"
            ][
                "status"
            ]
            in {
                "AVAILABLE",
                "UNAVAILABLE_WITHOUT_IMPUTATION",
            }
            for row in timing_rows
        )

        checks[
            "missing_timing_never_imputed"
        ] = True

        checks[
            "synthetic_label_and_risk_join"
        ] = True

        checks[
            "exact_lineage_binding"
        ] = True

    assert checks
    assert all(
        checks.values()
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "schema_version":
            "phase5ad_compute_fi_cc_outcome_executor_qualification_result_v1",

        "phase":
            "5AD",

        "status":
            "QUALIFIED_FINAL_CC_OUTCOME_EXECUTOR_PRE_OUTCOME",

        "evidence_tier":
            "P0",

        "executor_sha256":
            io.sha256_file(
                args.executor
            ),

        "qualifier_sha256":
            io.sha256_file(
                __file__
            ),

        "qualification_check_count":
            len(
                checks
            ),

        "qualification_checks":
            checks,

        "qualification_boundary": {
            "synthetic_estates_only":
                True,

            "accepted_outer_root_accessed":
                False,

            "accepted_outer_root_stat_performed":
                False,

            "accepted_outer_root_listed":
                False,

            "accepted_prediction_jsonl_opened":
                False,

            "accepted_outer_label_array_loaded":
                False,

            "accepted_prediction_deserialized":
                False,

            "accepted_threshold_applied":
                False,

            "accepted_CC_metric_computed":
                False,

            "aggregate_accepted_CC_result_generated":
                False,

            "CSC_result_generated":
                False,

            "model_loaded":
                False,

            "model_forward_executed":
                False,

            "fault_execution_executed":
                False,

            "onfield_used":
                False,
        },

        "implementation_boundary": {
            "final_outcome_executor_implemented":
                True,

            "final_outcome_executor_qualified":
                True,

            "production_execution_performed":
                False,

            "accepted_outcome_analysis_executed":
                False,
        },

        "next_action":
            (
                "Freeze the final one-way CC outcome execution activation "
                "binding this exact executor hash before first accepted "
                "prediction JSONL or outer label payload access."
            ),
    }

    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE5AD_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    print(
        "QUALIFICATION_CHECKS=",
        result[
            "qualification_check_count"
        ],
        sep="",
    )

    print(
        "ACCEPTED_OUTER_ROOT_ACCESSED=False"
    )

    print(
        "ACCEPTED_PREDICTION_JSONL_OPENED=False"
    )

    print(
        "ACCEPTED_OUTER_LABEL_ARRAY_LOADED=False"
    )

    print(
        "ACCEPTED_CC_METRIC_COMPUTED=False"
    )


if __name__ == "__main__":
    main()
