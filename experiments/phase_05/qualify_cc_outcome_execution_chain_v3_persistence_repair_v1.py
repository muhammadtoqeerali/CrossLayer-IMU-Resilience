from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import cc_outcome_analyzer_v2 as analyzer_v2
import cc_outcome_analyzer_v3 as analyzer
import cc_outcome_io_runner_v3 as io
import cc_outcome_executor_v3 as executor


CANONICAL_PERSISTENT = (
    "persistent_from_onset_until_trial_end"
)

NONCANONICAL_PERSISTENT = (
    "persistent_from_onset_to_trial_end"
)


def write_json(
    path,
    value,
):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path,
    rows,
):
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


def artifact(
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

    binding = json.loads(
        Path(
            args.config
        ).read_text()
    )

    checks = {}

    assert binding[
        "scientific_protocol_changed"
    ] is False

    assert binding[
        "canonical_persistence_contract"
    ][
        "persistent"
    ] == CANONICAL_PERSISTENT

    assert binding[
        "canonical_persistence_contract"
    ][
        "noncanonical_alias_authorized"
    ] is False

    assert (
        io.sha256_file(
            analyzer.__file__
        )
        == binding[
            "implementation_lineage"
        ][
            "v3_analyzer"
        ][
            "sha256"
        ]
    )

    assert (
        io.sha256_file(
            io.__file__
        )
        == binding[
            "implementation_lineage"
        ][
            "v3_io_runner"
        ][
            "sha256"
        ]
    )

    assert (
        io.sha256_file(
            executor.__file__
        )
        == binding[
            "implementation_lineage"
        ][
            "v3_executor"
        ][
            "sha256"
        ]
    )

    checks[
        "exact_v3_module_hash_binding"
    ] = True

    # V2 must demonstrate the exact diagnosed compatibility failure.
    try:
        analyzer_v2.reconstruct_fault_scenario(
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                {
                    "outer_instance_id":
                        "diagnostic",

                    "execution_window_index":
                        1,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [
                            0.4,
                            0.6,
                        ],
                },
                {
                    "outer_instance_id":
                        "diagnostic",

                    "execution_window_index":
                        2,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [
                            0.4,
                            0.6,
                        ],
                },
            ],
            outer_instance_id="diagnostic",
            persistence=CANONICAL_PERSISTENT,
        )
    except analyzer_v2.AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "V2 unexpectedly accepted canonical persistent vocabulary"
        )

    checks[
        "v2_failure_mode_reproduced"
    ] = True

    # V3 accepts only the canonical frozen spelling.
    canonical = analyzer.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
            0.3,
        ],
        [
            {
                "outer_instance_id":
                    "canonical",

                "execution_window_index":
                    1,

                "onset_or_inference_index":
                    1,

                "faulted_softmax_values":
                    [
                        0.4,
                        0.6,
                    ],
            },
            {
                "outer_instance_id":
                    "canonical",

                "execution_window_index":
                    2,

                "onset_or_inference_index":
                    1,

                "faulted_softmax_values":
                    [
                        0.4,
                        0.6,
                    ],
            },
        ],
        outer_instance_id="canonical",
        persistence=CANONICAL_PERSISTENT,
    )

    assert np.allclose(
        canonical,
        np.asarray(
            [
                0.1,
                0.6,
                0.6,
            ]
        ),
    )

    checks[
        "canonical_persistent_exact_suffix_reconstruction"
    ] = True

    try:
        analyzer.reconstruct_fault_scenario(
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                {
                    "outer_instance_id":
                        "legacy",

                    "execution_window_index":
                        1,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [
                            0.4,
                            0.6,
                        ],
                },
                {
                    "outer_instance_id":
                        "legacy",

                    "execution_window_index":
                        2,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [
                            0.4,
                            0.6,
                        ],
                },
            ],
            outer_instance_id="legacy",
            persistence=NONCANONICAL_PERSISTENT,
        )
    except analyzer.AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "V3 accepted noncanonical persistence alias"
        )

    checks[
        "noncanonical_persistence_alias_rejected"
    ] = True

    # Transient path must remain exactly equivalent between V2 and V3.
    transient_records = [
        {
            "outer_instance_id":
                "transient",

            "parent": {
                "window_index":
                    1,
            },

            "faulted_softmax_values":
                [
                    0.25,
                    0.75,
                ],
        },
    ]

    v2_transient = analyzer_v2.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
            0.3,
        ],
        transient_records,
        outer_instance_id="transient",
        persistence="transient_one_inference",
    )

    v3_transient = analyzer.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
            0.3,
        ],
        transient_records,
        outer_instance_id="transient",
        persistence="transient_one_inference",
    )

    assert np.array_equal(
        v2_transient,
        v3_transient,
    )

    checks[
        "transient_v2_v3_behavior_equivalent"
    ] = True

    # Preserve R1 nonfinite compatibility under canonical persistent mode.
    nan_sequence = analyzer.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
            0.3,
        ],
        [
            {
                "outer_instance_id":
                    "nan-persistent",

                "execution_window_index":
                    1,

                "onset_or_inference_index":
                    1,

                "faulted_output_nonfinite":
                    True,

                "faulted_softmax_values":
                    [
                        {
                            "nonfinite":
                                "nan",
                        },
                        {
                            "nonfinite":
                                "nan",
                        },
                    ],
            },
            {
                "outer_instance_id":
                    "nan-persistent",

                "execution_window_index":
                    2,

                "onset_or_inference_index":
                    1,

                "faulted_output_nonfinite":
                    True,

                "faulted_softmax_values":
                    [
                        {
                            "nonfinite":
                                "nan",
                        },
                        {
                            "nonfinite":
                                "nan",
                        },
                    ],
            },
        ],
        outer_instance_id="nan-persistent",
        persistence=CANONICAL_PERSISTENT,
    )

    assert nan_sequence[
        1
    ] != nan_sequence[
        1
    ]

    assert nan_sequence[
        2
    ] != nan_sequence[
        2
    ]

    checks[
        "canonical_persistent_preserves_nonfinite_nan"
    ] = True

    protocol_path = binding[
        "phase5z_protocol"
    ][
        "path"
    ]

    protocol = json.loads(
        Path(
            protocol_path
        ).read_text()
    )

    gate_path = binding[
        "base_phase5ac_gate"
    ][
        "path"
    ]

    with tempfile.TemporaryDirectory(
        prefix="phase5af_r4_"
    ) as temp:
        root = Path(temp)

        outer = root / "outer"
        dataset = root / "dataset"

        outer.mkdir()
        dataset.mkdir()

        subject = 1
        fold = 1
        seed = 42

        for task, labels in (
            (
                1,
                [
                    "Activity",
                    "Activity",
                    "Activity",
                ],
            ),
            (
                2,
                [
                    "Activity",
                    "Falling",
                    "Falling",
                ],
            ),
        ):
            directory = (
                dataset
                / str(subject)
                / str(task)
                / "1"
            )

            directory.mkdir(
                parents=True
            )

            np.save(
                directory
                / "labels.npy",
                np.asarray(
                    labels,
                    dtype="<U8",
                ),
            )

        risk = root / "risk.csv"

        risk.write_text(
            (
                "subject_id,task_id,trial_id,"
                "fall_start_position,impact_position,"
                "sampling_rate_hz,event_id,dataset_id\n"
                "SYN_1,2,1,40,100,100,SYN_01_T02_R01,SYN\n"
            ),
            encoding="utf-8",
        )

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
            min(thresholds)
            - 0.10,
        )

        high = min(
            0.999,
            max(thresholds)
            + 0.05,
        )

        clean_cache_id = "r4-clean"

        clean_rows = []

        for task, probabilities in (
            (
                1,
                [
                    low,
                    low,
                    low,
                ],
            ),
            (
                2,
                [
                    low,
                    high,
                    high,
                ],
            ),
        ):
            for window_index, probability in enumerate(
                probabilities
            ):
                clean_rows.append({
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

                    "clean_softmax_values": [
                        1.0 - probability,
                        probability,
                    ],
                })

        transient_id = "r4-transient"
        persistent_id = "r4-persistent"

        transient_rows = [
            {
                "outer_instance_id":
                    "r4-transient-activity",

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

                "faulted_softmax_values": [
                    1.0 - high,
                    high,
                ],
            },
            {
                "outer_instance_id":
                    "r4-transient-fall",

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
                        1,
                },

                "faulted_output_nonfinite":
                    False,

                "faulted_softmax_values": [
                    1.0 - low,
                    low,
                ],
            },
        ]

        persistent_rows = []

        for oid, task in (
            (
                "r4-persistent-activity",
                1,
            ),
            (
                "r4-persistent-fall",
                2,
            ),
        ):
            for index in (
                1,
                2,
            ):
                persistent_rows.append({
                    "outer_instance_id":
                        oid,

                    "model_variant":
                        "fp32",

                    "checkpoint_seed":
                        seed,

                    "fold":
                        fold,

                    "persistence":
                        CANONICAL_PERSISTENT,

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
                        True,

                    "faulted_softmax_values": [
                        {
                            "nonfinite":
                                "nan",
                        },
                        {
                            "nonfinite":
                                "nan",
                        },
                    ],
                })

        plan = {
            "trial_inventory": [
                {
                    "canonical_subject":
                        "SYN_01",

                    "fold":
                        fold,

                    "subject":
                        subject,

                    "task":
                        1,

                    "trial":
                        1,

                    "window_count":
                        3,
                },
                {
                    "canonical_subject":
                        "SYN_01",

                    "fold":
                        fold,

                    "subject":
                        subject,

                    "task":
                        2,

                    "trial":
                        1,

                    "window_count":
                        3,
                },
            ],

            "clean_caches": [
                {
                    "clean_cache_id":
                        clean_cache_id,

                    "subject":
                        subject,

                    "fold":
                        fold,

                    "checkpoint_seed":
                        seed,

                    "model_variant":
                        "fp32",

                    "expected_clean_model_window_evaluations":
                        6,
                },
            ],

            "shards": [
                {
                    "shard_id":
                        transient_id,

                    "clean_cache_id":
                        clean_cache_id,

                    "subject":
                        subject,

                    "fold":
                        fold,

                    "checkpoint_seed":
                        seed,

                    "model_variant":
                        "fp32",

                    "persistence":
                        "transient_one_inference",

                    "expected_faulted_model_window_evaluations":
                        2,

                    "expected_outer_instance_ids":
                        2,
                },
                {
                    "shard_id":
                        persistent_id,

                    "clean_cache_id":
                        clean_cache_id,

                    "subject":
                        subject,

                    "fold":
                        fold,

                    "checkpoint_seed":
                        seed,

                    "model_variant":
                        "fp32",

                    "persistence":
                        CANONICAL_PERSISTENT,

                    "expected_faulted_model_window_evaluations":
                        4,

                    "expected_outer_instance_ids":
                        2,
                },
            ],
        }

        plan_path = root / "plan.json"

        write_json(
            plan_path,
            plan,
        )

        plan_sha = io.sha256_file(
            plan_path
        )

        synthetic_executor_sha = (
            "8" * 64
        )

        artifact(
            outer,
            artifact_id=clean_cache_id,
            filename="clean_outputs.jsonl",
            rows=clean_rows,
            plan_sha=plan_sha,
            executor_sha=synthetic_executor_sha,
        )

        artifact(
            outer,
            artifact_id=transient_id,
            filename="fault_outputs.jsonl",
            rows=transient_rows,
            plan_sha=plan_sha,
            executor_sha=synthetic_executor_sha,
        )

        artifact(
            outer,
            artifact_id=persistent_id,
            filename="fault_outputs.jsonl",
            rows=persistent_rows,
            plan_sha=plan_sha,
            executor_sha=synthetic_executor_sha,
        )

        results = []

        for shard_id in (
            transient_id,
            persistent_id,
        ):
            result = executor.run_shard_job({
                "qualification_mode":
                    True,

                "protocol_path":
                    protocol_path,

                "gate_path":
                    gate_path,

                "repair_binding_path":
                    args.config,

                "plan_path":
                    str(
                        plan_path
                    ),

                "outer_root":
                    str(
                        outer
                    ),

                "dataset_root":
                    str(
                        dataset
                    ),

                "risk_csv":
                    str(
                        risk
                    ),

                "shard_id":
                    shard_id,

                "artifact_verification": {
                    "clean": {
                        "mode":
                            "phase5r_success",

                        "expected_plan_sha256":
                            plan_sha,

                        "expected_executor_sha256":
                            synthetic_executor_sha,
                    },

                    "fault": {
                        "mode":
                            "phase5r_success",

                        "expected_plan_sha256":
                            plan_sha,

                        "expected_executor_sha256":
                            synthetic_executor_sha,
                    },
                },
            })

            assert result[
                "qualification_mode"
            ] is True

            assert result[
                "outer_instance_id_count"
            ] == 2

            assert len(
                result[
                    "strata"
                ]
            ) == 1

            results.append(
                result
            )

        transient_result = next(
            result
            for result in results
            if result[
                "persistence"
            ] == "transient_one_inference"
        )

        persistent_result = next(
            result
            for result in results
            if result[
                "persistence"
            ] == CANONICAL_PERSISTENT
        )

        assert persistent_result[
            "strata"
        ][0][
            "nonfinite_fault_record_count"
        ] == 4

        checks[
            "canonical_persistent_end_to_end_v3_chain"
        ] = True

        assert transient_result[
            "strata"
        ][0][
            "nonfinite_fault_record_count"
        ] == 0

        checks[
            "transient_end_to_end_v3_chain"
        ] = True

        for result in results:
            assert set(
                result[
                    "clean_unique_metrics"
                ]
            ) == {
                "balanced",
                "low_false_alarm",
                "timely_150ms",
            }

            for stratum in result[
                "strata"
            ]:
                assert set(
                    stratum[
                        "operating_points"
                    ]
                ) == {
                    "balanced",
                    "low_false_alarm",
                    "timely_150ms",
                }

                for op in stratum[
                    "operating_points"
                ].values():
                    assert (
                        "paired_clean_metrics"
                        in op
                    )

                    assert (
                        "faulted_metrics"
                        in op
                    )

                    assert (
                        "degradation"
                        in op
                    )

        checks[
            "all_three_frozen_operating_points"
        ] = True

        checks[
            "paired_clean_faulted_metrics_constructed"
        ] = True

    assert len(
        checks
    ) == 10

    assert all(
        checks.values()
    )

    result = {
        "schema_version":
            "phase5af_r4_canonical_persistence_execution_chain_qualification_result_v1",

        "phase":
            "5AF-R4",

        "status":
            "QUALIFIED_POST_BOUNDARY_CANONICAL_PERSISTENCE_V3_CHAIN",

        "scientific_protocol_changed":
            False,

        "canonical_persistence":
            CANONICAL_PERSISTENT,

        "noncanonical_alias_authorized":
            False,

        "v3_analyzer_sha256":
            io.sha256_file(
                analyzer.__file__
            ),

        "v3_io_runner_sha256":
            io.sha256_file(
                io.__file__
            ),

        "v3_executor_sha256":
            io.sha256_file(
                executor.__file__
            ),

        "qualification_check_count":
            len(
                checks
            ),

        "qualification_checks":
            checks,

        "qualification_boundary": {
            "synthetic_only":
                True,

            "phase5af_production_resumed":
                False,

            "preserved_completed_shard_deleted":
                False,

            "preserved_completed_shard_rerun":
                False,

            "accepted_threshold_applied":
                False,

            "accepted_CC_metric_computed":
                False,

            "scientific_protocol_changed":
                False,

            "CSC_generated":
                False,

            "OnField_used":
                False,

            "model_forward_executed":
                False,

            "new_fault_execution_executed":
                False,
        },

        "phase5af_resume_authorized_by_r4":
            False,

        "next_action":
            (
                "Freeze a new post-boundary continuation authorization "
                "binding the exact V3 chain and explicitly preserving "
                "the already-completed transient R3/V2 shard before "
                "production continuation."
            ),
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "PHASE5AF_R4_STATUS=",
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
        "V3_ANALYZER_SHA256=",
        result[
            "v3_analyzer_sha256"
        ],
        sep="",
    )

    print(
        "V3_IO_RUNNER_SHA256=",
        result[
            "v3_io_runner_sha256"
        ],
        sep="",
    )

    print(
        "V3_EXECUTOR_SHA256=",
        result[
            "v3_executor_sha256"
        ],
        sep="",
    )

    print(
        "PHASE5AF_PRODUCTION_RESUMED=False"
    )


if __name__ == "__main__":
    main()
