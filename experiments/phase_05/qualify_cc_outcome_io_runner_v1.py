from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from cc_outcome_io_runner_v1 import (
    IOErrorContractError,
    clean_sequence,
    event_row,
    fault_identity_groups,
    load_risk_index_csv,
    load_trial_labels,
    read_verified_jsonl,
    reconstruct_identity_sequence,
    sha256_file,
    verify_frozen_hash_artifact,
    verify_phase5r_artifact,
)


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


def make_phase5r_artifact(
    root,
    *,
    artifact_id,
    record_filename,
    rows,
    plan_sha,
    executor_sha,
):
    directory = (
        Path(root)
        / artifact_id
    )

    directory.mkdir(
        parents=True
    )

    metadata_path = (
        directory
        / "metadata.json"
    )

    records_path = (
        directory
        / record_filename
    )

    write_json(
        metadata_path,
        {
            "status":
                "COMPLETE",

            "artifact_id":
                artifact_id,
        },
    )

    write_jsonl(
        records_path,
        rows,
    )

    success = {
        "status":
            "PASS",

        "artifact_kind":
            (
                "clean_cache"
                if record_filename
                == "clean_outputs.jsonl"
                else "fault_shard"
            ),

        "plan_sha256":
            plan_sha,

        "executor_sha256":
            executor_sha,

        "output_hashes": {
            "metadata.json":
                sha256_file(
                    metadata_path
                ),

            record_filename:
                sha256_file(
                    records_path
                ),
        },
    }

    write_json(
        directory
        / "_SUCCESS.json",
        success,
    )

    return directory


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--runner",
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
            sha256_file(
                rec["path"]
            )
            == rec["sha256"]
        )

    checks = {}

    synthetic_plan_sha = hashlib.sha256(
        b"phase5ab-synthetic-plan"
    ).hexdigest()

    synthetic_executor_sha = hashlib.sha256(
        b"phase5ab-synthetic-executor"
    ).hexdigest()

    with tempfile.TemporaryDirectory(
        prefix="phase5ab_"
    ) as temporary:
        root = Path(
            temporary
        )

        dataset_root = (
            root
            / "dataset"
        )

        trial_dir = (
            dataset_root
            / "9"
            / "1"
            / "1"
        )

        trial_dir.mkdir(
            parents=True
        )

        np.save(
            trial_dir
            / "labels.npy",
            np.asarray(
                [
                    "Activity",
                    "Activity",
                    "Falling",
                    "Falling",
                ],
                dtype=object,
            ),
            allow_pickle=True,
        )

        labels = load_trial_labels(
            dataset_root,
            subject=9,
            task=1,
            trial=1,
            expected_window_count=4,
        )

        assert np.array_equal(
            labels,
            np.asarray(
                [0, 0, 1, 1],
                dtype=np.int8,
            ),
        )

        checks[
            "stored_label_normalization"
        ] = True

        try:
            load_trial_labels(
                dataset_root,
                subject=9,
                task=1,
                trial=1,
                expected_window_count=5,
            )
        except IOErrorContractError:
            pass
        else:
            raise AssertionError(
                "label-length mismatch not rejected"
            )

        checks[
            "label_length_guard"
        ] = True

        clean_rows = [
            {
                "parent": {
                    "subject":
                        9,

                    "task":
                        1,

                    "trial":
                        1,

                    "window_index":
                        index,
                },

                "clean_softmax_values":
                    [
                        1.0 - probability,
                        probability,
                    ],
            }
            for index, probability in enumerate(
                [
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                ]
            )
        ]

        clean_dir = make_phase5r_artifact(
            root,
            artifact_id="synthetic-clean",
            record_filename="clean_outputs.jsonl",
            rows=clean_rows,
            plan_sha=synthetic_plan_sha,
            executor_sha=synthetic_executor_sha,
        )

        verified_clean = (
            verify_phase5r_artifact(
                clean_dir,
                record_filename="clean_outputs.jsonl",
                expected_plan_sha256=synthetic_plan_sha,
                expected_executor_sha256=synthetic_executor_sha,
            )
        )

        parsed_clean = (
            read_verified_jsonl(
                verified_clean
            )
        )

        clean = clean_sequence(
            parsed_clean,
            subject=9,
            task=1,
            trial=1,
            expected_window_count=4,
        )

        assert np.allclose(
            clean,
            np.asarray(
                [
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                ]
            ),
        )

        checks[
            "phase5r_clean_hash_before_parse"
        ] = True

        checks[
            "clean_sequence_join"
        ] = True

        # Accepted-canary style exact externally frozen hashes.
        canary_dir = (
            root
            / "synthetic-canary"
        )

        canary_dir.mkdir()

        write_json(
            canary_dir
            / "metadata.json",
            {
                "status":
                    "COMPLETE",

                "artifact_id":
                    "synthetic-canary",
            },
        )

        transient_rows = [
            {
                "outer_instance_id":
                    "oid-transient",

                "parent": {
                    "subject":
                        9,

                    "task":
                        1,

                    "trial":
                        1,

                    "window_index":
                        2,
                },

                "faulted_softmax_values":
                    [
                        0.01,
                        0.99,
                    ],
            }
        ]

        write_jsonl(
            canary_dir
            / "fault_outputs.jsonl",
            transient_rows,
        )

        write_json(
            canary_dir
            / "_SUCCESS.json",
            {
                "status":
                    "PASS",
            },
        )

        verified_canary = (
            verify_frozen_hash_artifact(
                canary_dir,
                record_filename="fault_outputs.jsonl",
                expected_success_sha256=sha256_file(
                    canary_dir
                    / "_SUCCESS.json"
                ),
                expected_metadata_sha256=sha256_file(
                    canary_dir
                    / "metadata.json"
                ),
                expected_records_sha256=sha256_file(
                    canary_dir
                    / "fault_outputs.jsonl"
                ),
            )
        )

        parsed_transient = (
            read_verified_jsonl(
                verified_canary
            )
        )

        transient_groups = (
            fault_identity_groups(
                parsed_transient
            )
        )

        transient = (
            reconstruct_identity_sequence(
                clean,
                transient_groups[
                    "oid-transient"
                ],
                outer_instance_id="oid-transient",
                persistence="transient_one_inference",
            )
        )

        assert np.allclose(
            transient,
            np.asarray(
                [
                    0.10,
                    0.20,
                    0.99,
                    0.40,
                ]
            ),
        )

        checks[
            "phase5m_frozen_hash_adapter"
        ] = True

        checks[
            "phase5m_transient_adapter"
        ] = True

        persistent_rows = [
            {
                "outer_instance_id":
                    "oid-persistent",

                "execution_window_index":
                    index,

                "onset_or_inference_index":
                    1,

                "parent": {
                    "subject":
                        9,

                    "task":
                        1,

                    "trial":
                        1,

                    "window_index":
                        None,
                },

                "faulted_softmax_values":
                    [
                        1.0 - probability,
                        probability,
                    ],
            }
            for index, probability in (
                (1, 0.91),
                (2, 0.92),
                (3, 0.93),
            )
        ]

        persistent_dir = (
            make_phase5r_artifact(
                root,
                artifact_id="synthetic-persistent",
                record_filename="fault_outputs.jsonl",
                rows=persistent_rows,
                plan_sha=synthetic_plan_sha,
                executor_sha=synthetic_executor_sha,
            )
        )

        verified_persistent = (
            verify_phase5r_artifact(
                persistent_dir,
                record_filename="fault_outputs.jsonl",
                expected_plan_sha256=synthetic_plan_sha,
                expected_executor_sha256=synthetic_executor_sha,
            )
        )

        parsed_persistent = (
            read_verified_jsonl(
                verified_persistent
            )
        )

        persistent_groups = (
            fault_identity_groups(
                parsed_persistent
            )
        )

        persistent = (
            reconstruct_identity_sequence(
                clean,
                persistent_groups[
                    "oid-persistent"
                ],
                outer_instance_id="oid-persistent",
                persistence="persistent_from_onset_to_trial_end",
            )
        )

        assert np.allclose(
            persistent,
            np.asarray(
                [
                    0.10,
                    0.91,
                    0.92,
                    0.93,
                ]
            ),
        )

        checks[
            "phase5r_persistent_adapter"
        ] = True

        # Risk/timing join and historical fallback truth.
        risk_path = (
            root
            / "risk.csv"
        )

        risk_path.write_text(
            "subject_id,task_id,trial_id,fall_start_position,"
            "impact_position,sampling_rate_hz,event_id,dataset_id\n"
            "KFALL_106,27,5,100,160,100,"
            "KFALL_106_T27_R05,KFALL\n",
            encoding="utf-8",
        )

        risk = load_risk_index_csv(
            risk_path
        )

        assert (
            106,
            27,
            5,
        ) in risk

        checks[
            "risk_index_join"
        ] = True

        fallback_row = event_row(
            probabilities=[
                0.1,
                0.2,
            ],
            labels=[
                0,
                0,
            ],
            risk_row=risk[
                (
                    106,
                    27,
                    5,
                )
            ],
        )

        assert (
            fallback_row[
                "true_event"
            ]
            == "FALLING"
        )

        checks[
            "historical_fallback_truth_rule"
        ] = True

        falling_row = event_row(
            probabilities=[
                0.1,
                0.2,
                0.9,
                0.9,
            ],
            labels=[
                0,
                0,
                1,
                1,
            ],
            risk_row={
                "fall_start_position":
                    50,

                "impact_position":
                    130,

                "sampling_rate_hz":
                    100.0,
            },
        )

        assert np.array_equal(
            falling_row[
                "window_ends"
            ],
            np.asarray(
                [
                    30,
                    45,
                    80,
                    95,
                ]
            ),
        )

        checks[
            "historical_window_end_reconstruction"
        ] = True

        # Hash mismatch must abort before JSONL parse.
        tampered = (
            root
            / "tampered"
        )

        tampered.mkdir()

        write_json(
            tampered
            / "metadata.json",
            {
                "status":
                    "COMPLETE",
            },
        )

        (
            tampered
            / "clean_outputs.jsonl"
        ).write_text(
            "{this is deliberately malformed JSON\n",
            encoding="utf-8",
        )

        write_json(
            tampered
            / "_SUCCESS.json",
            {
                "status":
                    "PASS",

                "plan_sha256":
                    synthetic_plan_sha,

                "executor_sha256":
                    synthetic_executor_sha,

                "output_hashes": {
                    "metadata.json":
                        sha256_file(
                            tampered
                            / "metadata.json"
                        ),

                    "clean_outputs.jsonl":
                        "0" * 64,
                },
            },
        )

        try:
            verify_phase5r_artifact(
                tampered,
                record_filename="clean_outputs.jsonl",
                expected_plan_sha256=synthetic_plan_sha,
                expected_executor_sha256=synthetic_executor_sha,
            )
        except IOErrorContractError as exc:
            assert (
                "record-file hash mismatch"
                in str(exc)
            )
        else:
            raise AssertionError(
                "tampered JSONL was not rejected by hash verification"
            )

        checks[
            "tamper_abort_before_parse"
        ] = True

        # Valid hash but malformed JSON must fail at parsing.
        malformed = (
            root
            / "malformed"
        )

        malformed.mkdir()

        write_json(
            malformed
            / "metadata.json",
            {
                "status":
                    "COMPLETE",
            },
        )

        (
            malformed
            / "fault_outputs.jsonl"
        ).write_text(
            "{not valid json}\n",
            encoding="utf-8",
        )

        write_json(
            malformed
            / "_SUCCESS.json",
            {
                "status":
                    "PASS",

                "plan_sha256":
                    synthetic_plan_sha,

                "executor_sha256":
                    synthetic_executor_sha,

                "output_hashes": {
                    "metadata.json":
                        sha256_file(
                            malformed
                            / "metadata.json"
                        ),

                    "fault_outputs.jsonl":
                        sha256_file(
                            malformed
                            / "fault_outputs.jsonl"
                        ),
                },
            },
        )

        verified_malformed = (
            verify_phase5r_artifact(
                malformed,
                record_filename="fault_outputs.jsonl",
                expected_plan_sha256=synthetic_plan_sha,
                expected_executor_sha256=synthetic_executor_sha,
            )
        )

        try:
            read_verified_jsonl(
                verified_malformed
            )
        except IOErrorContractError as exc:
            assert (
                "malformed JSONL"
                in str(exc)
            )
        else:
            raise AssertionError(
                "malformed JSONL was not rejected"
            )

        checks[
            "malformed_json_abort"
        ] = True

    assert len(checks) == 12
    assert all(
        checks.values()
    )

    result = {
        "schema_version":
            "phase5ab_compute_fi_cc_io_runner_qualification_result_v1",

        "phase":
            "5AB",

        "status":
            "QUALIFIED_PROSPECTIVE_CC_IO_RUNNER_PRE_OUTCOME",

        "evidence_tier":
            "P0",

        "runner_sha256":
            sha256_file(
                args.runner
            ),

        "qualifier_sha256":
            sha256_file(
                __file__
            ),

        "phase5z_protocol_sha256":
            cfg[
                "frozen_dependencies"
            ][
                "phase5z_protocol"
            ][
                "sha256"
            ],

        "phase5aa_analyzer_sha256":
            cfg[
                "frozen_dependencies"
            ][
                "phase5aa_analyzer"
            ][
                "sha256"
            ],

        "qualification_check_count":
            len(checks),

        "qualification_checks":
            checks,

        "synthetic_fixture_boundary": {
            "synthetic_filesystem_used":
                True,

            "synthetic_labels_npy_used":
                True,

            "synthetic_jsonl_used":
                True,

            "synthetic_risk_csv_used":
                True,

            "accepted_outer_result_root_used":
                False,

            "accepted_outer_prediction_payload_used":
                False,

            "accepted_outer_label_payload_used":
                False,
        },

        "scientific_boundary": {
            "accepted_outer_result_root_accessed":
                False,

            "accepted_clean_outputs_jsonl_opened":
                False,

            "accepted_fault_outputs_jsonl_opened":
                False,

            "accepted_outer_label_array_loaded":
                False,

            "outer_prediction_deserialized":
                False,

            "outer_threshold_applied":
                False,

            "outer_CC_metric_computed":
                False,

            "aggregate_CC_result_generated":
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
            "prospective_IO_runner_implemented":
                True,

            "prospective_IO_runner_qualified":
                True,

            "prospective_outer_ingestion_executed":
                False,

            "prospective_CC_outcome_analysis_executed":
                False,
        },

        "next_action":
            (
                "Freeze a one-way prospective CC outcome execution gate "
                "binding Phase-5Z, Phase-5AA, and Phase-5AB before the "
                "accepted outer prediction JSONL or labels are opened."
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
        "PHASE5AB_STATUS=",
        result["status"],
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
        "ACCEPTED_OUTER_RESULT_ROOT_USED=False"
    )

    print(
        "ACCEPTED_OUTER_PREDICTION_PAYLOAD_USED=False"
    )

    print(
        "ACCEPTED_OUTER_LABEL_PAYLOAD_USED=False"
    )


if __name__ == "__main__":
    main()
