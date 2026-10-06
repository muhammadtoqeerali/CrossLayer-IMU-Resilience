from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

PHASE05 = (
    ROOT
    / "experiments/phase_05"
)

sys.path.insert(
    0,
    str(PHASE05),
)

from cc_outcome_io_runner_v1 import (  # noqa: E402
    IOErrorContractError,
    clean_sequence,
    event_row,
    fault_identity_groups,
    historical_window_ends,
    reconstruct_identity_sequence,
)


RUNNER = (
    ROOT
    / "experiments/phase_05/"
    "cc_outcome_io_runner_v1.py"
)

QUALIFIER = (
    ROOT
    / "experiments/phase_05/"
    "qualify_cc_outcome_io_runner_v1.py"
)

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ab_compute_fi_cc_io_runner_qualification_v1/qualification.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_qualification_result_is_complete():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "QUALIFIED_PROSPECTIVE_CC_IO_RUNNER_PRE_OUTCOME"
    )

    assert (
        x["qualification_check_count"]
        == 12
    )

    assert len(
        x[
            "qualification_checks"
        ]
    ) == 12

    assert all(
        x[
            "qualification_checks"
        ].values()
    )


def test_result_binds_current_runner_and_qualifier():
    x = load(
        RESULT
    )

    assert (
        x["runner_sha256"]
        == sha(RUNNER)
    )

    assert (
        x["qualifier_sha256"]
        == sha(QUALIFIER)
    )


def test_expected_named_qualification_checks():
    x = load(
        RESULT
    )

    assert set(
        x[
            "qualification_checks"
        ]
    ) == {
        "clean_sequence_join",
        "historical_fallback_truth_rule",
        "historical_window_end_reconstruction",
        "label_length_guard",
        "malformed_json_abort",
        "phase5m_frozen_hash_adapter",
        "phase5m_transient_adapter",
        "phase5r_clean_hash_before_parse",
        "phase5r_persistent_adapter",
        "risk_index_join",
        "stored_label_normalization",
        "tamper_abort_before_parse",
    }


def test_clean_sequence_exact_join():
    rows = [
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
                0.1,
                0.2,
                0.3,
            ]
        )
    ]

    output = clean_sequence(
        rows,
        subject=9,
        task=1,
        trial=1,
        expected_window_count=3,
    )

    assert np.allclose(
        output,
        np.asarray(
            [
                0.1,
                0.2,
                0.3,
            ]
        ),
    )


def test_missing_clean_window_aborts():
    with pytest.raises(
        IOErrorContractError
    ):
        clean_sequence(
            [
                {
                    "parent": {
                        "subject":
                            9,

                        "task":
                            1,

                        "trial":
                            1,

                        "window_index":
                            0,
                    },

                    "clean_softmax_values":
                        [
                            0.9,
                            0.1,
                        ],
                },
            ],
            subject=9,
            task=1,
            trial=1,
            expected_window_count=2,
        )


def test_duplicate_clean_window_aborts():
    row = {
        "parent": {
            "subject":
                9,

            "task":
                1,

            "trial":
                1,

            "window_index":
                0,
        },

        "clean_softmax_values":
            [
                0.9,
                0.1,
            ],
    }

    with pytest.raises(
        IOErrorContractError
    ):
        clean_sequence(
            [
                row,
                dict(row),
            ],
            subject=9,
            task=1,
            trial=1,
            expected_window_count=1,
        )


def test_fault_identity_grouping():
    groups = fault_identity_groups([
        {
            "outer_instance_id":
                "a",
        },
        {
            "outer_instance_id":
                "b",
        },
        {
            "outer_instance_id":
                "a",
        },
    ])

    assert set(
        groups
    ) == {
        "a",
        "b",
    }

    assert len(
        groups["a"]
    ) == 2


def test_fault_identity_missing_id_aborts():
    with pytest.raises(
        IOErrorContractError
    ):
        fault_identity_groups([
            {
                "outer_instance_id":
                    "",
            },
        ])


def test_phase5m_transient_adapter():
    output = reconstruct_identity_sequence(
        [
            0.1,
            0.2,
            0.3,
        ],
        [
            {
                "outer_instance_id":
                    "x",

                "parent": {
                    "window_index":
                        1,
                },

                "faulted_softmax_values":
                    [
                        0.01,
                        0.99,
                    ],
            },
        ],
        outer_instance_id="x",
        persistence="transient_one_inference",
    )

    assert np.allclose(
        output,
        np.asarray(
            [
                0.1,
                0.99,
                0.3,
            ]
        ),
    )


def test_phase5r_persistent_adapter():
    output = reconstruct_identity_sequence(
        [
            0.1,
            0.2,
            0.3,
        ],
        [
            {
                "outer_instance_id":
                    "p",

                "execution_window_index":
                    1,

                "onset_or_inference_index":
                    1,

                "faulted_softmax_values":
                    [
                        0.1,
                        0.9,
                    ],
            },
            {
                "outer_instance_id":
                    "p",

                "execution_window_index":
                    2,

                "onset_or_inference_index":
                    1,

                "faulted_softmax_values":
                    [
                        0.2,
                        0.8,
                    ],
            },
        ],
        outer_instance_id="p",
        persistence="persistent_from_onset_to_trial_end",
    )

    assert np.allclose(
        output,
        np.asarray(
            [
                0.1,
                0.9,
                0.8,
            ]
        ),
    )


def test_phase5r_persistent_gap_aborts():
    with pytest.raises(
        IOErrorContractError
    ):
        reconstruct_identity_sequence(
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                {
                    "outer_instance_id":
                        "p",

                    "execution_window_index":
                        1,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [
                            0.1,
                            0.9,
                        ],
                },
            ],
            outer_instance_id="p",
            persistence="persistent_from_onset_to_trial_end",
        )


def test_historical_window_ends():
    ends = historical_window_ends(
        [
            0,
            0,
            1,
            1,
        ],
        fall_start_position=50,
    )

    assert np.array_equal(
        ends,
        np.asarray(
            [
                30,
                45,
                80,
                95,
            ]
        ),
    )


def test_historical_window_order_violation_aborts():
    with pytest.raises(
        IOErrorContractError
    ):
        historical_window_ends(
            [
                0,
                1,
                0,
            ],
            fall_start_position=50,
        )


def test_event_row_risk_record_can_define_fall_truth():
    row = event_row(
        probabilities=[
            0.1,
            0.2,
        ],
        labels=[
            0,
            0,
        ],
        risk_row={
            "fall_start_position":
                100,

            "impact_position":
                160,

            "sampling_rate_hz":
                100,
        },
    )

    assert (
        row[
            "true_event"
        ]
        == "FALLING"
    )


def test_event_row_activity_without_risk():
    row = event_row(
        probabilities=[
            0.1,
            0.2,
        ],
        labels=[
            0,
            0,
        ],
        risk_row=None,
    )

    assert (
        row[
            "true_event"
        ]
        == "ACTIVITY"
    )


def test_event_row_probability_label_length_mismatch_aborts():
    with pytest.raises(
        IOErrorContractError
    ):
        event_row(
            probabilities=[
                0.1,
            ],
            labels=[
                0,
                0,
            ],
            risk_row=None,
        )


def test_boundary_remains_pre_outcome():
    x = load(
        RESULT
    )

    assert all(
        value is False
        for value in x[
            "scientific_boundary"
        ].values()
    )

    assert x[
        "implementation_boundary"
    ][
        "prospective_IO_runner_implemented"
    ] is True

    assert x[
        "implementation_boundary"
    ][
        "prospective_IO_runner_qualified"
    ] is True

    assert x[
        "implementation_boundary"
    ][
        "prospective_outer_ingestion_executed"
    ] is False

    assert x[
        "implementation_boundary"
    ][
        "prospective_CC_outcome_analysis_executed"
    ] is False


def test_synthetic_fixture_boundary():
    x = load(
        RESULT
    )

    boundary = x[
        "synthetic_fixture_boundary"
    ]

    assert boundary[
        "synthetic_filesystem_used"
    ] is True

    assert boundary[
        "synthetic_labels_npy_used"
    ] is True

    assert boundary[
        "synthetic_jsonl_used"
    ] is True

    assert boundary[
        "synthetic_risk_csv_used"
    ] is True

    assert boundary[
        "accepted_outer_result_root_used"
    ] is False

    assert boundary[
        "accepted_outer_prediction_payload_used"
    ] is False

    assert boundary[
        "accepted_outer_label_payload_used"
    ] is False


def test_runner_has_no_hardcoded_accepted_outer_root():
    source = RUNNER.read_text(
        encoding="utf-8"
    )

    assert (
        "phase5_outer_compute_fi_prospective_v1"
        not in source
    )
