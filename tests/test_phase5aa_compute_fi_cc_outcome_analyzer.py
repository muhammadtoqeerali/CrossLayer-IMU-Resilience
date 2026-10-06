from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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

from cc_outcome_analyzer_v1 import (  # noqa: E402
    AnalysisContractError,
    clean_falling_probability,
    count_nonfinite_fault_records,
    equal_subject_mean,
    event_metrics,
    faulted_falling_probability,
    first_valid_trigger,
    normalise_activity_label,
    paired_clean_counterfactual,
    paired_metric_degradation,
    reconstruct_fault_scenario,
    seed_equal_subject_values,
    subject_cluster_percentile_ci,
    threshold_index,
    threshold_rule,
    trigger_episodes,
)


ANALYZER = (
    ROOT
    / "experiments/phase_05/"
    "cc_outcome_analyzer_v1.py"
)

QUALIFIER = (
    ROOT
    / "experiments/phase_05/"
    "qualify_cc_outcome_analyzer_v1.py"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5aa_compute_fi_cc_analyzer_qualification_v1.json"
)

PHASE5Z = (
    ROOT
    / "configs/evaluation/"
    "phase5z_compute_fi_cc_outcome_analysis_protocol_v1.json"
)

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5aa_compute_fi_cc_analyzer_qualification_v1/qualification.json"
)

HISTORICAL = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/master_training/evaluation.py"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def historical():
    spec = importlib.util.spec_from_file_location(
        "phase5aa_test_historical",
        HISTORICAL,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_qualification_result():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "QUALIFIED_CC_ANALYZER_CORE_PRE_OUTCOME"
    )

    assert (
        x["qualification_check_count"]
        == 16
    )

    assert all(
        x[
            "qualification_checks"
        ].values()
    )

    assert all(
        x[
            "historical_source_equivalence"
        ].values()
    )


def test_result_binds_current_analyzer_and_qualifier():
    x = load(
        RESULT
    )

    assert (
        x["analyzer_sha256"]
        == sha(ANALYZER)
    )

    assert (
        x["qualifier_sha256"]
        == sha(QUALIFIER)
    )


def test_threshold_matrix_exact_45():
    protocol = load(
        PHASE5Z
    )

    index = threshold_index(
        protocol
    )

    assert len(index) == 45

    for key, row in index.items():
        threshold, consecutive = (
            threshold_rule(
                protocol,
                seed=key[0],
                fold=key[1],
                operating_point=key[2],
            )
        )

        assert (
            threshold
            == row["threshold"]
        )

        assert (
            consecutive
            == row[
                "required_consecutive"
            ]
        )


def test_label_normalization_matches_historical():
    h = historical()

    for value in (
        "Activity",
        "Falling",
        "FALLING",
        "activity",
        b"Falling",
        "walk",
    ):
        assert (
            normalise_activity_label(
                value
            )
            == h._normalise_activity_label(
                value
            )
        )


def test_trigger_semantics_match_historical():
    h = historical()

    probabilities = np.asarray(
        [
            0.9,
            0.9,
            0.1,
            0.9,
            0.9,
        ],
        dtype=float,
    )

    assert trigger_episodes(
        probabilities,
        0.8,
        2,
    ) == h.trigger_episodes(
        probabilities,
        0.8,
        2,
    )


def test_first_valid_trigger_matches_historical():
    h = historical()

    probabilities = np.asarray(
        [
            0.9,
            0.9,
            0.1,
            0.9,
            0.9,
        ],
        dtype=float,
    )

    labels = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
        ],
        dtype=int,
    )

    assert first_valid_trigger(
        probabilities,
        0.8,
        2,
        labels,
    ) == h.first_valid_trigger(
        probabilities,
        0.8,
        2,
        labels,
    )


def test_event_metrics_match_historical():
    h = historical()

    rows = [
        {
            "true_event":
                "ACTIVITY",

            "segment_probabilities":
                np.asarray(
                    [0.1, 0.9, 0.9, 0.1],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [0, 0, 0, 0],
                    dtype=int,
                ),

            "window_ends":
                np.asarray(
                    [30, 45, 60, 75],
                    dtype=int,
                ),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                -1,

            "impact_position":
                -1,
        },
        {
            "true_event":
                "FALLING",

            "segment_probabilities":
                np.asarray(
                    [0.2, 0.3, 0.92, 0.93],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [0, 0, 1, 1],
                    dtype=int,
                ),

            "window_ends":
                np.asarray(
                    [30, 45, 80, 95],
                    dtype=int,
                ),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                50,

            "impact_position":
                130,
        },
    ]

    ours = event_metrics(
        rows,
        0.8,
        2,
    )

    theirs = h.event_metrics(
        rows,
        0.8,
        2,
    )

    assert set(ours) == set(theirs)

    for key in ours:
        a = float(
            ours[key]
        )

        b = float(
            theirs[key]
        )

        if (
            math.isnan(a)
            and math.isnan(b)
        ):
            continue

        assert a == pytest.approx(
            b,
            abs=1e-12,
            rel=0.0,
        )


def test_direct_softmax_probability_semantics():
    assert clean_falling_probability({
        "clean_softmax_values":
            [0.2, 0.8],
    }) == 0.8

    assert faulted_falling_probability({
        "faulted_softmax_values":
            [0.1, 0.9],
    }) == 0.9


def test_transient_reconstruction():
    out = reconstruct_fault_scenario(
        [0.1, 0.2, 0.3],
        [
            {
                "outer_instance_id":
                    "x",

                "parent": {
                    "window_index":
                        1,
                },

                "faulted_softmax_values":
                    [0.01, 0.99],
            },
        ],
        outer_instance_id="x",
        persistence="transient_one_inference",
    )

    assert np.array_equal(
        out,
        np.asarray(
            [0.1, 0.99, 0.3]
        ),
    )


def test_persistent_reconstruction():
    rows = [
        {
            "outer_instance_id":
                "p",

            "execution_window_index":
                1,

            "onset_or_inference_index":
                1,

            "faulted_softmax_values":
                [0.1, 0.9],
        },
        {
            "outer_instance_id":
                "p",

            "execution_window_index":
                2,

            "onset_or_inference_index":
                1,

            "faulted_softmax_values":
                [0.2, 0.8],
        },
    ]

    out = reconstruct_fault_scenario(
        [0.1, 0.2, 0.3],
        rows,
        outer_instance_id="p",
        persistence="persistent_from_onset_to_trial_end",
    )

    assert np.array_equal(
        out,
        np.asarray(
            [0.1, 0.9, 0.8]
        ),
    )


def test_persistent_gap_aborts():
    with pytest.raises(
        AnalysisContractError
    ):
        reconstruct_fault_scenario(
            [0.1, 0.2, 0.3],
            [
                {
                    "outer_instance_id":
                        "p",

                    "execution_window_index":
                        1,

                    "onset_or_inference_index":
                        1,

                    "faulted_softmax_values":
                        [0.1, 0.9],
                },
            ],
            outer_instance_id="p",
            persistence="persistent_from_onset_to_trial_end",
        )


def test_duplicate_execution_index_aborts():
    row = {
        "outer_instance_id":
            "p",

        "execution_window_index":
            1,

        "onset_or_inference_index":
            1,

        "faulted_softmax_values":
            [0.1, 0.9],
    }

    with pytest.raises(
        AnalysisContractError
    ):
        reconstruct_fault_scenario(
            [0.1, 0.2],
            [row, dict(row)],
            outer_instance_id="p",
            persistence="persistent_from_onset_to_trial_end",
        )


def test_paired_clean_is_non_aliasing():
    original = [
        0.1,
        0.2,
    ]

    paired = paired_clean_counterfactual(
        original
    )

    paired[0] = 0.9

    assert original == [
        0.1,
        0.2,
    ]


def test_degradation_direction():
    assert paired_metric_degradation(
        "fall_recall",
        clean_value=0.9,
        faulted_value=0.7,
    ) == pytest.approx(
        0.2,
        abs=1e-12,
    )

    assert paired_metric_degradation(
        "false_triggers_per_activity_hour",
        clean_value=2.0,
        faulted_value=5.0,
    ) == pytest.approx(
        3.0,
        abs=1e-12,
    )

    assert paired_metric_degradation(
        "median_trigger_lead_ms",
        clean_value=300.0,
        faulted_value=250.0,
    ) == pytest.approx(
        50.0,
        abs=1e-12,
    )


def test_equal_seed_then_equal_subject_weighting():
    rows = [
        {
            "subject": subject,
            "seed": seed,
            "value": value,
        }
        for subject, values in {
            1: {
                42: 1.0,
                123: 2.0,
                2025: 3.0,
            },
            2: {
                42: 4.0,
                123: 5.0,
                2025: 6.0,
            },
        }.items()
        for seed, value in values.items()
    ]

    values = seed_equal_subject_values(
        rows
    )

    assert values == {
        1: 2.0,
        2: 5.0,
    }

    assert equal_subject_mean(
        values
    ) == 3.5


def test_bootstrap_is_deterministic():
    values = {
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }

    left = subject_cluster_percentile_ci(
        values,
        replicates=200,
        rng_seed=20261006,
    )

    right = subject_cluster_percentile_ci(
        values,
        replicates=200,
        rng_seed=20261006,
    )

    assert left == right


def test_nonfinite_subject_value_not_silently_dropped():
    with pytest.raises(
        AnalysisContractError
    ):
        seed_equal_subject_values([
            {
                "subject":
                    1,

                "seed":
                    42,

                "value":
                    math.nan,
            },
        ])


def test_nonfinite_fault_count():
    assert count_nonfinite_fault_records([
        {
            "faulted_output_nonfinite":
                False,
        },
        {
            "faulted_output_nonfinite":
                True,
        },
    ]) == 1


def test_qualification_boundary_and_pure_core():
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
        "pure_analysis_core_qualified"
    ] is True

    assert x[
        "implementation_boundary"
    ][
        "prospective_IO_runner_implemented"
    ] is False

    for path in (
        ANALYZER,
        QUALIFIER,
    ):
        source = path.read_text()

        for forbidden in (
            "phase5_outer_compute_fi_prospective_v1",
            "clean_outputs.jsonl",
            "fault_outputs.jsonl",
            "labels.npy",
            "np.load(",
            "torch.load(",
        ):
            assert forbidden not in source
