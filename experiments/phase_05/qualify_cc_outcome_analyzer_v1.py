from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cc_outcome_analyzer_v1 import (
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


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text()
    )


def load_historical(path: str | Path):
    spec = importlib.util.spec_from_file_location(
        "phase5aa_historical_evaluator",
        Path(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load historical evaluator"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def metric_equal(
    left: dict[str, float | int],
    right: dict[str, float | int],
) -> bool:
    if set(left) != set(right):
        return False

    for key in left:
        a = left[key]
        b = right[key]

        if isinstance(
            a,
            (int, np.integer),
        ) and isinstance(
            b,
            (int, np.integer),
        ):
            if int(a) != int(b):
                return False
            continue

        af = float(a)
        bf = float(b)

        if (
            math.isnan(af)
            and math.isnan(bf)
        ):
            continue

        if not math.isclose(
            af,
            bf,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False

    return True


def qualify(args) -> dict[str, Any]:
    cfg = load_json(
        args.config
    )

    protocol = load_json(
        cfg[
            "phase5z_protocol"
        ][
            "path"
        ]
    )

    if sha256_file(
        cfg[
            "phase5z_protocol"
        ][
            "path"
        ]
    ) != cfg[
        "phase5z_protocol"
    ][
        "sha256"
    ]:
        raise ValueError(
            "Phase-5Z protocol hash mismatch"
        )

    if sha256_file(
        cfg[
            "phase5z_freeze_manifest"
        ][
            "path"
        ]
    ) != cfg[
        "phase5z_freeze_manifest"
    ][
        "sha256"
    ]:
        raise ValueError(
            "Phase-5Z manifest hash mismatch"
        )

    if sha256_file(
        cfg[
            "historical_evaluator"
        ][
            "path"
        ]
    ) != cfg[
        "historical_evaluator"
    ][
        "sha256"
    ]:
        raise ValueError(
            "historical evaluator hash mismatch"
        )

    if sha256_file(
        cfg[
            "phase5u_technical_acceptance"
        ][
            "path"
        ]
    ) != cfg[
        "phase5u_technical_acceptance"
    ][
        "sha256"
    ]:
        raise ValueError(
            "Phase-5U acceptance hash mismatch"
        )

    historical = load_historical(
        cfg[
            "historical_evaluator"
        ][
            "path"
        ]
    )

    checks: dict[str, Any] = {}

    # Exact 45-row threshold matrix.
    index = threshold_index(
        protocol
    )

    assert len(index) == 45

    for key, row in index.items():
        threshold, consecutive = threshold_rule(
            protocol,
            seed=key[0],
            fold=key[1],
            operating_point=key[2],
        )

        assert threshold == row[
            "threshold"
        ]

        assert consecutive == row[
            "required_consecutive"
        ]

    checks[
        "threshold_matrix_45_exact"
    ] = True

    # Historical label normalization equivalence.
    label_cases = [
        "Activity",
        "Falling",
        "FALLING",
        "activity",
        b"Falling",
        "not_fall",
        "walk",
    ]

    for value in label_cases:
        assert (
            normalise_activity_label(
                value
            )
            == historical._normalise_activity_label(
                value
            )
        )

    checks[
        "label_normalization_equivalence"
    ] = True

    # Exact threshold/consecutive trigger equivalence.
    trigger_cases = [
        (
            [0.1, 0.8, 0.8, 0.1],
            0.8,
            1,
        ),
        (
            [0.1, 0.8, 0.8, 0.1],
            0.8,
            2,
        ),
        (
            [0.9, 0.9, 0.2, 0.9, 0.9],
            0.8,
            2,
        ),
        (
            [float("nan"), 0.9, 0.1],
            0.8,
            1,
        ),
    ]

    for probabilities, threshold, consecutive in trigger_cases:
        assert trigger_episodes(
            probabilities,
            threshold,
            consecutive,
        ) == historical.trigger_episodes(
            np.asarray(
                probabilities,
                dtype=float,
            ),
            threshold,
            consecutive,
        )

    checks[
        "trigger_episode_equivalence"
    ] = True

    first_trigger_cases = [
        (
            [0.1, 0.9, 0.9, 0.2],
            0.8,
            2,
            [0, 1, 1, 0],
        ),
        (
            [0.9, 0.9, 0.2, 0.9, 0.9],
            0.8,
            2,
            [0, 0, 0, 1, 1],
        ),
        (
            [0.1, 0.9, 0.1],
            0.8,
            1,
            None,
        ),
    ]

    for probabilities, threshold, consecutive, labels in first_trigger_cases:
        ours = first_valid_trigger(
            probabilities,
            threshold,
            consecutive,
            labels,
        )

        theirs = historical.first_valid_trigger(
            np.asarray(
                probabilities,
                dtype=float,
            ),
            threshold,
            consecutive,
            (
                None
                if labels is None
                else np.asarray(
                    labels,
                    dtype=int,
                )
            ),
        )

        assert ours == theirs

    checks[
        "first_valid_trigger_equivalence"
    ] = True

    # Exact Phase-4 event metric equivalence on synthetic trial rows.
    rows = [
        {
            "true_event":
                "ACTIVITY",

            "segment_probabilities":
                np.asarray(
                    [
                        0.1,
                        0.9,
                        0.9,
                        0.1,
                    ],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [
                        0,
                        0,
                        0,
                        0,
                    ],
                    dtype=int,
                ),

            "window_ends":
                np.asarray(
                    [
                        30,
                        45,
                        60,
                        75,
                    ],
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
                    [
                        0.2,
                        0.3,
                        0.92,
                        0.93,
                    ],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [
                        0,
                        0,
                        1,
                        1,
                    ],
                    dtype=int,
                ),

            "window_ends":
                np.asarray(
                    [
                        30,
                        45,
                        80,
                        95,
                    ],
                    dtype=int,
                ),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                50,

            "impact_position":
                130,
        },
        {
            "true_event":
                "FALLING",

            "segment_probabilities":
                np.asarray(
                    [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                    ],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [
                        0,
                        0,
                        1,
                        1,
                    ],
                    dtype=int,
                ),

            "window_ends":
                np.asarray(
                    [
                        30,
                        45,
                        75,
                        90,
                    ],
                    dtype=int,
                ),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                50,

            "impact_position":
                150,
        },
    ]

    for threshold, consecutive in (
        (0.8, 1),
        (0.8, 2),
        (0.5, 1),
    ):
        ours = event_metrics(
            rows,
            threshold,
            consecutive,
        )

        theirs = historical.event_metrics(
            rows,
            threshold,
            consecutive,
        )

        assert metric_equal(
            ours,
            theirs,
        ), (
            ours,
            theirs,
        )

    checks[
        "event_metric_equivalence"
    ] = True

    # Direct Phase-5M / Phase-5R softmax semantics.
    clean_record = {
        "clean_softmax_values":
            [
                0.25,
                0.75,
            ],
    }

    fault_record = {
        "faulted_softmax_values":
            [
                0.05,
                0.95,
            ],
    }

    assert clean_falling_probability(
        clean_record
    ) == 0.75

    assert faulted_falling_probability(
        fault_record
    ) == 0.95

    checks[
        "mixed_provenance_direct_softmax"
    ] = True

    # Phase-5M-style transient record: execution index from parent.window_index.
    transient_clean = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    transient_rows = [
        {
            "outer_instance_id":
                "synthetic-transient",

            "parent": {
                "window_index":
                    2,
            },

            "faulted_softmax_values":
                [
                    0.01,
                    0.99,
                ],
        },
    ]

    transient = reconstruct_fault_scenario(
        transient_clean,
        transient_rows,
        outer_instance_id="synthetic-transient",
        persistence="transient_one_inference",
    )

    assert np.array_equal(
        transient,
        np.asarray(
            [
                0.1,
                0.2,
                0.99,
                0.4,
            ],
            dtype=float,
        ),
    )

    checks[
        "transient_reconstruction"
    ] = True

    # Phase-5R persistent suffix.
    persistent_clean = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]

    persistent_rows = [
        {
            "outer_instance_id":
                "synthetic-persistent",

            "execution_window_index":
                2,

            "onset_or_inference_index":
                2,

            "faulted_softmax_values":
                [
                    0.1,
                    0.9,
                ],
        },
        {
            "outer_instance_id":
                "synthetic-persistent",

            "execution_window_index":
                3,

            "onset_or_inference_index":
                2,

            "faulted_softmax_values":
                [
                    0.2,
                    0.8,
                ],
        },
        {
            "outer_instance_id":
                "synthetic-persistent",

            "execution_window_index":
                4,

            "onset_or_inference_index":
                2,

            "faulted_softmax_values":
                [
                    0.3,
                    0.7,
                ],
        },
    ]

    persistent = reconstruct_fault_scenario(
        persistent_clean,
        persistent_rows,
        outer_instance_id="synthetic-persistent",
        persistence="persistent_from_onset_to_trial_end",
    )

    assert np.array_equal(
        persistent,
        np.asarray(
            [
                0.1,
                0.2,
                0.9,
                0.8,
                0.7,
            ],
            dtype=float,
        ),
    )

    checks[
        "persistent_reconstruction"
    ] = True

    # Missing persistent suffix must abort.
    incomplete = [
        persistent_rows[0],
        persistent_rows[2],
    ]

    try:
        reconstruct_fault_scenario(
            persistent_clean,
            incomplete,
            outer_instance_id="synthetic-persistent",
            persistence="persistent_from_onset_to_trial_end",
        )
    except AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "missing persistent suffix was not rejected"
        )

    checks[
        "persistent_suffix_abort"
    ] = True

    # Duplicate execution index must abort.
    duplicate = [
        persistent_rows[0],
        {
            **persistent_rows[0],
        },
    ]

    try:
        reconstruct_fault_scenario(
            persistent_clean,
            duplicate,
            outer_instance_id="synthetic-persistent",
            persistence="persistent_from_onset_to_trial_end",
        )
    except AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "duplicate execution index was not rejected"
        )

    checks[
        "duplicate_execution_index_abort"
    ] = True

    # Paired clean counterfactual must be identical and non-aliasing.
    paired = paired_clean_counterfactual(
        transient_clean
    )

    assert np.array_equal(
        paired,
        np.asarray(
            transient_clean,
            dtype=float,
        ),
    )

    paired[0] = 0.99

    assert transient_clean[
        0
    ] == 0.1

    checks[
        "paired_clean_counterfactual"
    ] = True

    # Positive degradation always means worse.
    assert math.isclose(
        paired_metric_degradation(
            "fall_recall",
            clean_value=0.9,
            faulted_value=0.7,
        ),
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert paired_metric_degradation(
        "false_triggers_per_activity_hour",
        clean_value=2.0,
        faulted_value=5.0,
    ) == 3.0

    assert paired_metric_degradation(
        "median_trigger_lead_ms",
        clean_value=300.0,
        faulted_value=250.0,
    ) == 50.0

    checks[
        "paired_degradation_direction"
    ] = True

    # Equal seed weighting, then equal subject weighting.
    aggregate_rows = [
        {
            "subject":
                subject,

            "seed":
                seed,

            "value":
                value,
        }
        for subject, values in {
            10: {
                42: 1.0,
                123: 2.0,
                2025: 3.0,
            },
            11: {
                42: 4.0,
                123: 5.0,
                2025: 6.0,
            },
        }.items()
        for seed, value in values.items()
    ]

    subject_values = seed_equal_subject_values(
        aggregate_rows
    )

    assert subject_values == {
        10: 2.0,
        11: 5.0,
    }

    assert equal_subject_mean(
        subject_values
    ) == 3.5

    checks[
        "seed_subject_equal_weighting"
    ] = True

    # Bootstrap deterministic under frozen seed.
    ci_a = subject_cluster_percentile_ci(
        subject_values,
        replicates=1000,
        confidence_level=0.95,
        rng_seed=20261006,
    )

    ci_b = subject_cluster_percentile_ci(
        subject_values,
        replicates=1000,
        confidence_level=0.95,
        rng_seed=20261006,
    )

    assert ci_a == ci_b
    assert ci_a[
        "estimate"
    ] == 3.5
    assert ci_a[
        "subject_count"
    ] == 2

    checks[
        "subject_bootstrap_determinism"
    ] = True

    # Non-finite output count remains explicit.
    assert count_nonfinite_fault_records([
        {
            "faulted_output_nonfinite":
                False,
        },
        {
            "faulted_output_nonfinite":
                True,
        },
        {
            "faulted_output_nonfinite":
                True,
        },
    ]) == 2

    checks[
        "nonfinite_counting"
    ] = True

    # Do not silently aggregate non-finite subject/seed timing summaries.
    try:
        seed_equal_subject_values([
            {
                "subject":
                    10,

                "seed":
                    42,

                "value":
                    math.nan,
            },
        ])
    except AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "non-finite subject scalar was silently accepted"
        )

    checks[
        "nonfinite_aggregation_requires_explicit_future_policy"
    ] = True

    assert all(
        checks.values()
    )

    result = {
        "schema_version":
            "phase5aa_compute_fi_cc_analyzer_qualification_result_v1",

        "phase":
            "5AA",

        "status":
            "QUALIFIED_CC_ANALYZER_CORE_PRE_OUTCOME",

        "evidence_tier":
            "P0",

        "phase5z_protocol_sha256":
            sha256_file(
                cfg[
                    "phase5z_protocol"
                ][
                    "path"
                ]
            ),

        "analyzer_sha256":
            sha256_file(
                args.analyzer
            ),

        "qualifier_sha256":
            sha256_file(
                __file__
            ),

        "qualification_checks":
            checks,

        "qualification_check_count":
            len(checks),

        "historical_source_equivalence": {
            "label_normalization":
                True,

            "trigger_episodes":
                True,

            "first_valid_trigger":
                True,

            "event_metrics":
                True,
        },

        "synthetic_only": {
            "synthetic_records_used":
                True,

            "training_calibration_payload_used":
                False,

            "prospective_outer_prediction_payload_used":
                False,

            "prospective_outer_label_payload_used":
                False,
        },

        "scientific_boundary": {
            "prospective_outer_result_root_accessed":
                False,

            "clean_outputs_jsonl_opened":
                False,

            "fault_outputs_jsonl_opened":
                False,

            "prediction_json_deserialized":
                False,

            "prediction_outcomes_interpreted":
                False,

            "outer_label_array_loaded":
                False,

            "threshold_applied_to_outer":
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
            "pure_analysis_core_qualified":
                True,

            "prospective_IO_runner_implemented":
                False,

            "prospective_IO_runner_qualified":
                False,

            "outcome_analysis_executed":
                False,
        },

        "next_action":
            (
                "Implement and statically qualify a prospective CC I/O runner "
                "that binds this exact analyzer and Phase-5Z protocol before "
                "opening accepted prediction artifacts."
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

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--analyzer",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    result = qualify(
        args
    )

    print(
        "PHASE5AA_STATUS=",
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
        "PROSPECTIVE_OUTER_PREDICTION_PAYLOAD_USED=False"
    )

    print(
        "PROSPECTIVE_OUTER_LABEL_PAYLOAD_USED=False"
    )

    print(
        "OUTER_CC_METRIC_COMPUTED=False"
    )


if __name__ == "__main__":
    main()
