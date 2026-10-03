import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EXEC = (
    ROOT
    / "experiments/phase_04/"
    "onfield_external_executor_v1.py"
)

CFG = (
    ROOT
    / "configs/evaluation/"
    "phase4i_onfield_external_executor_v1.json"
)

PROTO = (
    ROOT
    / "configs/evaluation/"
    "phase4i_onfield_external_evaluation_protocol_v1.json"
)


def module():
    spec = importlib.util.spec_from_file_location(
        "_phase4i_executor_test",
        EXEC,
    )

    m = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        m
    )

    return m


def test_activity_duration_formula():
    m = module()

    assert m.trial_activity_seconds(
        1
    ) == 0.3

    assert m.trial_activity_seconds(
        2
    ) == 0.45

    assert m.trial_activity_seconds(
        0
    ) == 0.0


def test_historical_trigger_reference_rearm_semantics():
    m = module()

    p = np.asarray([
        0.1,
        0.8,
        0.9,
        0.95,
        0.1,
        0.8,
        0.9,
    ])

    assert m.trigger_episodes_reference(
        p,
        0.5,
        2,
    ) == [
        1,
        5,
    ]


def test_trial_boundary_resets_trigger_state():
    m = module()

    first = m.trial_metrics(
        np.asarray([
            0.1,
            0.9,
        ]),
        threshold=0.5,
        required_consecutive=2,
        trigger_fn=
            m.trigger_episodes_reference,
    )

    second = m.trial_metrics(
        np.asarray([
            0.9,
            0.1,
        ]),
        threshold=0.5,
        required_consecutive=2,
        trigger_fn=
            m.trigger_episodes_reference,
    )

    assert first[
        "false_trigger_episode_count"
    ] == 0

    assert second[
        "false_trigger_episode_count"
    ] == 0

    # Concatenating the trials would incorrectly make one trigger.
    assert m.trigger_episodes_reference(
        np.asarray([
            0.1,
            0.9,
            0.9,
            0.1,
        ]),
        0.5,
        2,
    ) == [
        1,
    ]


def test_subject_checkpoint_uses_denominator_aggregation():
    m = module()

    base = {
        "subject": "1001",
        "model_variant":
            "prospective_fp32_300ms",
        "checkpoint_seed": 42,
        "fold": 1,
        "operating_point": "balanced",
        "threshold": 0.5,
        "required_consecutive": 2,
        "probability_reused_across_operating_points":
            True,
    }

    rows = [
        {
            **base,
            "trial_id": "a",
            "activity_window_count": 10,
            "true_negative_window_count": 9,
            "false_positive_window_count": 1,
            "activity_specificity": 0.9,
            "false_trigger_episode_count": 1,
            "activity_seconds": 2.0,
            "false_triggers_per_activity_hour": 1800.0,
        },
        {
            **base,
            "trial_id": "b",
            "activity_window_count": 90,
            "true_negative_window_count": 81,
            "false_positive_window_count": 9,
            "activity_specificity": 0.9,
            "false_trigger_episode_count": 3,
            "activity_seconds": 18.0,
            "false_triggers_per_activity_hour": 600.0,
        },
    ]

    out = m.aggregate_subject_checkpoint(
        rows
    )

    assert len(out) == 1

    x = out[0]

    assert x[
        "activity_window_count"
    ] == 100

    assert x[
        "false_positive_window_count"
    ] == 10

    assert x[
        "activity_specificity"
    ] == 0.9

    assert x[
        "false_trigger_episode_count"
    ] == 4

    assert x[
        "activity_seconds"
    ] == 20.0

    assert np.isclose(
        x[
            "false_triggers_per_activity_hour"
        ],
        720.0,
    )


def test_equal_fold_then_seed_estate_weighting():
    m = module()

    rows = []

    for seed_index, seed in enumerate(
        m.SEEDS
    ):
        for fold in m.FOLDS:
            rows.append({
                "checkpoint_seed":
                    seed,

                "fold":
                    fold,

                "activity_specificity":
                    float(
                        seed_index
                        + fold
                    ),
            })

    value, seed_values = (
        m.equal_fold_then_seed_mean(
            rows,
            value_key=
                "activity_specificity",
        )
    )

    expected_seed = {
        42: 3.0,
        123: 4.0,
        2025: 5.0,
    }

    assert seed_values == expected_seed

    assert value == 4.0


def test_bootstrap_is_deterministic_and_subject_level():
    m = module()

    rows = [
        {
            "subject":
                str(
                    1001
                    + i
                ),

            "activity_specificity":
                0.90
                + 0.001
                * i,
        }
        for i in range(
            10
        )
    ]

    a = m.bootstrap_subject_mean(
        rows,
        value_key=
            "activity_specificity",

        seed_parts=(
            "synthetic",
            "fp32",
            "balanced",
        ),

        replicates=1000,
    )

    b = m.bootstrap_subject_mean(
        rows,
        value_key=
            "activity_specificity",

        seed_parts=(
            "synthetic",
            "fp32",
            "balanced",
        ),

        replicates=1000,
    )

    assert a == b

    assert a[
        "eligible_subject_count"
    ] == 10

    assert a[
        "sampling_unit"
    ] == "subject"


def test_protocol_has_no_fall_side_external_metrics():
    p = json.loads(
        PROTO.read_text()
    )

    allowed = set(
        p[
            "claim_boundary"
        ][
            "allowed"
        ]
    )

    prohibited = set(
        p[
            "claim_boundary"
        ][
            "prohibited"
        ]
    )

    assert (
        "activity_specificity"
        in allowed
    )

    assert (
        "false_triggers_per_activity_hour"
        in allowed
    )

    assert (
        "fall_recall"
        in prohibited
    )

    assert (
        "sensor_lead_time"
        in prohibited
    )


def test_executor_config_forbids_selection_and_feedback():
    x = json.loads(
        CFG.read_text()
    )

    boundary = x[
        "scientific_boundary"
    ]

    assert all(
        value is False
        for value
        in boundary.values()
    )

    assert x[
        "estate_summary"
    ][
        "checkpoint_selection"
    ] is False

    assert x[
        "estate_summary"
    ][
        "checkpoint_performance_weighting"
    ] is False

    assert x[
        "execution"
    ][
        "store_probabilities"
    ] is False
