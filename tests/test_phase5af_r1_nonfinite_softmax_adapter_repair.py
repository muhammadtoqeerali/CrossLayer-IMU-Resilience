from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

V1 = (
    ROOT
    / "experiments/phase_05/"
    "cc_outcome_analyzer_v1.py"
)

V2 = (
    ROOT
    / "experiments/phase_05/"
    "cc_outcome_analyzer_v2.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
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


v1 = load_module(
    "phase5af_r1_test_v1",
    V1,
)

v2 = load_module(
    "phase5af_r1_test_v2",
    V2,
)


def test_v1_is_historical_and_v2_is_distinct():
    assert V1.read_bytes() != V2.read_bytes()


def test_finite_probability_behavior_is_identical():
    record = {
        "clean_softmax_values":
            [0.25, 0.75],

        "faulted_softmax_values":
            [0.10, 0.90],
    }

    assert (
        v1.clean_falling_probability(
            record
        )
        == v2.clean_falling_probability(
            record
        )
    )

    assert (
        v1.faulted_falling_probability(
            record
        )
        == v2.faulted_falling_probability(
            record
        )
    )


def test_nan_token_decodes_to_nan():
    value = v2.faulted_falling_probability({
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

    assert math.isnan(
        value
    )


def test_infinity_tokens_decode_exactly():
    assert (
        v2._decode_float_token({
            "nonfinite":
                "+inf",
        })
        == math.inf
    )

    assert (
        v2._decode_float_token({
            "nonfinite":
                "-inf",
        })
        == -math.inf
    )


def test_unknown_token_aborts():
    with pytest.raises(
        v2.AnalysisContractError
    ):
        v2._decode_float_token({
            "nonfinite":
                "other",
        })


def test_extra_dictionary_key_aborts():
    with pytest.raises(
        v2.AnalysisContractError
    ):
        v2._decode_float_token({
            "nonfinite":
                "nan",

            "extra":
                True,
        })


def test_nan_does_not_satisfy_threshold():
    assert v2.trigger_episodes(
        [
            math.nan,
            math.nan,
        ],
        0.0,
        1,
    ) == []


def test_transient_nan_reconstruction():
    output = v2.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
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
        outer_instance_id="x",
        persistence="transient_one_inference",
    )

    assert output[
        0
    ] == pytest.approx(
        0.1
    )

    assert math.isnan(
        float(
            output[
                1
            ]
        )
    )


def test_event_metric_semantics_remain_identical():
    rows = [
        {
            "true_event":
                "ACTIVITY",

            "segment_probabilities":
                np.asarray(
                    [
                        math.nan,
                        0.9,
                    ]
                ),

            "segment_labels":
                np.asarray(
                    [
                        0,
                        0,
                    ]
                ),

            "window_ends":
                np.asarray(
                    [
                        30,
                        45,
                    ]
                ),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                -1,

            "impact_position":
                -1,
        },
    ]

    left = v1.event_metrics(
        rows,
        0.8,
        1,
    )

    right = v2.event_metrics(
        rows,
        0.8,
        1,
    )

    assert left.keys() == right.keys()

    for key in left:
        a = float(
            left[
                key
            ]
        )

        b = float(
            right[
                key
            ]
        )

        if (
            math.isnan(
                a
            )
            and math.isnan(
                b
            )
        ):
            continue

        assert a == pytest.approx(
            b,
            abs=1e-12,
            rel=0.0,
        )
