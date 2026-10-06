from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_module(
    name,
    path,
):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--v1",
        required=True,
    )

    parser.add_argument(
        "--v2",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    config = json.loads(
        Path(
            args.config
        ).read_text()
    )

    v1 = load_module(
        "phase5af_r1_v1",
        args.v1,
    )

    v2 = load_module(
        "phase5af_r1_v2",
        args.v2,
    )

    checks = {}

    finite_records = [
        {
            "clean_softmax_values":
                [0.2, 0.8],

            "faulted_softmax_values":
                [0.1, 0.9],
        },
        {
            "clean_softmax_values":
                [1.0, 0.0],

            "faulted_softmax_values":
                [0.75, 0.25],
        },
    ]

    for record in finite_records:
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

    checks[
        "finite_probability_behavior_equivalent_to_v1"
    ] = True

    nan_record = {
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
    }

    assert math.isnan(
        v2.faulted_falling_probability(
            nan_record
        )
    )

    checks[
        "nan_token_decodes_to_ieee_nan"
    ] = True

    assert (
        v2._decode_float_token({
            "nonfinite":
                "+inf",
        })
        == math.inf
    )

    checks[
        "positive_infinity_token_decodes_exactly"
    ] = True

    assert (
        v2._decode_float_token({
            "nonfinite":
                "-inf",
        })
        == -math.inf
    )

    checks[
        "negative_infinity_token_decodes_exactly"
    ] = True

    try:
        v2._decode_float_token({
            "nonfinite":
                "bogus",
        })
    except v2.AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "unknown nonfinite token was accepted"
        )

    checks[
        "unknown_token_aborts"
    ] = True

    try:
        v2._decode_float_token({
            "nonfinite":
                "nan",

            "other":
                "unexpected",
        })
    except v2.AnalysisContractError:
        pass
    else:
        raise AssertionError(
            "unexpected token dictionary shape was accepted"
        )

    checks[
        "unexpected_dictionary_shape_aborts"
    ] = True

    episodes = v2.trigger_episodes(
        [
            math.nan,
            0.9,
            0.1,
        ],
        0.8,
        1,
    )

    assert episodes == [
        1
    ]

    assert v2.trigger_episodes(
        [
            math.nan,
            math.nan,
        ],
        0.0,
        1,
    ) == []

    checks[
        "nan_preserves_frozen_threshold_nontrigger_semantics"
    ] = True

    reconstructed = v2.reconstruct_fault_scenario(
        [
            0.1,
            0.2,
            0.3,
        ],
        [
            {
                "outer_instance_id":
                    "repair-nan",

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
        outer_instance_id="repair-nan",
        persistence="transient_one_inference",
    )

    assert reconstructed.shape == (
        3,
    )

    assert reconstructed[
        0
    ] == 0.1

    assert math.isnan(
        float(
            reconstructed[
                1
            ]
        )
    )

    assert reconstructed[
        2
    ] == 0.3

    checks[
        "nan_fault_scenario_reconstruction_preserves_nan"
    ] = True

    rows = [
        {
            "true_event":
                "ACTIVITY",

            "segment_probabilities":
                np.asarray(
                    [
                        0.1,
                        math.nan,
                        0.9,
                    ],
                    dtype=float,
                ),

            "segment_labels":
                np.asarray(
                    [
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
                        0.1,
                        math.nan,
                        0.9,
                        0.9,
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
    ]

    v1_metrics = v1.event_metrics(
        rows,
        0.8,
        1,
    )

    v2_metrics = v2.event_metrics(
        rows,
        0.8,
        1,
    )

    assert set(
        v1_metrics
    ) == set(
        v2_metrics
    )

    for key in v1_metrics:
        left = float(
            v1_metrics[
                key
            ]
        )

        right = float(
            v2_metrics[
                key
            ]
        )

        if (
            math.isnan(
                left
            )
            and math.isnan(
                right
            )
        ):
            continue

        assert math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    checks[
        "historical_metric_semantics_unchanged_with_nan_probabilities"
    ] = True

    assert len(
        checks
    ) == 9

    assert all(
        checks.values()
    )

    result = {
        "schema_version":
            "phase5af_r1_nonfinite_softmax_adapter_repair_qualification_v1",

        "phase":
            "5AF-R1",

        "status":
            "QUALIFIED_POST_BOUNDARY_NONFINITE_ADAPTER_REPAIR",

        "scientific_protocol_changed":
            False,

        "v1_analyzer_sha256":
            sha(
                args.v1
            ),

        "v2_analyzer_sha256":
            sha(
                args.v2
            ),

        "qualification_check_count":
            len(
                checks
            ),

        "qualification_checks":
            checks,

        "repair_semantics": {
            "finite_values":
                "unchanged",

            "nan_token":
                "decode to IEEE NaN",

            "positive_infinity_token":
                "decode to IEEE +inf",

            "negative_infinity_token":
                "decode to IEEE -inf",

            "unknown_dictionary":
                "abort",

            "imputation":
                False,

            "clipping":
                False,

            "threshold_change":
                False,

            "metric_change":
                False,

            "aggregation_change":
                False,
        },

        "qualification_boundary": {
            "synthetic_only":
                True,

            "phase5af_shard_resume":
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

            "fault_execution_executed":
                False,
        },

        "next_action":
            (
                "Create and qualify a repaired Phase-5AD executor variant "
                "bound to this exact V2 analyzer, then freeze a "
                "post-boundary implementation-repair continuation "
                "authorization before resuming Phase-5AF."
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
        "PHASE5AF_R1_STATUS=",
        result[
            "status"
        ],
        sep="",
    )

    print(
        "QUALIFICATION_CHECKS=",
        len(
            checks
        ),
        sep="",
    )

    print(
        "V2_ANALYZER_SHA256=",
        result[
            "v2_analyzer_sha256"
        ],
        sep="",
    )

    print(
        "SCIENTIFIC_PROTOCOL_CHANGED=False"
    )

    print(
        "PHASE5AF_SHARD_RESUME=False"
    )


if __name__ == "__main__":
    main()
