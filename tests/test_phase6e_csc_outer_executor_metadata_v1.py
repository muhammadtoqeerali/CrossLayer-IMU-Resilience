from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import csc_outer_executor_v1 as csc


ROOT = Path(__file__).resolve().parents[1]


def cj(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_execution_is_hard_disabled():
    assert csc.EXECUTION_ENABLED is False

    with pytest.raises(
        RuntimeError,
        match="PHASE6E_EXECUTION_DISABLED_METADATA_ONLY",
    ):
        csc.execute_shard(
            shard_id="anything"
        )


def test_module_does_not_import_model_or_fault_execution_runtime():
    source = (
        ROOT
        / "experiments/phase_06/csc_outer_executor_v1.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "\nimport torch" not in source
    assert "\nfrom torch" not in source
    assert "import sensor_fi_operators" not in source
    assert "execute_fault_sequence(" not in source
    assert "load_model_bundle(" not in source


def test_frozen_plan_validation():
    validated = csc.validate_frozen_plan()

    plan = validated[
        "plan"
    ]

    assert (
        plan[
            "status"
        ]
        == "FROZEN_PRE_EXECUTION_CSC_PLAN"
    )

    assert (
        plan[
            "pair_surface"
        ][
            "model_independent_csc_pair_count"
        ]
        == 4237835
    )

    assert (
        plan[
            "model_variant_surface"
        ][
            "combined_seed_variant_expanded_pair_member_count"
        ]
        == 21793038
    )


def test_stored_window_hash_min_selection():
    validated = csc.validate_frozen_plan()

    phase6d = validated[
        "phase6d"
    ]

    candidates = [
        {
            "parent_kind":
                "stored_window",

            "parent_sequence_id":
                "subject=9|task=1|trial=1|window=0",

            "family":
                "bias",

            "fault_id":
                "fault-A",

            "replay_id":
                "replay-A",
        },

        {
            "parent_kind":
                "stored_window",

            "parent_sequence_id":
                "subject=9|task=1|trial=1|window=0",

            "family":
                "bias",

            "fault_id":
                "fault-B",

            "replay_id":
                "replay-B",
        },
    ]

    namespace = phase6d[
        "sensor_surface"
    ][
        "selected_instance_rule"
    ][
        "selection_namespace"
    ]

    expected = min(
        candidates,
        key=lambda row: (
            hashlib.sha256(
                cj(
                    {
                        "namespace":
                            namespace,

                        "fold":
                            5,

                        "subject":
                            9,

                        "parent_kind":
                            row[
                                "parent_kind"
                            ],

                        "parent_sequence_id":
                            row[
                                "parent_sequence_id"
                            ],

                        "family":
                            "bias",

                        "severity":
                            "L1",

                        "fault_id":
                            row[
                                "fault_id"
                            ],

                        "replay_id":
                            row[
                                "replay_id"
                            ],
                    }
                ).encode(
                    "utf-8"
                )
            ).hexdigest(),
            row[
                "fault_id"
            ],
        ),
    )

    observed = csc.select_stored_window_candidate(
        candidates=candidates,
        fold=5,
        subject=9,
        family="bias",
        severity="L1",
        phase6d=phase6d,
    )

    assert observed == expected


def test_r2_source_trial_filters_before_hash_min():
    validated = csc.validate_frozen_plan()

    phase6d = validated[
        "phase6d"
    ]

    candidates = [
        {
            "parent_kind":
                "source_trial",

            "parent_sequence_id":
                "subject=9|task=1|trial=1",

            "family":
                "drift",

            "fault_id":
                "fault-unobservable",

            "replay_id":
                "replay-unobservable",
        },

        {
            "parent_kind":
                "source_trial",

            "parent_sequence_id":
                "subject=9|task=1|trial=1",

            "family":
                "drift",

            "fault_id":
                "fault-observable",

            "replay_id":
                "replay-observable",
        },
    ]

    observed = csc.select_source_trial_candidate(
        candidates=candidates,
        exposure_count_by_fault_id={
            "fault-unobservable":
                0,

            "fault-observable":
                3,
        },
        fold=5,
        subject=9,
        family="drift",
        severity="L1",
        phase6d=phase6d,
    )

    assert (
        observed[
            "status"
        ]
        == "ELIGIBLE_SELECTED"
    )

    assert (
        observed[
            "selected"
        ][
            "fault_id"
        ]
        == "fault-observable"
    )


def test_r2_source_trial_structural_omission_has_no_fallback():
    validated = csc.validate_frozen_plan()

    observed = csc.select_source_trial_candidate(
        candidates=[
            {
                "parent_kind":
                    "source_trial",

                "parent_sequence_id":
                    "subject=9|task=1|trial=1",

                "family":
                    "drift",

                "fault_id":
                    "fault-1",

                "replay_id":
                    "replay-1",
            },
            {
                "parent_kind":
                    "source_trial",

                "parent_sequence_id":
                    "subject=9|task=1|trial=1",

                "family":
                    "drift",

                "fault_id":
                    "fault-2",

                "replay_id":
                    "replay-2",
            },
        ],
        exposure_count_by_fault_id={
            "fault-1":
                0,

            "fault-2":
                0,
        },
        fold=5,
        subject=9,
        family="drift",
        severity="L2",
        phase6d=validated[
            "phase6d"
        ],
    )

    assert observed == {
        "status":
            "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR",

        "selected":
            None,
    }


def test_compute_stratum_assignment_matches_manual_hash():
    validated = csc.validate_frozen_plan()

    phase6d = validated[
        "phase6d"
    ]

    sensor = {
        "fault_id":
            "sensor-fault-example",

        "replay_id":
            "sensor-replay-example",
    }

    payload = {
        "namespace":
            phase6d[
                "compute_stratum_assignment"
            ][
                "namespace"
            ],

        "sensor_fault_id":
            sensor[
                "fault_id"
            ],

        "sensor_replay_id":
            sensor[
                "replay_id"
            ],
    }

    digest = hashlib.sha256(
        cj(
            payload
        ).encode(
            "utf-8"
        )
    ).digest()

    expected_index = (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        % 28
    )

    index, stratum = csc.assign_compute_stratum(
        sensor_instance=sensor,
        phase6d=phase6d,
    )

    assert index == expected_index

    assert (
        stratum
        == phase6d[
            "compute_surface"
        ][
            "compute_strata"
        ][
            expected_index
        ]
    )


def test_source_transient_window_matches_manual_r1_hash():
    validated = csc.validate_frozen_plan()

    r1 = validated[
        "phase6d_r1"
    ]

    windows = [
        2,
        5,
        8,
        9,
    ]

    rule = r1[
        "source_trial_transient_compute_window_selection"
    ]

    payload = {
        "namespace":
            rule[
                "namespace"
            ],

        "sensor_replay_id":
            "sensor-replay-example",

        "target_name":
            "front_end_output_fp32",
    }

    digest = hashlib.sha256(
        cj(
            payload
        ).encode(
            "utf-8"
        )
    ).digest()

    expected = windows[
        (
            int.from_bytes(
                digest[:8],
                byteorder="big",
                signed=False,
            )
            % len(
                windows
            )
        )
    ]

    observed = csc.source_trial_transient_window_index(
        sensor_replay_id="sensor-replay-example",
        target_name="front_end_output_fp32",
        exposed_window_indices=windows,
        phase6d_r1=r1,
    )

    assert observed == expected


def test_compute_coordinate_matches_frozen_phase5_sampler():
    validated = csc.validate_frozen_plan()

    target = validated[
        "phase5d"
    ][
        "target_inventory"
    ][
        0
    ]

    observed = csc.build_compute_coordinate(
        fold=5,
        subject=9,
        task=1,
        trial=1,
        trial_window_count=242,
        target=target,
        persistence="transient_one_inference",
        transient_window_index=17,
    )

    payload = csc.compute_sampling.canonical_sampling_payload(
        partition="outer_test",
        fold=5,
        subject=9,
        task=1,
        trial=1,
        parent_kind="window",
        window_index=17,
        representation_class=target[
            "representation_class"
        ],
        target_name=target[
            "target_name"
        ],
        target_role=target[
            "target_role"
        ],
        persistence="transient_one_inference",
        replicate_index=0,
    )

    assert (
        observed[
            "sampling_instance_id"
        ]
        == csc.compute_sampling.sampling_instance_id(
            payload
        )
    )

    assert (
        observed[
            "element_index"
        ]
        == csc.compute_sampling.derive_element_index(
            payload,
            target_numel=target[
                "target_numel_per_inference"
            ],
        )
    )

    assert (
        observed[
            "bit_position"
        ]
        == csc.compute_sampling.derive_bit_position(
            payload,
            eligible_bits=target[
                "eligible_bit_positions"
            ],
        )
    )

    assert observed["inference_index"] == 17


def test_transient_temporal_accounting_requires_simultaneous_overlap():
    observed = csc.temporal_accounting(
        sensor_exposed_window_indices=[
            3,
            4,
            5,
        ],
        trial_window_count=10,
        persistence="transient_one_inference",
        compute_inference_index=4,
    )

    assert observed == {
        "sensor_exposed_window_count":
            3,

        "compute_active_window_count":
            1,

        "temporal_overlap":
            True,

        "temporal_overlap_window_count":
            1,

        "sensor_compute_union_window_count":
            3,

        "overlap_window_indices":
            [4],
    }


def test_persistent_zero_overlap_is_retained_as_metadata():
    observed = csc.temporal_accounting(
        sensor_exposed_window_indices=[
            0,
            1,
        ],
        trial_window_count=10,
        persistence="persistent_from_onset_until_trial_end",
        compute_inference_index=5,
    )

    assert observed["temporal_overlap"] is False
    assert observed["temporal_overlap_window_count"] == 0
    assert observed["compute_active_window_count"] == 5
    assert observed["sensor_compute_union_window_count"] == 7


def test_pair_metadata_derivation_executes_nothing():
    validated = csc.validate_frozen_plan()

    phase6d = validated[
        "phase6d"
    ]

    # Search only over synthetic identities until one is assigned to
    # a transient stratum. No outer data or model is touched.
    sensor = None
    stratum = None

    for index in range(
        1000
    ):
        candidate = {
            "fault_id":
                f"synthetic-fault-{index}",

            "replay_id":
                f"synthetic-replay-{index}",
        }

        _, candidate_stratum = (
            csc.assign_compute_stratum(
                sensor_instance=candidate,
                phase6d=phase6d,
            )
        )

        if (
            candidate_stratum[
                "persistence"
            ]
            == "transient_one_inference"
        ):
            sensor = candidate
            stratum = candidate_stratum
            break

    assert sensor is not None
    assert stratum is not None

    metadata = csc.derive_pair_metadata(
        sensor_parent_kind="stored_window",
        sensor_instance=sensor,
        sensor_exposed_window_indices=[
            7
        ],
        fold=5,
        subject=9,
        task=1,
        trial=1,
        trial_window_count=242,
        validated=validated,
    )

    assert metadata["execution_performed"] is False
    assert metadata["persistence"] == "transient_one_inference"
    assert metadata["temporal_accounting"]["temporal_overlap"] is True
    assert metadata["temporal_accounting"]["temporal_overlap_window_count"] == 1

    assert metadata["pair_member_multiplier"] in {
        3,
        6,
    }


def test_validate_config_cli_is_metadata_only():
    script = (
        ROOT
        / "experiments/phase_06/csc_outer_executor_v1.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                script
            ),
            "validate-config",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "PHASE6E_METADATA_EXECUTOR_VALIDATE_CONFIG=PASS"
        in result.stdout
    )

    assert (
        "MODEL_INDEPENDENT_CSC_PAIR_COUNT=4237835"
        in result.stdout
    )

    assert (
        "PAIR_MEMBER_COUNT=21793038"
        in result.stdout
    )

    assert (
        "OUTER_ARRAY_READ=False"
        in result.stdout
    )

    assert (
        "MODEL_LOADED=False"
        in result.stdout
    )

    assert (
        "MODEL_FORWARD_EXECUTED=False"
        in result.stdout
    )

    assert (
        "FAULT_EXECUTION_EXECUTED=False"
        in result.stdout
    )
