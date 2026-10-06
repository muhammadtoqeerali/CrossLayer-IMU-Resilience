from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_outer_sampling_v1 import (  # noqa: E402
    canonical_sampling_payload,
    derive_bit_position,
    derive_element_index,
    derive_persistent_onset_index,
    sampling_instance_id,
)


CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5d_compute_fi_outer_protocol_v1.json"
)

SAMPLER = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_outer_sampling_v1.py"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5D_COMPUTE_FI_OUTER_PROTOCOL_V1.md"
)


def load():
    return json.loads(
        CONFIG.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_status_and_partition():
    x = load()

    assert (
        x["status"]
        == "FROZEN_PROSPECTIVE_OUTER_COMPUTE_FI_PROTOCOL"
    )

    assert x[
        "partition"
    ][
        "name"
    ] == "outer_test"

    assert x[
        "partition"
    ][
        "subject_count"
    ] == 61

    assert x[
        "partition"
    ][
        "trial_count"
    ] == 6309

    assert x[
        "partition"
    ][
        "window_count"
    ] == 273830


def test_full_model_estate_is_retained():
    x = load()

    estate = x[
        "model_estate"
    ]

    assert estate[
        "model_variants"
    ] == [
        "fp32",
        "ptq_v7",
    ]

    assert estate[
        "checkpoint_seeds"
    ] == [
        42,
        123,
        2025,
    ]

    assert estate[
        "folds"
    ] == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert estate[
        "checkpoint_member_count"
    ] == 15

    assert estate[
        "single_checkpoint_selection_allowed"
    ] is False


def test_target_inventory_cardinality():
    x = load()

    assert len(
        x[
            "target_inventory"
        ]
    ) == 14

    assert (
        x[
            "target_inventory_summary"
        ][
            "common_fp32_targets"
        ]
        == 10
    )

    assert (
        x[
            "target_inventory_summary"
        ][
            "ptq_only_targets"
        ]
        == 4
    )

    assert (
        x[
            "target_inventory_summary"
        ][
            "variant_target_strata"
        ]
        == 24
    )


def test_all_five_families_frozen():
    assert set(
        load()[
            "fault_families"
        ]
    ) == {
        "int8_weight_single_bit_flip",
        "quantized_activation_single_bit_flip",
        "quantized_buffer_single_bit_flip",
        "fp32_activation_single_bit_flip",
        "fp32_buffer_single_bit_flip",
    }


def test_bit_domains_are_complete():
    x = load()

    for row in x[
        "target_inventory"
    ]:
        if row[
            "representation_class"
        ] in {
            "int8_persistent_weight",
            "quantized_activation",
            "quantized_buffer",
        }:
            assert row[
                "eligible_bit_positions"
            ] == list(
                range(
                    8
                )
            )

        else:
            assert row[
                "eligible_bit_positions"
            ] == list(
                range(
                    32
                )
            )


def test_cardinality_is_exact():
    c = load()[
        "cardinality"
    ]

    assert c[
        "unique_sampling_ids"
    ][
        "transient"
    ] == 3833620

    assert c[
        "unique_sampling_ids"
    ][
        "persistent"
    ] == 88326

    assert c[
        "unique_sampling_ids"
    ][
        "total"
    ] == 3921946

    assert c[
        "model_specific_fault_ids"
    ][
        "transient"
    ] == 19715760

    assert c[
        "model_specific_fault_ids"
    ][
        "persistent"
    ] == 454248

    assert c[
        "model_specific_fault_ids"
    ][
        "total"
    ] == 20170008

    assert c[
        "clean_C0_model_window_forwards"
    ] == 1642980


def test_sampling_is_model_and_seed_independent():
    s = load()[
        "sampling"
    ]

    assert s[
        "model_variant_in_sampling_identity"
    ] is False

    assert s[
        "checkpoint_seed_in_sampling_identity"
    ] is False

    assert s[
        "global_rng_used"
    ] is False

    assert s[
        "multiplicity"
    ] == 1

    assert s[
        "replicate_indices"
    ] == [
        0
    ]


def test_sampler_replay_and_coordinate_ranges():
    row = next(
        row
        for row in load()[
            "target_inventory"
        ]
        if row[
            "target_name"
        ]
        == "fc1_output_fp32"
    )

    payload = canonical_sampling_payload(
        partition="outer_test",
        fold=2,
        subject=7,
        task=3,
        trial=9,
        parent_kind="window",
        window_index=4,
        representation_class=row[
            "representation_class"
        ],
        target_name=row[
            "target_name"
        ],
        target_role=row[
            "target_role"
        ],
        persistence="transient_one_inference",
        replicate_index=0,
    )

    a = (
        sampling_instance_id(
            payload
        ),
        derive_element_index(
            payload,
            target_numel=row[
                "target_numel_per_inference"
            ],
        ),
        derive_bit_position(
            payload,
            eligible_bits=row[
                "eligible_bit_positions"
            ],
        ),
    )

    b = (
        sampling_instance_id(
            payload
        ),
        derive_element_index(
            payload,
            target_numel=row[
                "target_numel_per_inference"
            ],
        ),
        derive_bit_position(
            payload,
            eligible_bits=row[
                "eligible_bit_positions"
            ],
        ),
    )

    assert a == b

    assert (
        0
        <= a[
            1
        ]
        < row[
            "target_numel_per_inference"
        ]
    )

    assert a[
        2
    ] in row[
        "eligible_bit_positions"
    ]


def test_persistent_onset_is_deterministic():
    row = next(
        row
        for row in load()[
            "target_inventory"
        ]
        if row[
            "target_name"
        ]
        == "conv_2.0.weight"
    )

    payload = canonical_sampling_payload(
        partition="outer_test",
        fold=5,
        subject=61,
        task=1,
        trial=2,
        parent_kind="trial",
        window_index=None,
        representation_class=row[
            "representation_class"
        ],
        target_name=row[
            "target_name"
        ],
        target_role=row[
            "target_role"
        ],
        persistence="persistent_from_onset_until_trial_end",
        replicate_index=0,
    )

    a = derive_persistent_onset_index(
        payload,
        trial_window_count=51,
    )

    b = derive_persistent_onset_index(
        payload,
        trial_window_count=51,
    )

    assert a == b
    assert 0 <= a < 51


def test_temporal_modes_are_not_pooled():
    x = load()

    assert x[
        "temporal_modes"
    ][
        "transient_one_inference"
    ][
        "sequence_recomposition"
    ] is False

    assert x[
        "temporal_modes"
    ][
        "persistent_from_onset_until_trial_end"
    ][
        "sequence_recomposition"
    ] is True

    assert x[
        "reporting"
    ][
        "persistence_modes_pooled_in_primary_analysis"
    ] is False


def test_nonfinite_policy_is_explicit():
    x = load()[
        "nonfinite_policy"
    ]

    assert x[
        "fault_operator_sanitization_allowed"
    ] is False

    assert x[
        "nonfinite_output_must_be_recorded"
    ] is True

    assert x[
        "nonfinite_output_must_not_be_silently_coerced_to_activity_or_falling"
    ] is True


def test_outer_results_cannot_redefine_protocol():
    x = load()

    assert x[
        "partition"
    ][
        "outer_results_may_change_protocol"
    ] is False

    assert x[
        "reporting"
    ][
        "outer_result_based_metric_selection_allowed"
    ] is False

    assert x[
        "reporting"
    ][
        "outer_result_based_target_selection_allowed"
    ] is False

    assert x[
        "reporting"
    ][
        "outer_result_based_sampling_change_allowed"
    ] is False


def test_subject_level_uncertainty():
    u = load()[
        "reporting"
    ][
        "uncertainty"
    ]

    assert u[
        "resampling_unit"
    ] == "subject"

    assert u[
        "bootstrap_replicates"
    ] == 10000

    assert u[
        "checkpoint_seed_not_resampled_as_independent_subject"
    ] is True

    assert u[
        "overlapping_window_not_resampled_as_independent_observation"
    ] is True


def test_freeze_does_not_claim_execution():
    x = load()

    assert x[
        "execution_governance"
    ][
        "outer_sampling_plan_frozen"
    ] is True

    assert x[
        "execution_governance"
    ][
        "outer_model_execution_performed_in_this_freeze"
    ] is False

    assert x[
        "execution_governance"
    ][
        "outer_arrays_loaded_in_this_freeze"
    ] is False

    assert x[
        "claim_boundary"
    ][
        "CC_outer_result_exists"
    ] is False

    assert x[
        "claim_boundary"
    ][
        "CSC_result_exists"
    ] is False


def test_source_hashes_bind_sampler():
    x = load()

    assert (
        x[
            "source_hashes"
        ][
            "sampling_implementation"
        ][
            "sha256"
        ]
        == sha(
            SAMPLER
        )
    )

    assert DOC.is_file()


def test_phase5a_fault_id_is_not_claimed_parent_unique():
    x = load()

    hierarchy = x[
        "identity_hierarchy"
    ]

    assert (
        hierarchy[
            "phase5a_fault_id"
        ][
            "parent_bound"
        ]
        is False
    )

    assert (
        hierarchy[
            "phase5a_fault_id"
        ][
            "preserved_unchanged"
        ]
        is True
    )

    assert (
        x[
            "cardinality"
        ][
            "model_specific_fault_ids"
        ][
            "phase5a_fault_id_globally_unique_across_outer_parents"
        ]
        is False
    )


def test_outer_instance_id_is_parent_and_model_bound():
    x = load()

    outer = x[
        "identity_hierarchy"
    ][
        "outer_instance_id"
    ]

    assert outer[
        "parent_bound"
    ] is True

    assert outer[
        "model_variant_bound"
    ] is True

    assert outer[
        "checkpoint_seed_bound"
    ] is True

    assert outer[
        "required_primary_execution_key"
    ] is True

    assert (
        x[
            "execution_governance"
        ][
            "required_primary_execution_key"
        ]
        == "outer_instance_id"
    )


def test_outer_instance_id_deterministic_and_parent_distinct():
    from compute_fi_outer_sampling_v1 import (
        canonical_sampling_payload,
        outer_instance_id,
        sampling_instance_id,
    )

    base = dict(
        partition="outer_test",
        fold=1,
        task=1,
        trial=1,
        parent_kind="window",
        window_index=0,
        representation_class="fp32_activation",
        target_name="fc1_output_fp32",
        target_role="activation",
        persistence="transient_one_inference",
        replicate_index=0,
    )

    sampling_a = sampling_instance_id(
        canonical_sampling_payload(
            subject=1,
            **base,
        )
    )

    sampling_b = sampling_instance_id(
        canonical_sampling_payload(
            subject=2,
            **base,
        )
    )

    shared_fault_id = (
        "abcdef0123456789"
        * 4
    )

    a = outer_instance_id(
        sampling_instance_id_value=sampling_a,
        phase5a_fault_id=shared_fault_id,
        model_variant="fp32",
        checkpoint_seed=42,
    )

    a_replay = outer_instance_id(
        sampling_instance_id_value=sampling_a,
        phase5a_fault_id=shared_fault_id,
        model_variant="fp32",
        checkpoint_seed=42,
    )

    b = outer_instance_id(
        sampling_instance_id_value=sampling_b,
        phase5a_fault_id=shared_fault_id,
        model_variant="fp32",
        checkpoint_seed=42,
    )

    assert a == a_replay
    assert a != b


def test_identity_repair_does_not_change_cardinality():
    x = load()

    repair = x[
        "technical_identity_repair"
    ]

    for key in (
        "phase5a_fault_id_changed",
        "sampling_payload_changed",
        "sampling_coordinates_changed",
        "target_inventory_changed",
        "fault_family_changed",
        "eligible_bits_changed",
        "persistence_modes_changed",
        "multiplicity_changed",
        "replicate_count_changed",
        "cardinality_changed",
        "reporting_endpoints_changed",
        "outer_results_used",
        "outer_data_used",
        "outer_model_execution_used",
    ):
        assert repair[
            key
        ] is False, key

    counts = x[
        "cardinality"
    ]

    assert counts[
        "unique_sampling_ids"
    ][
        "total"
    ] == 3921946

    assert counts[
        "outer_execution_instance_ids"
    ][
        "total"
    ] == 20170008

    assert counts[
        "outer_execution_instance_ids"
    ][
        "cardinality_changed_by_repair"
    ] is False
