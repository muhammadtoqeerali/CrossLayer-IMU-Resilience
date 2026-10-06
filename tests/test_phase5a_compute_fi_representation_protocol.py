from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_contract import (  # noqa: E402
    FAULT_FAMILIES,
    FP32_BIT_POSITIONS,
    FP32_EXPONENT_BITS,
    FP32_MANTISSA_BITS,
    FP32_SIGN_BITS,
    INITIAL_MULTIPLICITY,
    INT8_BIT_POSITIONS,
    FaultIdentity,
    deterministic_index,
    fp32_bit_class,
    int8_bit_class,
    protocol_capabilities,
    validate_fault_identity,
)


CONFIG = (
    ROOT
    / "configs"
    / "faults"
    / "phase5a_compute_fi_representation_protocol_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "phase_5a_compute_fi_representation_protocol_freeze_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5A_COMPUTE_FI_REPRESENTATION_PROTOCOL_V1.md"
)

SOURCE = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_contract.py"
)


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_config():
    return json.loads(
        CONFIG.read_text()
    )


def load_manifest():
    return json.loads(
        MANIFEST.read_text()
    )


def sample_identity(**overrides):
    base = dict(
        protocol="phase5a_compute_fi_representation_protocol_v1",
        model_variant="ptq_v7",
        checkpoint_seed=42,
        fold=1,
        fault_family="int8_weight_single_bit_flip",
        representation_class="int8_persistent_weight",
        target_name="conv_2.0.weight",
        target_role="persistent_parameter",
        element_index=123,
        bit_position=7,
        inference_index=5,
        persistence="transient_one_inference",
        multiplicity=1,
        replicate_index=0,
    )

    base.update(
        overrides
    )

    return FaultIdentity(
        **base
    )


def test_frozen_status_and_scope():
    x = load_config()

    assert x["phase"] == "5A"
    assert (
        x["status"]
        == "FROZEN_PRE_OPERATOR_QUALIFICATION"
    )
    assert x["provenance_tier"] == "P0"

    assert x["roadmap_alignment"]["initial_targets"] == [
        "INT8 weights",
        "activations",
        "intermediate buffers",
    ]


def test_exact_fault_families():
    x = load_config()

    assert tuple(
        x["fault_families"].keys()
    ) == FAULT_FAMILIES

    assert all(
        row["enabled_in_v1"]
        for row in x[
            "fault_families"
        ].values()
    )


def test_true_int8_weight_surface_is_conv2_only():
    x = load_config()

    w = x[
        "representation_contract"
    ][
        "int8_persistent_weight"
    ]

    assert w["eligible_target_names"] == [
        "conv_2.0.weight"
    ]

    assert w["element_count"] == 16384

    assert w["eligible_bit_positions"] == list(
        range(8)
    )

    assert (
        w["quantization_metadata_faults_in_v1"]
        is False
    )


def test_fp32_weight_faults_not_silently_pooled():
    x = load_config()

    fp = x[
        "representation_contract"
    ][
        "fp32_persistent_parameters"
    ]

    assert (
        fp[
            "initial_primary_weight_family_enabled"
        ]
        is False
    )

    assert (
        "separately labeled"
        in fp["reason"]
    )


def test_bit_spaces_are_exact():
    assert INT8_BIT_POSITIONS == tuple(
        range(8)
    )

    assert FP32_BIT_POSITIONS == tuple(
        range(32)
    )

    assert FP32_MANTISSA_BITS == tuple(
        range(23)
    )

    assert FP32_EXPONENT_BITS == tuple(
        range(23, 31)
    )

    assert FP32_SIGN_BITS == (31,)

    assert fp32_bit_class(0) == "mantissa"
    assert fp32_bit_class(22) == "mantissa"
    assert fp32_bit_class(23) == "exponent"
    assert fp32_bit_class(30) == "exponent"
    assert fp32_bit_class(31) == "sign"

    assert int8_bit_class(0) == "payload_bit"
    assert int8_bit_class(6) == "payload_bit"
    assert int8_bit_class(7) == "msb_sign"


def test_invalid_bits_are_rejected():
    with pytest.raises(ValueError):
        fp32_bit_class(32)

    with pytest.raises(ValueError):
        int8_bit_class(8)


def test_initial_multiplicity_is_one_only():
    assert INITIAL_MULTIPLICITY == 1

    identity = sample_identity()

    validate_fault_identity(
        identity
    )

    invalid = sample_identity(
        multiplicity=2
    )

    with pytest.raises(ValueError):
        validate_fault_identity(
            invalid
        )


def test_fault_id_is_canonical_and_replayable():
    a = sample_identity()
    b = sample_identity()

    validate_fault_identity(
        a
    )

    validate_fault_identity(
        b
    )

    assert a.fault_id() == b.fault_id()

    assert len(
        a.fault_id()
    ) == 64

    changed = sample_identity(
        replicate_index=1
    )

    assert (
        changed.fault_id()
        != a.fault_id()
    )


def test_deterministic_index_is_replayable():
    fields = sample_identity().canonical_dict()

    a = deterministic_index(
        namespace="element",
        upper_bound=16384,
        fields=fields,
    )

    b = deterministic_index(
        namespace="element",
        upper_bound=16384,
        fields=fields,
    )

    assert a == b
    assert 0 <= a < 16384

    c = deterministic_index(
        namespace="other",
        upper_bound=16384,
        fields=fields,
    )

    assert 0 <= c < 16384


def test_no_global_rng_requirement():
    x = load_config()

    assert (
        x[
            "fault_instance_identity"
        ][
            "global_process_rng_state_permitted"
        ]
        is False
    )


def test_pairing_contract_is_strict():
    x = load_config()

    p = x[
        "pairing_contract"
    ]

    assert p[
        "clean_and_faulted_inputs_identical"
    ]

    assert p["same_checkpoint"]
    assert p["same_preprocessing"]
    assert p["same_partition"]
    assert p["same_operating_point"]
    assert p["same_trial_order"]

    assert (
        p["changed_quantity_only"]
        == "frozen compute fault instance"
    )


def test_outer_and_onfield_are_prohibited_for_design():
    x = load_config()

    q = x[
        "qualification_scope"
    ]

    assert q[
        "synthetic_tensor_bit_exact_unit_tests_allowed"
    ]

    assert q["training_partition_allowed"]
    assert q["calibration_partition_allowed"]

    assert (
        q["validation_partition_for_protocol_tuning"]
        is False
    )

    assert q["outer_test_allowed"] is False
    assert q["onfield_allowed"] is False

    assert (
        q[
            "sensor_outer_results_allowed_for_protocol_design"
        ]
        is False
    )


def test_outer_sampling_plan_not_frozen_prematurely():
    x = load_config()

    s = x[
        "sampling_contract"
    ]

    assert s[
        "initial_multiplicity"
    ] == 1

    assert (
        s[
            "additional_multiplicity_levels_frozen"
        ]
        is False
    )

    assert (
        s[
            "outer_replicate_cardinality_frozen"
        ]
        is False
    )

    assert (
        s[
            "outer_fault_instance_count_frozen"
        ]
        is False
    )

    assert (
        s[
            "outer_sampling_plan_frozen"
        ]
        is False
    )

    assert (
        s[
            "qualification_must_precede_outer_sampling_freeze"
        ]
        is True
    )


def test_claim_boundary_is_p0_only():
    x = load_config()

    c = x[
        "claim_boundary"
    ]

    assert c[
        "software_P0_only"
    ] is True

    for key in (
        "physical_SEU_equivalence",
        "MCU_fault_equivalence",
        "register_fault_claim",
        "cache_fault_claim",
        "hardware_target_frozen",
        "CC_results_generated",
        "CSC_results_generated",
    ):
        assert c[key] is False


def test_capabilities_have_no_data_or_inference_dependency():
    caps = protocol_capabilities()

    assert caps[
        "evidence_tier"
    ] == "P0"

    assert (
        caps[
            "model_forward_pass_required"
        ]
        is False
    )

    assert caps[
        "dataset_required"
    ] is False

    assert caps[
        "outer_test_required"
    ] is False

    assert caps[
        "onfield_required"
    ] is False

    assert (
        caps[
            "physical_fault_equivalence"
        ]
        is False
    )

    assert (
        caps[
            "mcu_fault_equivalence"
        ]
        is False
    )


def test_manifest_hashes_exactly_bind_protocol():
    m = load_manifest()

    assert (
        m["status"]
        == "FROZEN_PRE_OPERATOR_QUALIFICATION"
    )

    assert (
        m["protocol"]["sha256"]
        == sha(CONFIG)
    )

    assert (
        m[
            "contract_source"
        ][
            "sha256"
        ]
        == sha(SOURCE)
    )

    assert (
        m[
            "documentation"
        ][
            "sha256"
        ]
        == sha(DOC)
    )


def test_freeze_generated_no_results():
    m = load_manifest()

    b = m[
        "scientific_boundary"
    ]

    assert (
        b[
            "model_inference_executed_for_freeze"
        ]
        is False
    )

    assert (
        b[
            "dataset_read_for_freeze"
        ]
        is False
    )

    assert b[
        "outer_test_read"
    ] is False

    assert b[
        "onfield_read"
    ] is False

    assert (
        b[
            "compute_fault_performance_result_generated"
        ]
        is False
    )

    assert b[
        "CC_result_generated"
    ] is False

    assert b[
        "CSC_result_generated"
    ] is False


def test_documentation_preserves_claim_boundary():
    text = DOC.read_text()

    assert "P0 software fault injection" in text
    assert "No CC or CSC" in text
    assert "Outer-test and OnField" in text
    assert "conv_2.0.weight" in text
