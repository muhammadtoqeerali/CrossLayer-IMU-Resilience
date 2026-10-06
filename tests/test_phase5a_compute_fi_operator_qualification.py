from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "manifests"
    / "phase_5a_compute_fi_operator_qualification_v1.json"
)

FREEZE_CONFIG = (
    ROOT
    / "configs"
    / "faults"
    / "phase5a_compute_fi_representation_protocol_v1.json"
)

FREEZE_MANIFEST = (
    ROOT
    / "manifests"
    / "phase_5a_compute_fi_representation_protocol_freeze_v1.json"
)

OPERATOR_SOURCE = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_operators.py"
)

QUAL_SOURCE = (
    ROOT
    / "experiments"
    / "phase_05"
    / "qualify_compute_fi_operators_v1.py"
)

OPERATOR_TEST = (
    ROOT
    / "tests"
    / "test_phase5a_compute_fi_operators.py"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5A_COMPUTE_FI_OPERATOR_QUALIFICATION_V1.md"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def manifest():
    return json.loads(
        MANIFEST.read_text()
    )


def test_status_is_qualified_p0():
    x = manifest()

    assert (
        x["status"]
        == "QUALIFIED_BIT_EXACT_P0_OPERATORS"
    )

    assert x["provenance_tier"] == "P0"


def test_frozen_protocol_hash_is_preserved():
    x = manifest()

    assert (
        x[
            "frozen_representation_protocol"
        ][
            "sha256"
        ]
        == sha(
            FREEZE_CONFIG
        )
    )

    assert (
        x[
            "frozen_representation_manifest"
        ][
            "sha256"
        ]
        == sha(
            FREEZE_MANIFEST
        )
    )


def test_source_hashes_are_exact():
    x = manifest()

    assert (
        x[
            "operator_source"
        ][
            "sha256"
        ]
        == sha(
            OPERATOR_SOURCE
        )
    )

    assert (
        x[
            "qualification_source"
        ][
            "sha256"
        ]
        == sha(
            QUAL_SOURCE
        )
    )

    assert (
        x[
            "operator_regression_test"
        ][
            "sha256"
        ]
        == sha(
            OPERATOR_TEST
        )
    )

    assert (
        x[
            "documentation"
        ][
            "sha256"
        ]
        == sha(
            DOC
        )
    )


def test_qualification_counts_are_frozen():
    q = manifest()[
        "qualification_counts"
    ]

    assert q[
        "int8_scalar_cases"
    ] == 2048

    assert q[
        "qint8_tensor_cases"
    ] == 128

    assert q[
        "quint8_tensor_cases"
    ] == 128

    assert q[
        "fp32_tensor_cases"
    ] == 512

    assert q[
        "real_ptq_cases"
    ] == 4

    assert q[
        "real_fp32_tensor_cases"
    ] == 4


def test_operator_acceptance_is_complete():
    a = manifest()[
        "acceptance"
    ]

    required_true = (
        "exactly_one_selected_payload_bit_changes",
        "all_nonselected_payload_elements_unchanged",
        "double_flip_restores_exact_payload",
        "source_tensor_unchanged",
        "quantized_metadata_preserved",
        "FP32_nan_inf_not_sanitized",
        "transient_schedule_exact",
        "persistent_schedule_exact",
        "deterministic_target_mapping",
    )

    for key in required_true:
        assert a[key] is True

    assert (
        a[
            "cross_trial_state_leakage"
        ]
        is False
    )


def test_no_task_evaluation_occurred():
    b = manifest()[
        "scientific_boundary"
    ]

    assert b[
        "dataset_read"
    ] is False

    assert b[
        "model_forward_passes"
    ] == 0

    assert b[
        "faulted_model_inference"
    ] is False

    assert b[
        "outer_test_read"
    ] is False

    assert b[
        "onfield_read"
    ] is False

    assert b[
        "CC_result_generated"
    ] is False

    assert b[
        "CSC_result_generated"
    ] is False

    assert b[
        "physical_fault_equivalence"
    ] is False

    assert b[
        "hardware_claim"
    ] is False


def test_outer_sampling_still_not_frozen():
    assert (
        manifest()[
            "outer_sampling_plan_frozen"
        ]
        is False
    )
