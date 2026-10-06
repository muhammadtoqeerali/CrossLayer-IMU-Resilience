from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "manifests"
    / "phase_5b_compute_fi_execution_harness_qualification_v1.json"
)

HARNESS = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_execution_harness.py"
)

QUALIFIER = (
    ROOT
    / "experiments"
    / "phase_05"
    / "qualify_compute_fi_execution_harness_v1.py"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5B_COMPUTE_FI_EXECUTION_HARNESS_QUALIFICATION_V1.md"
)

PHASE5A_CONFIG = (
    ROOT
    / "configs"
    / "faults"
    / "phase5a_compute_fi_representation_protocol_v1.json"
)

PHASE5A_QUAL = (
    ROOT
    / "manifests"
    / "phase_5a_compute_fi_operator_qualification_v1.json"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load():
    return json.loads(
        MANIFEST.read_text()
    )


def test_status():
    x = load()

    assert (
        x["status"]
        == "QUALIFIED_PAIRED_SYNTHETIC_EXECUTION_HARNESS"
    )

    assert x["provenance_tier"] == "P0"


def test_phase5a_bindings_are_exact():
    x = load()

    assert (
        x[
            "phase5a_protocol"
        ][
            "sha256"
        ]
        == sha(
            PHASE5A_CONFIG
        )
    )

    assert (
        x[
            "phase5a_operator_qualification"
        ][
            "sha256"
        ]
        == sha(
            PHASE5A_QUAL
        )
    )


def test_harness_artifacts_are_hash_bound():
    x = load()

    assert (
        x[
            "harness_source"
        ][
            "sha256"
        ]
        == sha(
            HARNESS
        )
    )

    assert (
        x[
            "qualification_source"
        ][
            "sha256"
        ]
        == sha(
            QUALIFIER
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


def test_all_execution_cases_are_present():
    x = load()

    assert set(
        x[
            "qualified_execution_cases"
        ]
    ) == {
        "fp32_variant_fp32_activation",
        "fp32_variant_fp32_buffer",
        "ptq_variant_fp32_activation",
        "ptq_variant_fp32_buffer",
        "ptq_variant_qint8_weight",
        "ptq_variant_quint8_activation",
        "ptq_variant_quint8_buffer",
    }


def test_representation_binding_is_exact():
    x = load()

    binding = x[
        "representation_binding"
    ]

    assert (
        binding[
            "qint8_weight"
        ]
        == "torch.qint8 / torch.int8"
    )

    assert (
        binding[
            "quint8_activation_buffer"
        ]
        == "torch.quint8 / torch.uint8"
    )


def test_acceptance_is_complete():
    x = load()

    a = x[
        "acceptance"
    ]

    for key in (
        "same_input_clean_fault_pair",
        "same_checkpoint_clean_fault_pair",
        "clean_ptq_eager_matches_frozen_torchscript",
        "all_five_frozen_fault_families_executed",
        "fp32_family_executed_on_fp32_variant",
        "fp32_family_executed_on_ptq_mixed_precision_regions",
        "qint8_weight_execution_qualified",
        "quint8_activation_execution_qualified",
        "quint8_buffer_execution_qualified",
        "exact_single_element_single_bit_mutation_recorded",
        "deterministic_fault_replay",
        "transient_restore_after_call",
        "persistent_onset_schedule_exact",
        "new_trial_reset_clean",
        "source_artifacts_unchanged",
    ):
        assert a[
            key
        ] is True


def test_scientific_boundary_is_synthetic_only():
    x = load()

    b = x[
        "scientific_boundary"
    ]

    assert b[
        "synthetic_inputs_only"
    ] is True

    assert (
        b[
            "faulted_synthetic_model_forward_executed"
        ]
        is True
    )

    for key in (
        "dataset_read",
        "training_partition_read",
        "calibration_partition_read",
        "validation_partition_read",
        "outer_test_read",
        "onfield_read",
        "faulted_real_dataset_inference_executed",
        "CC_task_performance_result_generated",
        "CSC_result_generated",
        "outer_sampling_plan_frozen",
        "physical_fault_equivalence",
        "hardware_claim",
    ):
        assert b[
            key
        ] is False


def test_outer_plan_still_unfrozen():
    assert (
        load()[
            "outer_sampling_plan_frozen"
        ]
        is False
    )
