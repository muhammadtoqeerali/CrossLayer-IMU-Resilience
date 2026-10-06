from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5c_compute_fi_devcal_runner_v1.json"
)

RUNNER = (
    ROOT
    / "experiments"
    / "phase_05"
    / "qualify_compute_fi_devcal_runner_v1.py"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5C_COMPUTE_FI_DEVCAL_RUNNER_QUALIFICATION_V1.md"
)

MANIFEST = (
    ROOT
    / "manifests"
    / "phase_5c_compute_fi_devcal_runner_qualification_v1.json"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def config():
    return json.loads(
        CONFIG.read_text()
    )


def manifest():
    return json.loads(
        MANIFEST.read_text()
    )


def test_protocol_frozen_before_execution():
    x = config()

    assert (
        x["status"]
        == "FROZEN_PRE_EXECUTION_QUALIFICATION_PROTOCOL"
    )

    assert (
        x[
            "expected_cardinality"
        ][
            "paired_cases_total"
        ]
        == 105
    )

    assert (
        x["outer_sampling_plan_frozen"]
        is False
    )


def test_fixed_smoke_coordinates():
    fixed = config()[
        "fixed_fault_coordinates"
    ]

    assert fixed["element_index"] == 0
    assert fixed["bit_position"] == 0
    assert fixed["inference_index"] == 0
    assert fixed["multiplicity"] == 1
    assert fixed["replicate_index"] == 0

    assert (
        fixed["persistence"]
        == "transient_one_inference"
    )


def test_training_calibration_only():
    p = config()["partition"]

    assert (
        p["qualification_partition"]
        == "training_calibration"
    )

    assert (
        p["source_partition"]
        == "training only"
    )

    assert p["windows_per_fold"] == 4096

    assert p["validation_allowed"] is False
    assert p["outer_test_allowed"] is False
    assert p["onfield_allowed"] is False


def test_all15_estate_required():
    e = config()["estate"]

    assert e["member_count"] == 15

    assert e["seeds"] == [
        42,
        123,
        2025,
    ]

    assert e["folds"] == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert (
        e[
            "single_member_selection_allowed"
        ]
        is False
    )


def test_case_matrix_exact():
    x = config()

    assert len(
        x["fp32_cases_per_member"]
    ) == 2

    assert len(
        x["ptq_cases_per_member"]
    ) == 5

    counts = x[
        "expected_cardinality"
    ]

    assert counts[
        "fp32_cases_total"
    ] == 30

    assert counts[
        "ptq_cases_total"
    ] == 75

    assert counts[
        "paired_cases_total"
    ] == 105


def test_result_adaptation_prohibited():
    for value in config()[
        "prohibited_adaptation"
    ].values():
        assert value is True


def test_manifest_status_and_counts():
    x = manifest()

    assert (
        x["status"]
        == "QUALIFIED_REAL_TRAINING_CALIBRATION_PAIRED_RUNNER"
    )

    c = x[
        "qualification_counts"
    ]

    assert c["model_members"] == 15
    assert c["fp32_paired_cases"] == 30
    assert c["ptq_paired_cases"] == 75
    assert c["paired_cases_total"] == 105
    assert c["unique_fault_ids"] == 105


def test_hash_bindings():
    x = manifest()

    assert (
        x[
            "pre_execution_protocol"
        ][
            "sha256"
        ]
        == sha(CONFIG)
    )

    assert (
        x[
            "runner_source"
        ][
            "sha256"
        ]
        == sha(RUNNER)
    )

    assert (
        x[
            "documentation"
        ][
            "sha256"
        ]
        == sha(DOC)
    )


def test_serialization_repair_did_not_change_protocol():
    r = manifest()[
        "serialization_repair"
    ]

    assert (
        r[
            "model_execution_semantics_changed"
        ]
        is False
    )

    assert (
        r[
            "fault_protocol_changed"
        ]
        is False
    )

    assert (
        r[
            "qualification_case_matrix_changed"
        ]
        is False
    )

    assert (
        r[
            "partition_changed"
        ]
        is False
    )

    assert (
        r[
            "acceptance_gate_changed"
        ]
        is False
    )


def test_acceptance_all_true():
    for key, value in manifest()[
        "acceptance"
    ].items():
        assert value is True, key


def test_scientific_boundary():
    b = manifest()[
        "scientific_boundary"
    ]

    assert (
        b[
            "real_training_calibration_windows_used"
        ]
        is True
    )

    assert (
        b[
            "faulted_training_calibration_forward_executed"
        ]
        is True
    )

    for key in (
        "raw_logits_recorded",
        "probabilities_recorded",
        "accuracy_or_recall_computed",
        "threshold_applied",
        "threshold_changed",
        "fault_family_selected_from_results",
        "target_selected_from_results",
        "bit_selected_from_results",
        "element_selected_from_results",
        "checkpoint_selected_from_results",
        "replicate_count_selected_from_results",
        "validation_used",
        "outer_test_used",
        "onfield_used",
        "CC_task_performance_result_generated",
        "CSC_result_generated",
        "outer_sampling_plan_frozen",
        "physical_fault_equivalence",
        "hardware_claim",
    ):
        assert b[key] is False, key


def test_source_artifacts_immutable():
    x = manifest()

    assert x[
        "source_artifact_count"
    ] == 45

    assert (
        x[
            "source_artifacts_unchanged"
        ]
        is True
    )

    assert (
        x["outer_sampling_plan_frozen"]
        is False
    )
