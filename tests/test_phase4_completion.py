import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMPLETION = (
    ROOT
    / "manifests/"
    "phase_4_completion_v1.json"
)

PROJECT = (
    ROOT
    / "configs/"
    "project.yaml"
)


def load():
    return json.loads(
        COMPLETION.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_phase4_completion_status_and_next_phase():
    x = load()

    assert x[
        "schema"
    ] == "crosslayer_phase4_completion_v1"

    assert x[
        "phase"
    ] == 4

    assert x[
        "status"
    ] == "COMPLETE"

    assert x[
        "phase_name"
    ] == "Sensor Fault Engine"

    assert x[
        "next_phase"
    ] == "phase_5_compute_fault_engine"


def test_project_yaml_phase4_closure_state():
    text = PROJECT.read_text()

    assert (
        "current_phase: phase_5_compute_fault_engine"
        in text
    )

    assert (
        "status: phase_4_complete"
        in text
    )

    assert (
        "sensor_faults:\n"
        "  status: frozen_phase4_sensor_fault_engine"
        in text
    )

    assert (
        "compute_faults:\n"
        "  status: candidate_taxonomy_not_frozen"
        in text
    )


def test_historical_phase2_baseline_boundary_preserved():
    text = PROJECT.read_text()

    assert (
        "status: fp32_reference_frozen"
        in text
    )

    assert (
        "primary_candidate: DATE2025_CNN_400MS_RECONSTRUCTED"
        in text
    )

    assert (
        "decision_primary: historical_streaming_0p9_strict"
        in text
    )

    assert (
        "final_int8_status: deferred_until_data_protocol_freeze"
        in text
    )

    assert (
        "target_status: candidate_not_frozen"
        in text
    )


def test_phase_boundary_is_explicit():
    x = load()

    b = x[
        "roadmap_boundary"
    ]

    assert b[
        "phase4"
    ] == "Sensor Fault Engine"

    assert b[
        "phase5"
    ] == "Compute Fault Engine"

    assert b[
        "phase6"
    ] == "Cross-Layer Vulnerability Characterization"

    assert b[
        "phase6_required_regimes"
    ] == [
        "C0",
        "CS",
        "CC",
        "CSC",
    ]

    assert b[
        "compute_fault_engine_completed_in_phase4"
    ] is False

    assert b[
        "combined_regime_completed_in_phase4"
    ] is False

    assert b[
        "compute_and_combined_work_deferred_by_roadmap"
    ] is True


def test_phase4_completion_scope():
    x = load()

    assert all(
        x[
            "completion_scope"
        ].values()
    )


def test_sensor_fi_completion_inventory():
    x = load()

    s = x[
        "sensor_fault_evidence"
    ]

    assert s[
        "provenance_tier"
    ] == "P0"

    assert s[
        "fault_family_count"
    ] == 12

    assert s[
        "heldout_outer_subjects"
    ] == 61

    assert s[
        "heldout_outer_rows"
    ] == 320616

    assert s[
        "unique_fault_instances"
    ] == 42766632

    assert s[
        "model_window_evaluations"
    ] == 479750160

    assert s[
        "outer_reporting_family_severity_records"
    ] == 4536

    assert s[
        "bootstrap_replicates"
    ] == 10000

    assert s[
        "global_binary_robustness_label"
    ] is False

    assert s[
        "heldout_results_used_for_retuning"
    ] is False


def test_onfield_completion_inventory_and_claim_boundary():
    x = load()

    o = x[
        "onfield_external_evidence"
    ]

    assert o[
        "scientific_role"
    ] == "ACTIVITY_ONLY_EXTERNAL_FIELD_EVALUATION"

    assert o[
        "retained_subjects"
    ] == 10

    assert o[
        "retained_trials"
    ] == 16

    assert o[
        "retained_windows"
    ] == 1023337

    assert o[
        "falling_windows"
    ] == 0

    assert o[
        "model_window_evaluations"
    ] == 30700110

    assert o[
        "threshold_applications"
    ] == 92100330

    assert o[
        "bootstrap_replicates"
    ] == 10000

    assert o[
        "raw_probabilities_stored"
    ] is False

    assert o[
        "fall_side_claims_supported"
    ] is False

    assert o[
        "results_used_for_retuning"
    ] is False


def test_quantization_and_hardware_boundary():
    x = load()

    d = x[
        "deployment_boundary"
    ]

    assert d[
        "ptq_v7_fully_int8"
    ] is False

    assert d[
        "hardware_target_frozen"
    ] is False

    assert d[
        "mcu_claim_made"
    ] is False


def test_scientific_guards_all_false():
    x = load()

    guards = x[
        "scientific_guards"
    ]

    assert all(
        value is False
        for value in guards.values()
    )


def test_completion_lineage_hashes_match_files():
    x = load()

    for rel, expected in x[
        "important_file_sha256"
    ].items():
        path = (
            Path(rel)
            if Path(rel).is_absolute()
            else ROOT / rel
        )

        assert path.is_file()

        assert sha(
            path
        ) == expected
