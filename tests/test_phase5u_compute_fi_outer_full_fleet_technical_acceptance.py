from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5u_compute_fi_outer_full_fleet_technical_acceptance_v1/technical_acceptance.json"
)

PLAN = (
    ROOT
    / "manifests/"
    "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

PHASE5R_EXECUTOR = (
    ROOT
    / "experiments/phase_05/"
    "compute_fi_outer_fleet_executor_v1.py"
)

PHASE5S_ACTIVATION = (
    ROOT
    / "configs/evaluation/"
    "phase5s_compute_fi_outer_full_fleet_execution_activation_v1.json"
)

DOC = (
    ROOT
    / "docs/"
    "PHASE_5U_COMPUTE_FI_OUTER_FULL_FLEET_TECHNICAL_ACCEPTANCE_V1.md"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status():
    x = load(RESULT)

    assert (
        x["status"]
        == "TECHNICALLY_ACCEPTED_COMPLETE_PHASE5E_ESTATE"
    )


def test_complete_estate_cardinality():
    x = load(RESULT)[
        "complete_estate"
    ]

    assert x["fault_shards"] == 732
    assert x["clean_caches"] == 366
    assert x["artifact_directories"] == 1098
    assert x["outer_instance_ids"] == 20170008
    assert x["clean_record_count"] == 1642980
    assert x["fault_record_count"] == 29798820
    assert x["total_record_count"] == 31441800


def test_lineage_hashes():
    x = load(RESULT)[
        "frozen_lineage"
    ]

    assert (
        x["phase5e_plan_sha256"]
        == sha(PLAN)
    )

    assert (
        x["phase5r_executor_sha256"]
        == sha(PHASE5R_EXECUTOR)
    )

    assert (
        x["phase5s_activation_sha256"]
        == sha(PHASE5S_ACTIVATION)
    )


def test_technical_checks_all_pass():
    x = load(RESULT)[
        "technical_checks"
    ]

    assert x
    assert all(
        value is True
        for value in x.values()
    )


def test_no_prediction_deserialization_or_outcome_analysis():
    x = load(RESULT)[
        "inspection_boundary"
    ]

    assert x
    assert all(
        value is False
        for value in x.values()
    )


def test_no_scientific_mutation():
    x = load(RESULT)[
        "scientific_governance"
    ]

    assert x
    assert all(
        value is False
        for value in x.values()
    )


def test_mixed_executor_provenance_explicit():
    x = load(RESULT)[
        "executor_provenance"
    ]

    assert x[
        "accepted_canary_fault_shards"
    ] == 1

    assert x[
        "accepted_canary_clean_caches"
    ] == 1

    assert x[
        "phase5r_fault_shards"
    ] == 731

    assert x[
        "phase5r_clean_caches"
    ] == 365

    assert x[
        "accepted_canary_reverified_from_phase5o"
    ] is True

    assert x[
        "phase5r_artifacts_success_marker_verified"
    ] is True


def test_documentation_exists():
    assert DOC.is_file()


def test_frozen_phase5r_regression_artifact_remains_hash_consistent():
    phase5r_test = (
        ROOT
        / "tests/"
        "test_phase5r_compute_fi_outer_full_fleet_executor.py"
    )

    phase5r_manifest = load(
        ROOT
        / "manifests/"
        "phase_5r_compute_fi_outer_full_fleet_executor_qualification_v1.json"
    )

    assert (
        phase5r_manifest[
            "regression_test"
        ][
            "sha256"
        ]
        == sha(phase5r_test)
    )
