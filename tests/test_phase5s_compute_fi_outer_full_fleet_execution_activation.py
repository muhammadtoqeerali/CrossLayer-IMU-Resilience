from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5s_compute_fi_outer_full_fleet_execution_activation_v1.json"
)

PLAN = (
    ROOT
    / "manifests/"
    "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

PHASE5P = (
    ROOT
    / "configs/evaluation/"
    "phase5p_compute_fi_outer_full_fleet_completion_gate_v1.json"
)

PHASE5R_EXECUTOR = (
    ROOT
    / "experiments/phase_05/"
    "compute_fi_outer_fleet_executor_v1.py"
)

PHASE5R_RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/"
    "results/phase5r_compute_fi_outer_fleet_executor_qualification_v1/"
    "qualification.json"
)

PHASE5R_MANIFEST = (
    ROOT
    / "manifests/"
    "phase_5r_compute_fi_outer_full_fleet_executor_qualification_v1.json"
)

DOC = (
    ROOT
    / "docs/"
    "PHASE_5S_COMPUTE_FI_OUTER_FULL_FLEET_EXECUTION_ACTIVATION_V1.md"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status_and_activation():
    x = load(CONFIG)

    assert (
        x["status"]
        == "FROZEN_FINAL_FULL_FLEET_EXECUTION_ACTIVATION"
    )

    assert x[
        "activation"
    ][
        "full_fleet_execution_activated"
    ] is True

    assert x[
        "activation"
    ][
        "full_fleet_execution_started"
    ] is False


def test_exact_frozen_dependencies():
    x = load(CONFIG)[
        "frozen_dependencies"
    ]

    assert x[
        "phase5e_plan"
    ][
        "sha256"
    ] == sha(PLAN)

    assert x[
        "phase5p_completion_gate"
    ][
        "sha256"
    ] == sha(PHASE5P)

    assert x[
        "phase5r_executor"
    ][
        "sha256"
    ] == sha(PHASE5R_EXECUTOR)

    assert x[
        "phase5r_qualification_result"
    ][
        "sha256"
    ] == sha(PHASE5R_RESULT)

    assert x[
        "phase5r_qualification_manifest"
    ][
        "sha256"
    ] == sha(PHASE5R_MANIFEST)


def test_exact_remaining_estate():
    x = load(CONFIG)[
        "remaining_execution_estate"
    ]

    assert x[
        "fault_shards"
    ] == 731

    assert x[
        "clean_caches"
    ] == 365

    assert x[
        "outer_instance_ids"
    ] == 20145198

    assert x[
        "clean_model_window_evaluations"
    ] == 1640499

    assert x[
        "faulted_model_window_evaluations"
    ] == 29774010

    assert x[
        "total_model_window_evaluations"
    ] == 31414509


def test_complete_estate_unchanged():
    x = load(CONFIG)[
        "complete_estate_after_successful_execution"
    ]

    assert x[
        "fault_shards"
    ] == 732

    assert x[
        "clean_caches"
    ] == 366

    assert x[
        "outer_instance_ids"
    ] == 20170008

    assert x[
        "total_model_window_evaluations"
    ] == 31441800


def test_canary_is_reuse_only():
    x = load(CONFIG)

    assert x[
        "activation"
    ][
        "accepted_canary_fault_shard_reuse_only"
    ] is True

    assert x[
        "activation"
    ][
        "accepted_canary_clean_cache_reuse_only"
    ] is True

    assert x[
        "already_satisfied_estate"
    ][
        "reuse_required"
    ] is True


def test_no_selection_relaxed():
    x = load(CONFIG)[
        "selection_prohibitions"
    ]

    assert x

    assert all(
        value is False
        for value in x.values()
    )


def test_activation_is_outcome_independent():
    x = load(CONFIG)[
        "scientific_boundary"
    ]

    assert all(
        value is False
        for value in x.values()
    )


def test_execution_not_started_in_activation_freeze():
    x = load(CONFIG)[
        "execution_state_at_activation_freeze"
    ]

    assert x[
        "executed_outer_fault_shards"
    ] == 1

    assert x[
        "executed_outer_clean_caches"
    ] == 1

    assert x[
        "remaining_outer_fault_shards"
    ] == 731

    assert x[
        "full_fleet_execution_started"
    ] is False

    assert x[
        "additional_outer_payload_read_in_phase5s"
    ] is False

    assert x[
        "additional_outer_model_forward_in_phase5s"
    ] is False

    assert x[
        "additional_outer_fault_execution_in_phase5s"
    ] is False

    assert x[
        "aggregate_CC_result_generated"
    ] is False

    assert x[
        "CSC_result_generated"
    ] is False


def test_execution_guards_are_all_enabled():
    x = load(CONFIG)[
        "mandatory_execution_guards"
    ]

    assert x

    assert all(
        value is True
        for value in x.values()
    )


def test_documentation_exists():
    assert DOC.is_file()
