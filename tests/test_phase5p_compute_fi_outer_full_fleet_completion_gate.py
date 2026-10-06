from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5p_compute_fi_outer_full_fleet_completion_gate_v1.json"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

PHASE5L = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5l_compute_fi_outer_canary_execution_v1.json"
)

PHASE5O = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori"
    "/results/phase5o_compute_fi_outer_canary_technical_acceptance_v1"
    "/technical_acceptance.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5P_COMPUTE_FI_OUTER_FULL_FLEET_COMPLETION_GATE_V1.md"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def digest(values):
    return hashlib.sha256(
        json.dumps(
            values,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_status_and_authorization():
    x = load(CONFIG)

    assert (
        x["status"]
        == "FROZEN_PROSPECTIVE_FULL_FLEET_COMPLETION_AUTHORIZATION"
    )

    assert x[
        "authorization"
    ][
        "full_fleet_completion_authorized"
    ] is True

    assert x[
        "authorization"
    ][
        "full_fleet_execution_started"
    ] is False


def test_authorization_basis_is_technical_only():
    x = load(CONFIG)[
        "authorization"
    ]

    assert (
        x["authorization_basis"]
        == "PHASE5O_TECHNICAL_ACCEPTANCE_ONLY"
    )

    assert x[
        "authorization_uses_prediction_outcome"
    ] is False

    assert x[
        "authorization_uses_metric"
    ] is False

    assert x[
        "authorization_uses_threshold"
    ] is False

    assert x[
        "authorization_uses_labels"
    ] is False

    assert x[
        "authorization_uses_onfield"
    ] is False


def test_phase5l_canary_config_remains_restricted():
    x = load(PHASE5L)

    assert x[
        "authorization"
    ][
        "full_fleet_execution_authorized"
    ] is False


def test_remaining_estate_exact_plan_complement():
    gate = load(CONFIG)
    plan = load(PLAN)
    acceptance = load(PHASE5O)

    canary = acceptance[
        "artifact_integrity"
    ][
        "fault_shard"
    ][
        "artifact_id"
    ]

    clean = acceptance[
        "artifact_integrity"
    ][
        "clean_cache"
    ][
        "artifact_id"
    ]

    shard_ids = [
        row["shard_id"]
        for row in plan["shards"]
        if row["shard_id"] != canary
    ]

    clean_ids = [
        row["clean_cache_id"]
        for row in plan["clean_caches"]
        if row["clean_cache_id"] != clean
    ]

    remaining = gate[
        "remaining_authorized_estate"
    ]

    assert len(shard_ids) == 731
    assert len(clean_ids) == 365

    assert (
        remaining["remaining_shard_ids_sha256"]
        == digest(shard_ids)
    )

    assert (
        remaining["remaining_clean_cache_ids_sha256"]
        == digest(clean_ids)
    )


def test_remaining_cardinality():
    x = load(CONFIG)[
        "remaining_authorized_estate"
    ]

    assert x["fault_shards"] == 731
    assert x["clean_caches"] == 365
    assert x["outer_instance_ids"] == 20145198

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
        "complete_estate"
    ]

    assert x["fault_shards"] == 732
    assert x["clean_caches"] == 366
    assert x["subjects"] == 61
    assert x["folds"] == 5

    assert x[
        "checkpoint_seeds"
    ] == [
        42,
        123,
        2025,
    ]

    assert x[
        "outer_instance_ids"
    ] == 20170008

    assert x[
        "total_model_window_evaluations"
    ] == 31441800


def test_no_selection_prohibitions_relaxed():
    x = load(CONFIG)[
        "selection_prohibitions"
    ]

    assert x

    assert all(
        value is False
        for value in x.values()
    )


def test_no_additional_execution_in_phase5p():
    x = load(CONFIG)[
        "execution_state_at_freeze"
    ]

    assert x[
        "executed_fault_shards"
    ] == 1

    assert x[
        "executed_clean_caches"
    ] == 1

    assert x[
        "remaining_fault_shards"
    ] == 731

    assert x[
        "full_fleet_execution_started"
    ] is False

    assert x[
        "additional_outer_payload_read_in_phase5p"
    ] is False

    assert x[
        "additional_outer_model_forward_in_phase5p"
    ] is False

    assert x[
        "additional_outer_fault_execution_in_phase5p"
    ] is False

    assert x[
        "aggregate_CC_result_generated"
    ] is False

    assert x[
        "CSC_result_generated"
    ] is False


def test_new_fleet_executor_required_before_execution():
    x = load(CONFIG)[
        "authorization"
    ]

    assert x[
        "new_fleet_executor_required"
    ] is True

    assert x[
        "fleet_executor_must_bind_this_gate_sha"
    ] is True

    assert x[
        "fleet_executor_must_be_qualified_before_execution"
    ] is True


def test_documentation_exists():
    assert DOC.is_file()
