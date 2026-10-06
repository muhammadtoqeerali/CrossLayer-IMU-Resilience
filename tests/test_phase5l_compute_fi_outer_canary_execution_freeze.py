from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5l_compute_fi_outer_canary_execution_v1.json"
)

PHASE5D = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5d_compute_fi_outer_protocol_v1.json"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

FAULT_ONLY = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_fault_only_execution_v1.py"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5L_COMPUTE_FI_OUTER_CANARY_EXECUTION_FREEZE_V1.md"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_status():
    x = load(CONFIG)

    assert (
        x["status"]
        == "FROZEN_HASH_BOUND_CANARY_ONLY_OUTER_EXECUTION_CONFIG"
    )


def test_exact_single_shard_authorization():
    auth = load(CONFIG)[
        "authorization"
    ]

    assert auth[
        "canary_execution_authorized"
    ] is True

    assert auth[
        "canary_execution_performed"
    ] is False

    assert auth[
        "full_fleet_execution_authorized"
    ] is False

    assert auth[
        "authorized_shard_count"
    ] == 1

    assert auth[
        "authorized_clean_cache_count"
    ] == 1


def test_authorized_shard_is_frozen_plan_zero():
    x = load(CONFIG)
    plan = load(PLAN)

    assert (
        x["authorization"]["authorized_shard_id"]
        == plan["shards"][0]["shard_id"]
    )

    assert x["canary"]["shard_index"] == 0


def test_fixed_canary_identity():
    x = load(CONFIG)[
        "canary"
    ][
        "shard"
    ]

    assert x["subject"] == 9
    assert x["fold"] == 5
    assert x["model_variant"] == "fp32"
    assert x["checkpoint_seed"] == 42
    assert x["persistence"] == "transient_one_inference"
    assert x["target_count"] == 10


def test_target_binding_uses_actual_phase5d_schema():
    cfg = load(CONFIG)
    phase5d = load(PHASE5D)

    expected = [
        row
        for row in phase5d["target_inventory"]
        if "fp32" in row["model_variants"]
    ]

    assert len(expected) == 10

    assert (
        cfg["canary"]["target_inventory"]
        == expected
    )

    assert (
        cfg["canary"]["target_schema_binding"]
        == "PHASE5D_TARGET_INVENTORY_MODEL_VARIANTS_CONTAINS_FP32"
    )


def test_exact_forward_budget():
    x = load(CONFIG)[
        "forward_budget"
    ]

    assert x[
        "clean_model_window_evaluations"
    ] == 2481

    assert x[
        "faulted_model_window_evaluations"
    ] == 24810

    assert x[
        "total_model_window_evaluations"
    ] == 27291

    assert x[
        "embedded_clean_reference_forward_per_fault"
    ] is False


def test_fault_only_module_hash_binding():
    x = load(CONFIG)

    assert (
        x[
            "frozen_dependencies"
        ][
            "phase5k_fault_only_module"
        ][
            "sha256"
        ]
        == sha(FAULT_ONLY)
    )


def test_outcome_cannot_expand_authorization():
    x = load(CONFIG)

    assert x[
        "authorization"
    ][
        "outcome_may_not_expand_authorization"
    ] is True

    post = x[
        "post_canary_governance"
    ]

    assert post[
        "successful_canary_does_not_itself_authorize_full_fleet"
    ] is True

    assert post[
        "canary_outcome_may_be_used_to_select_science"
    ] is False


def test_clean_freeze_state():
    x = load(CONFIG)[
        "execution_state_at_freeze"
    ]

    assert all(
        value is False
        for value in x.values()
    )


def test_documentation_exists():
    assert DOC.is_file()
