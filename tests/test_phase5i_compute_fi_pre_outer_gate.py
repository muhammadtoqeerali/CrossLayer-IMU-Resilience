from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GATE = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5i_compute_fi_outer_execution_gate_v1.json"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

LEGACY_GATE = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5g_compute_fi_outer_shard_executor_v1.json"
)

PHASE5D = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5d_compute_fi_outer_protocol_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5I_COMPUTE_FI_PRE_OUTER_NO_SELECTION_GATE_V1.md"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load(path):
    return json.loads(
        path.read_text()
    )


def test_gate_status_and_authorization():
    x = load(
        GATE
    )

    assert (
        x["status"]
        == "FROZEN_PROSPECTIVE_OUTER_EXECUTION_AUTHORIZATION"
    )

    assert x[
        "authorization"
    ][
        "outer_execution_authorized"
    ] is True

    assert x[
        "authorization"
    ][
        "outer_execution_performed"
    ] is False


def test_legacy_dry_run_gate_stays_false():
    x = load(
        LEGACY_GATE
    )

    assert x[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ] is False


def test_gate_binds_exact_phase5e_plan():
    gate = load(
        GATE
    )

    assert (
        gate[
            "frozen_dependencies"
        ][
            "phase5e_plan"
        ][
            "sha256"
        ]
        == sha(
            PLAN
        )
    )


def test_gate_binds_frozen_phase5d_protocol():
    gate = load(
        GATE
    )

    assert (
        gate[
            "frozen_dependencies"
        ][
            "phase5d_protocol"
        ][
            "sha256"
        ]
        == sha(
            PHASE5D
        )
    )


def test_complete_estate_cardinality():
    x = load(
        GATE
    )[
        "execution_estate"
    ]

    assert x[
        "fault_shards"
    ] == 732

    assert x[
        "clean_caches"
    ] == 366

    assert x[
        "subjects"
    ] == 61

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


def test_forward_exposure_is_exact():
    x = load(
        GATE
    )[
        "execution_estate"
    ]

    assert x[
        "clean_model_window_evaluations"
    ] == 1642980

    assert x[
        "faulted_model_window_evaluations"
    ] == 29798820

    assert x[
        "total_model_window_evaluations"
    ] == 31441800


def test_no_selection_flags_are_all_false():
    x = load(
        GATE
    )[
        "selection_prohibitions"
    ]

    assert x

    assert all(
        value is False
        for value in x.values()
    )


def test_operating_points_are_frozen():
    x = load(
        GATE
    )

    assert x[
        "operating_points"
    ] == [
        "balanced",
        "low_false_alarm",
        "timely_150ms",
    ]


def test_activation_requires_new_hash_bound_execution_config():
    x = load(
        GATE
    )

    assert x[
        "authorization"
    ][
        "activation_requires_new_execution_config_bound_to_this_gate_sha"
    ] is True


def test_phase5_scope_excludes_csc():
    x = load(
        GATE
    )[
        "scientific_scope"
    ]

    assert x[
        "regimes_generated_in_phase5"
    ] == [
        "C0",
        "CC",
    ]

    assert x[
        "CSC_generated_in_phase5"
    ] is False


def test_execution_state_is_clean_at_freeze():
    x = load(
        GATE
    )[
        "execution_state_at_freeze"
    ]

    assert all(
        value is False
        for value in x.values()
    )


def test_documentation_exists():
    assert DOC.is_file()
