"""Pre-authorization tests for Phase6K metadata workload auditor."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import csc_phase6k_metadata_fleet_auditor_v1 as audit


def pair(
    *,
    sensor,
    compute,
    overlap,
    multiplier=6,
    persistence="persistent_from_onset_until_trial_end",
):
    return {
        "persistence": persistence,
        "pair_member_multiplier": multiplier,
        "temporal_accounting": {
            "sensor_exposed_window_count": sensor,
            "compute_active_window_count": compute,
            "temporal_overlap_window_count": overlap,
            "sensor_compute_union_window_count":
                sensor + compute - overlap,
        },
    }


def test_exact_persistent_numeric_accounting():
    contribution = audit.pair_contribution(
        pair(sensor=5, compute=10, overlap=3)
    )

    assert contribution["pair_count"] == 1
    assert contribution["pair_member_count"] == 6
    assert contribution["sensor_exposed_window_sum"] == 5
    assert contribution["compute_active_window_sum"] == 10
    assert contribution["simultaneous_overlap_window_sum"] == 3
    assert contribution["sensor_compute_union_window_sum"] == 12
    assert contribution["member_sensor_reference_window_sum"] == 30
    assert contribution["member_compute_faulted_window_sum"] == 60
    assert contribution["member_overlap_window_sum"] == 18
    assert contribution["member_union_window_sum"] == 72


def test_persistent_zero_overlap_is_retained():
    contribution = audit.pair_contribution(
        pair(sensor=2, compute=7, overlap=0, multiplier=3)
    )

    assert contribution["pair_count"] == 1
    assert contribution["pair_member_count"] == 3
    assert contribution["member_overlap_window_sum"] == 0
    assert contribution["member_union_window_sum"] == 27


def test_transient_overlap_is_exactly_one():
    result = audit.pair_contribution(
        pair(
            sensor=4,
            compute=1,
            overlap=1,
            persistence="transient_one_inference",
        )
    )

    assert result["member_overlap_window_sum"] == 6

    with pytest.raises(ValueError, match="transient"):
        audit.pair_contribution(
            pair(
                sensor=4,
                compute=1,
                overlap=0,
                persistence="transient_one_inference",
            )
        )


def test_union_inconsistency_is_rejected():
    invalid = pair(
        sensor=3,
        compute=8,
        overlap=2,
    )

    invalid["temporal_accounting"][
        "sensor_compute_union_window_count"
    ] = 10

    with pytest.raises(ValueError, match="union"):
        audit.pair_contribution(invalid)


def test_frozen_numeric_mismatch_aborts():
    with pytest.raises(ValueError, match="frozen"):
        audit.assert_fields_equal(
            {"pair_count": 2},
            {"pair_count": 3},
            ("pair_count",),
            "fixture",
        )


def test_new_canonical_digest_is_deterministic():
    import hashlib

    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}

    assert audit.canonical_line(left) == audit.canonical_line(right)

    assert hashlib.sha256(
        audit.canonical_line(left)
    ).hexdigest() == hashlib.sha256(
        audit.canonical_line(right)
    ).hexdigest()


def test_execution_barriers_remain_closed():
    assert audit.csc.EXECUTION_ENABLED is False
    assert audit.runtime.EXECUTION_AUTHORIZED is False


def test_auditor_contains_no_direct_signal_or_model_load_calls():
    source = Path(audit.__file__).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    called = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            called.add(node.func.id)

        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    forbidden = {
        "load_trial_segments_and_labels",
        "load_model",
        "apply_fault",
        "torch_load",
        "np_load",
        "execute_shard",
        "run_model_member_stream",
        "condition_sensor_parent",
        "execute_bound_compute_fault_sequence",
    }

    assert not (called & forbidden)
