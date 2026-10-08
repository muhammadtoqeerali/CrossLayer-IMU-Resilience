"""Tests for Phase6M frozen-shard pre-forward orchestration."""

from collections import Counter

import pytest

import csc_phase6m_pre_forward_orchestrator_v1 as phase6m


@pytest.fixture(scope="module")
def plan():
    return phase6m.FrozenCSCPreForwardPlan()


def test_execution_is_unconditionally_disabled():
    assert phase6m.EXECUTION_AUTHORIZED is False
    assert phase6m.MODEL_FORWARD_AUTHORIZED is False
    assert phase6m.REAL_CSC_BODY_RELEASED is False


def test_all_frozen_shards_and_caches(plan):
    assert plan.shard_count == 732
    assert plan.clean_cache_count == 366
    assert len(plan.ordered_shards()) == 732


def test_all_shard_identities_are_unique(plan):
    rows = plan.ordered_shards()
    ids = [row["runtime_shard_id"] for row in rows]

    assert len(ids) == len(set(ids)) == 732


def test_subject_and_family_coverage(plan):
    rows = plan.ordered_shards()

    subjects = Counter(row["subject"] for row in rows)
    families = Counter(row["sensor_family"] for row in rows)
    parents = Counter(row["sensor_parent_kind"] for row in rows)

    assert len(subjects) == 61
    assert set(subjects.values()) == {12}
    assert len(families) == 12
    assert set(families.values()) == {61}

    assert parents == {
        "stored_window": 305,
        "source_trial": 427,
    }


def test_frozen_workload_totals(plan):
    assert plan.numerical_totals() == phase6m.EXPECTED_TOTALS


def test_per_shard_candidate_cache_ownership(plan):
    for row in plan.ordered_shards():
        assert row["prospective_candidate_cache_count"] == 6
        assert len(
            set(row["prospective_candidate_clean_cache_ids"])
        ) == 6
        assert row["pair_member_eligibility_resolved"] is False
        assert row["execution_authorized"] is False


def test_all_shards_have_nonempty_workloads(plan):
    assert all(
        row["pair_count"] > 0
        and row["pair_member_count"] > 0
        for row in plan.ordered_shards()
    )


def test_unknown_shard_rejected(plan):
    with pytest.raises(RuntimeError, match=phase6m.UNKNOWN_SHARD):
        plan.shard("phase6h_forged_shard")


def test_shard_descriptor_is_returned_as_copy(plan):
    original = plan.ordered_shards()[0]
    shard_id = original["runtime_shard_id"]

    mutated = plan.shard(shard_id)
    mutated["prospective_candidate_clean_cache_ids"].clear()

    assert len(
        plan.shard(shard_id)["prospective_candidate_clean_cache_ids"]
    ) == 6


def test_canary_cache_belongs_to_subject_nine(plan):
    canary = phase6m.phase6l_v1.CANARY_ID

    owners = [
        row for row in plan.ordered_shards()
        if canary in row["prospective_candidate_clean_cache_ids"]
    ]

    assert len(owners) == 12
    assert {row["subject"] for row in owners} == {9}
    assert {row["fold"] for row in owners} == {5}


def test_execution_gate_cannot_override_block(tmp_path, monkeypatch):
    called = []

    def prohibited(*args, **kwargs):
        called.append("unexpected model activity")
        raise AssertionError("model activity forbidden")

    monkeypatch.setattr(
        phase6m.phase6l_v2,
        "FrozenProducerAwareCacheValidator",
        prohibited,
    )

    with pytest.raises(
        RuntimeError,
        match=phase6m.EXECUTION_BLOCK,
    ):
        phase6m.execute_shard(
            shard_id_value="forged-or-real-shard",
            dataset_root=tmp_path / "dataset",
            phase5_output_root=tmp_path / "cache",
            output_root=tmp_path / "out",
            gate_path=tmp_path / "fake-gate.json",
        )

    assert called == []


def test_no_execution_body_present():
    import inspect

    source = inspect.getsource(phase6m.execute_shard)

    assert "raise RuntimeError(EXECUTION_BLOCK)" in source
    assert "load_model_bundle" not in source
    assert "execute_fault_sequence" not in source



def test_compact_cache_reference_inherits_parent_shard_identity():
    member = {
        "model_variant": "fp32",
        "checkpoint_seed": 42,
    }

    assert phase6m._member_key(
        member,
        fold=1,
        subject=10,
    ) == (1, 10, "fp32", 42)


def test_matching_explicit_cache_identity_is_accepted():
    member = {
        "fold": 1,
        "subject": 10,
        "model_variant": "fp32",
        "checkpoint_seed": 42,
    }

    assert phase6m._member_key(
        member,
        fold=1,
        subject=10,
    ) == (1, 10, "fp32", 42)


@pytest.mark.parametrize(
    "field,incorrect",
    [
        ("fold", 99),
        ("subject", 999),
    ],
)
def test_contradictory_cache_identity_fails_closed(
    field,
    incorrect,
):
    member = {
        field: incorrect,
        "model_variant": "fp32",
        "checkpoint_seed": 42,
    }

    with pytest.raises(
        RuntimeError,
        match=phase6m.PREFLIGHT_ABORT,
    ):
        phase6m._member_key(
            member,
            fold=1,
            subject=10,
        )
