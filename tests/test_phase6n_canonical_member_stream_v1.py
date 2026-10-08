"""Phase6N complete subject canonical member metadata unit tests."""

import pytest

import csc_phase6n_canonical_member_stream_v1 as phase6n


def test_no_real_execution_capabilities_enabled():
    assert phase6n.EXECUTION_AUTHORIZED is False
    assert phase6n.MODEL_FORWARD_AUTHORIZED is False
    assert phase6n.PRODUCTION_BODY_RELEASED is False


def test_one_eligible_variant_expands_to_three_seeds():
    pair = {
        "eligible_model_variants": ["fp32"],
        "checkpoint_seeds": [42, 123, 2025],
    }

    assert phase6n.validate_member_surface(pair) == (
        ("fp32", 42),
        ("fp32", 123),
        ("fp32", 2025),
    )


def test_two_eligible_variants_expand_to_six_members():
    pair = {
        "eligible_model_variants": ["fp32", "ptq_v7"],
        "checkpoint_seeds": [42, 123, 2025],
    }

    members = phase6n.validate_member_surface(pair)

    assert len(members) == 6
    assert members[0] == ("fp32", 42)
    assert members[-1] == ("ptq_v7", 2025)


@pytest.mark.parametrize(
    "pair",
    [
        {
            "eligible_model_variants": [],
            "checkpoint_seeds": [42, 123, 2025],
        },
        {
            "eligible_model_variants": ["fp32", "fp32"],
            "checkpoint_seeds": [42, 123, 2025],
        },
        {
            "eligible_model_variants": ["unknown"],
            "checkpoint_seeds": [42, 123, 2025],
        },
        {
            "eligible_model_variants": ["fp32"],
            "checkpoint_seeds": [42, 2025],
        },
        {
            "eligible_model_variants": ["ptq_v7"],
            "checkpoint_seeds": [42, 123, 2025, 9999],
        },
    ],
)
def test_invalid_frozen_member_surface_rejected(pair):
    with pytest.raises(
        RuntimeError,
        match=phase6n.STREAM_ABORT,
    ):
        phase6n.validate_member_surface(pair)


def test_canonical_record_matches_frozen_sort_function():
    parent = phase6n.canonical_pair_identity(
        task=2,
        trial=3,
        kind="stored_window",
        parent_local_index=4,
        severity="L2",
        replay_id="replay-a",
    )

    record = {
        **parent,
        "model_variant": "fp32",
        "checkpoint_seed": 42,
    }

    assert (
        phase6n.runtime.canonical_pair_member_sort_key(record)
        == (
            0,
            0,
            2,
            3,
            4,
            1,
            "replay-a",
        )
    )


@pytest.mark.parametrize(
    "kind,index",
    [
        ("source_trial", 0),
        ("stored_window", -1),
        ("forged", 2),
    ],
)
def test_invalid_parent_index_or_kind_fails_closed(kind, index):
    with pytest.raises(
        RuntimeError,
        match=phase6n.STREAM_ABORT,
    ):
        phase6n.canonical_pair_identity(
            task=1,
            trial=1,
            kind=kind,
            parent_local_index=index,
            severity="L1",
            replay_id="test",
        )


def test_independent_new_descriptor_digest_is_stable():
    import hashlib

    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}

    assert phase6n.canonical_line(a) == phase6n.canonical_line(b)

    assert hashlib.sha256(
        phase6n.canonical_line(a)
    ).hexdigest() == hashlib.sha256(
        phase6n.canonical_line(b)
    ).hexdigest()


def test_nonqualified_subject_rejected_before_metadata_audit():
    with pytest.raises(
        RuntimeError,
        match=phase6n.STREAM_ABORT,
    ):
        phase6n.qualify_complete_subject(subject=999)


def test_wrong_pinned_source_sha_rejected(monkeypatch):
    original = phase6n.file_sha256

    def fake_hash(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(phase6n, "file_sha256", fake_hash)

        with pytest.raises(
            RuntimeError,
            match=phase6n.STREAM_ABORT,
        ):
            phase6n._check_sources()


def test_fake_execution_gate_cannot_perform_model_forward():
    with pytest.raises(
        RuntimeError,
        match=phase6n.EXECUTION_BLOCK,
    ):
        phase6n.execute_shard(
            shard_id_value="fake-shard",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
            gate_path="/unused/fake-gate",
        )


def test_no_real_model_or_fault_execution_invocation_in_source():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(phase6n.__file__).read_text(encoding="utf-8")
    )

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    forbidden = {
        "load_model_bundle",
        "execute_fault_sequence",
        "execute_bound_compute_fault_sequence",
        "condition_sensor_parent",
        "load_source_trial",
        "apply_fault",
        "run_model_member_stream",
        "synthetic_execute_pair_member",
        "execute_shard",
        "commit_phase6h_artifact",
    }

    assert not calls.intersection(forbidden)
