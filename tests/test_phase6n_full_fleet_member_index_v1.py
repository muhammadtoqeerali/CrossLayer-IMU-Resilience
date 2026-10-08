"""Phase6N full-fleet canonical metadata index regression tests."""

import hashlib

import pytest

import csc_phase6n_full_fleet_member_index_v1 as fleet


def test_execution_switches_are_closed():
    assert fleet.EXECUTION_AUTHORIZED is False
    assert fleet.MODEL_FORWARD_AUTHORIZED is False
    assert fleet.PRODUCTION_BODY_RELEASED is False


def test_new_pair_fingerprint_is_deterministic():
    pair = {
        "sensor_fault_id": "sensor-A",
        "sensor_replay_id": "replay-A",
        "compute_stratum_index": 2,
        "compute_coordinate": {
            "sampling_instance_id": "compute-A",
        },
        "eligible_model_variants": ["fp32"],
        "checkpoint_seeds": [42, 123, 2025],
    }

    identity = {
        "task": 1,
        "trial": 3,
        "sensor_parent_kind": "source_trial",
        "parent_local_index": -1,
        "severity": "L1",
        "sensor_replay_id": "replay-A",
    }

    def fingerprint(**changes):
        return fleet.pair_fingerprint(
            identity=identity,
            pair={**pair, **changes},
            sensor_exposure_count=3,
            compute_count=5,
            overlap_count=2,
            union_count=6,
        )

    assert fingerprint() == fingerprint()
    assert len(fingerprint()) == 32
    assert fingerprint(sensor_fault_id="sensor-B") != fingerprint()


def test_stream_header_is_member_and_cache_bound():
    common = dict(
        shard_id="frozen-shard",
        variant="fp32",
        seed=42,
        cache_id="clean-1",
        producer_sha256="f" * 64,
    )

    first = fleet.stream_prefix(**common)

    assert first == fleet.stream_prefix(**common)
    assert first != fleet.stream_prefix(
        **{**common, "seed": 123}
    )
    assert first != fleet.stream_prefix(
        **{**common, "cache_id": "clean-2"}
    )
    assert first != fleet.stream_prefix(
        **{**common, "shard_id": "different-shard"}
    )

    assert hashlib.sha256(first).hexdigest() != (
        hashlib.sha256(b"").hexdigest()
    )


def test_pair_geometry_rejects_inconsistent_union():
    identity = {
        "task": 1,
        "trial": 1,
        "sensor_parent_kind": "source_trial",
        "parent_local_index": -1,
        "severity": "L1",
        "sensor_replay_id": "replay",
    }

    pair = {
        "eligible_model_variants": ["fp32"],
        "checkpoint_seeds": [42, 123, 2025],
        "persistence":
            "persistent_from_onset_until_trial_end",
        "temporal_accounting": {
            "sensor_exposed_window_count": 2,
            "compute_active_window_count": 3,
            "temporal_overlap_window_count": 1,
            "sensor_compute_union_window_count": 99,
        },
    }

    with pytest.raises(
        RuntimeError,
        match=fleet.FLEET_ABORT,
    ):
        fleet.prepare_pair(
            (identity, pair, (0, 1), 5)
        )


def test_transient_zero_overlap_rejected():
    identity = {
        "task": 1,
        "trial": 1,
        "sensor_parent_kind": "stored_window",
        "parent_local_index": 1,
        "severity": "L1",
        "sensor_replay_id": "replay",
    }

    pair = {
        "eligible_model_variants": ["fp32"],
        "checkpoint_seeds": [42, 123, 2025],
        "persistence": "transient_one_inference",
        "temporal_accounting": {
            "sensor_exposed_window_count": 1,
            "compute_active_window_count": 1,
            "temporal_overlap_window_count": 0,
            "sensor_compute_union_window_count": 2,
        },
    }

    with pytest.raises(
        RuntimeError,
        match=fleet.FLEET_ABORT,
    ):
        fleet.prepare_pair(
            (identity, pair, (1,), 5)
        )


def test_pinned_input_mismatch_rejected(monkeypatch):
    real = fleet.file_sha256

    def corrupted(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return real(path)

    with monkeypatch.context() as patch:
        patch.setattr(fleet, "file_sha256", corrupted)

        with pytest.raises(
            RuntimeError,
            match=fleet.FLEET_ABORT,
        ):
            fleet._check_inputs()


def test_fake_gate_cannot_trigger_execution():
    with pytest.raises(
        RuntimeError,
        match=fleet.EXECUTION_BLOCK,
    ):
        fleet.execute_shard(
            shard_id_value="forged-shard",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
            gate_path="/unused/claimed-authorization.json",
        )


def test_source_has_no_real_signal_or_model_invocations():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(fleet.__file__).read_text(encoding="utf-8")
    )

    called = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    forbidden = {
        "load_model_bundle",
        "load_source_trial",
        "load_trial_segments_and_labels",
        "condition_sensor_parent",
        "execute_bound_compute_fault_sequence",
        "execute_fault_sequence",
        "run_model_member_stream",
        "synthetic_execute_pair_member",
        "commit_phase6h_artifact",
        "prepare_phase6h_artifact",
        "apply_fault",
    }

    assert not called.intersection(forbidden)
