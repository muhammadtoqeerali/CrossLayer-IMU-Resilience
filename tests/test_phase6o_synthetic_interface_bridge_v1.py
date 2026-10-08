"""Phase6O synthetic execution-interface bridge regression tests."""

import pytest

import csc_phase6o_synthetic_interface_bridge_v1 as bridge


def test_execution_switches_stay_closed():
    assert bridge.EXECUTION_AUTHORIZED is False
    assert bridge.MODEL_FORWARD_AUTHORIZED is False
    assert bridge.PRODUCTION_BODY_RELEASED is False


def test_fake_gate_cannot_authorize_real_execution():
    with pytest.raises(
        RuntimeError,
        match=bridge.EXECUTION_BLOCK,
    ):
        bridge.execute_shard(
            gate_path="/unused/claimed-gate.json",
            shard_id_value="forged-shard",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
        )


def test_pinned_source_substitution_is_rejected(monkeypatch):
    original = bridge.file_sha256

    def fake_sha(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(bridge, "file_sha256", fake_sha)

        with pytest.raises(
            RuntimeError,
            match=bridge.BRIDGE_ABORT,
        ):
            bridge.check_pins()


def test_stored_window_synthetic_conditioning():
    witness = {
        "sensor_parent_kind": "stored_window",
        "sensor_exposed_window_indices": (2,),
        "trial_window_count": 5,
        "parent_local_index": 2,
        "sensor_family": "noise",
        "sensor_fault_id": "synthetic-fault",
        "sensor_replay_id": "synthetic-replay",
    }

    events = []

    conditioned, audit = bridge.synthetic_condition(
        witness,
        events=events,
    )

    assert set(conditioned) == {2}
    assert conditioned[2].startswith("SYNTHETIC_CONDITIONED:")
    assert audit["operator_calls"] == 1
    assert audit["rewindow_calls"] == 0
    assert events == ["synthetic_sensor_operator"]


def test_source_trial_synthetic_conditioning():
    witness = {
        "sensor_parent_kind": "source_trial",
        "sensor_exposed_window_indices": (1, 3),
        "trial_window_count": 5,
        "parent_local_index": -1,
        "sensor_family": "drift",
        "sensor_fault_id": "synthetic-source-fault",
        "sensor_replay_id": "synthetic-source-replay",
    }

    events = []

    conditioned, audit = bridge.synthetic_condition(
        witness,
        events=events,
    )

    assert set(conditioned) == {1, 3}
    assert audit["operator_calls"] == 1
    assert audit["rewindow_calls"] == 1
    assert events == [
        "synthetic_sensor_operator",
        "synthetic_rewindow",
    ]


def test_synthetic_operator_rejects_nonsynthetic_parent():
    operator = bridge.SyntheticOperators([])

    with pytest.raises(
        RuntimeError,
        match=bridge.BRIDGE_ABORT,
    ):
        operator.apply_fault(
            object(),
            {
                "synthetic_operator_fixture": True,
                "family": "noise",
            },
        )

    assert operator.calls == 0


def test_synthetic_phase5_rejects_nonsynthetic_model():
    executor = bridge.SyntheticPhase5Executor([])

    with pytest.raises(
        RuntimeError,
        match=bridge.BRIDGE_ABORT,
    ):
        executor.execute_fault_sequence(
            model=object(),
            inputs=["SYNTHETIC_INPUT"],
            inference_indices=[0],
            identities=["SYNTHETIC_COMPUTE_IDENTITY"],
        )

    assert executor.sequence_calls == 0


def test_synthetic_phase5_rejects_extra_compute_identity():
    executor = bridge.SyntheticPhase5Executor([])

    with pytest.raises(
        RuntimeError,
        match=bridge.BRIDGE_ABORT,
    ):
        executor.execute_fault_sequence(
            model="SYNTHETIC_MODEL_BUNDLE",
            inputs=["SYNTHETIC_INPUT"],
            inference_indices=[0],
            identities=[
                "SYNTHETIC_COMPUTE_IDENTITY",
                "SYNTHETIC_COMPUTE_IDENTITY",
            ],
        )


def test_synthetic_exposure_contract_rejects_real_fixture():
    helper = bridge.SyntheticExposure([0])

    with pytest.raises(AssertionError):
        helper.source_trial_exposed_window_indices(
            instance={"synthetic_operator_fixture": False},
            historical_window_ends=[1],
            source_length=1,
        )


def test_bridge_contains_no_real_heavy_runtime_import():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(bridge.__file__).read_text(encoding="utf-8")
    )

    called = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    forbidden = {
        "load_bound_runtime_modules",
        "load_model_bundle",
        "load_trial_segments_and_labels",
        "load_source_trial",
        "prepare_phase6h_artifact",
        "commit_phase6h_artifact",
        "execute_shard",
    }

    assert not called.intersection(forbidden)


def test_frozen_runtime_public_entrypoint_remains_blocked():
    assert bridge.runtime.EXECUTION_AUTHORIZED is False

    with pytest.raises(
        RuntimeError,
        match=bridge.runtime.OUTER_EXECUTION_BLOCK,
    ):
        bridge.runtime.execute_shard(
            gate_path="/unused/fake-gate",
            shard_id_value="forged",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
        )
