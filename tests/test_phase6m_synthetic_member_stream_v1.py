"""Prospective Phase6M synthetic-only member stream qualification."""

import pytest

import csc_execution_runtime_v1 as runtime
import csc_phase6m_synthetic_member_stream_v1 as streams


@pytest.fixture(scope="module")
def plan():
    return streams.SyntheticMemberStreamPlan()


def test_real_execution_remains_prohibited():
    assert streams.EXECUTION_AUTHORIZED is False
    assert streams.MODEL_FORWARD_AUTHORIZED is False
    assert streams.PRODUCTION_BODY_RELEASED is False


def test_archived_witness_request_census(plan):
    assert plan.request_count == 189
    assert 12 <= plan.member_stream_count <= 72


def test_actual_frozen_stream_helper_qualification(plan):
    report = plan.exercise_synthetic_member_streams()

    assert report["qualification_status"] == (
        "ALL_WITNESS_MODEL_MEMBER_STREAMS_SYNTHETIC_PASS"
    )
    assert report["distinct_request_ids_seen"] == 189
    assert report["synthetic_pair_execution_hook_count"] == 189
    assert report["synthetic_bundle_load_count"] == (
        plan.member_stream_count
    )
    assert report["synthetic_bundle_release_count"] == (
        plan.member_stream_count
    )
    assert report["canonical_full_fleet_order_qualified"] is False
    assert report["actual_model_loaded"] is False
    assert report["execution_authorized"] is False


def test_each_stream_is_single_model_member(plan):
    for key, rows in plan._groups.items():
        family, variant, seed = key

        assert rows

        assert all(
            row["sensor_family"] == family
            and row["model_variant"] == variant
            and int(row["checkpoint_seed"]) == seed
            for row in rows
        )


def test_unknown_request_fails_closed(plan):
    with pytest.raises(
        RuntimeError,
        match=streams.STREAM_ABORT,
    ):
        plan.request("f" * 64)


def test_request_returns_defensive_copy(plan):
    request_id = next(iter(plan._requests))

    original = plan.request(request_id)
    changed = plan.request(request_id)

    changed["execution_authorized"] = True
    changed["sensor_family"] = "forged"

    assert plan.request(request_id) == original


def test_wrong_variant_rejected_before_synthetic_bundle_load():
    loaded = []

    def loader(identity):
        loaded.append(identity)
        return {"release": lambda: None}

    with pytest.raises(ValueError, match="pair job"):
        runtime.run_model_member_stream(
            fold=5,
            subject=9,
            model_variant="fp32",
            checkpoint_seed=42,
            pair_jobs=[
                {
                    "request": {
                        "model_variant": "ptq_v7",
                        "checkpoint_seed": 42,
                    }
                }
            ],
            load_model_bundle_hook=loader,
            pair_executor_hook=lambda bundle, job: {},
        )

    assert loaded == []


def test_wrong_seed_rejected_before_synthetic_bundle_load():
    loaded = []

    with pytest.raises(ValueError, match="pair job"):
        runtime.run_model_member_stream(
            fold=5,
            subject=9,
            model_variant="fp32",
            checkpoint_seed=42,
            pair_jobs=[
                {
                    "request": {
                        "model_variant": "fp32",
                        "checkpoint_seed": 2025,
                    }
                }
            ],
            load_model_bundle_hook=lambda identity: loaded.append(identity),
            pair_executor_hook=lambda bundle, job: {},
        )

    assert loaded == []


def test_synthetic_bundle_released_after_pair_exception():
    releases = []

    def loader(identity):
        return {
            "synthetic_identity": identity,
            "release": lambda: releases.append("released"),
        }

    def faulting_synthetic_hook(bundle, job):
        raise RuntimeError("SYNTHETIC_TEST_FAILURE")

    with pytest.raises(
        RuntimeError,
        match="SYNTHETIC_TEST_FAILURE",
    ):
        runtime.run_model_member_stream(
            fold=5,
            subject=9,
            model_variant="fp32",
            checkpoint_seed=42,
            pair_jobs=[
                {
                    "request": {
                        "model_variant": "fp32",
                        "checkpoint_seed": 42,
                    }
                }
            ],
            load_model_bundle_hook=loader,
            pair_executor_hook=faulting_synthetic_hook,
        )

    assert releases == ["released"]


def test_source_pin_mismatch_fails_closed(monkeypatch):
    original = streams.file_sha256

    def fake_sha(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(streams, "file_sha256", fake_sha)

        with pytest.raises(
            RuntimeError,
            match=streams.STREAM_ABORT,
        ):
            streams.SyntheticMemberStreamPlan()


def test_fake_execution_gate_cannot_execute():
    with pytest.raises(
        RuntimeError,
        match=streams.EXECUTION_BLOCK,
    ):
        streams.execute_shard(
            shard_id_value="forged-shard",
            gate_path="fake-gate.json",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
        )


def test_no_model_forward_primitive_called_by_new_source():
    import ast
    from pathlib import Path

    source = Path(streams.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    assert not calls.intersection({
        "load_model_bundle",
        "execute_fault_sequence",
        "execute_bound_compute_fault_sequence",
        "condition_sensor_parent",
        "synthetic_execute_pair_member",
        "load_source_trial",
        "execute_shard",
        "apply_fault",
    })
