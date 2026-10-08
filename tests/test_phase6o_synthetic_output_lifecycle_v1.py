"""Phase6O synthetic output atomicity and safety tests."""

import json

import pytest

import csc_phase6o_synthetic_output_lifecycle_v1 as outputs


def test_production_execution_is_always_disabled():
    assert outputs.EXECUTION_AUTHORIZED is False
    assert outputs.MODEL_FORWARD_AUTHORIZED is False
    assert outputs.PRODUCTION_BODY_RELEASED is False


def test_frozen_artifact_names_are_exact():
    assert set(outputs.runtime.OUTPUT_FILES) == {
        "pair_members.jsonl",
        "sensor_reference.jsonl",
        "csc_fault.jsonl",
        "metadata.json",
    }


def test_synthetic_fixture_explicitly_disclaims_predictions():
    for name in outputs.runtime.OUTPUT_FILES:
        data = json.loads(
            outputs.synthetic_output_bytes(name).decode("utf-8")
        )

        assert data["synthetic_fixture"] is True
        assert data["qualification_only"] is True
        assert data["scientific_predictions_present"] is False
        assert data["execution_authorized"] is False
        assert data["output_name"] == name


def test_nonfrozen_filename_rejected():
    with pytest.raises(ValueError):
        outputs.synthetic_output_bytes("predictions.csv")


def test_pinned_runtime_substitution_fails_closed(monkeypatch):
    real_sha = outputs.file_sha256

    def wrong_sha(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return real_sha(path)

    with monkeypatch.context() as patch:
        patch.setattr(outputs, "file_sha256", wrong_sha)

        with pytest.raises(
            RuntimeError,
            match=outputs.OUTPUT_ABORT,
        ):
            outputs.check_pins()


def test_complete_synthetic_atomic_lifecycle_and_cleanup():
    report = outputs.qualify_synthetic_output_lifecycle()

    assert report["qualification_status"] == (
        "FROZEN_PHASE6H_SYNTHETIC_OUTPUT_LIFECYCLE_PASS"
    )
    assert report["frozen_output_file_count"] == 4
    assert report["temporary_fixture_cleanup_verified"] is True
    assert report["permanent_phase6h_success_marker_created"] is False
    assert report["real_csc_execution_performed"] is False
    assert report["execution_authorized"] is False

    counts = report["transition_counters"]

    assert counts["synthetic_success_commits"] == 2
    assert counts["marker_before_rename_checks"] == 1
    assert counts["valid_success_reuse_checks"] == 1
    assert counts["tamper_rejection_checks"] == 1
    assert counts["wrong_gate_rejection_checks"] == 1
    assert counts["recompute_required_checks"] == 1
    assert counts["explicit_recompute_checks"] == 1
    assert counts["incorrect_file_hash_rejections"] == 1
    assert counts["incorrect_output_set_rejections"] == 1
    assert counts["partial_interruption_rejections"] == 1
    assert counts["partial_recovery_checks"] == 1


def test_fake_authorization_cannot_trigger_execution():
    with pytest.raises(
        RuntimeError,
        match=outputs.EXECUTION_BLOCK,
    ):
        outputs.execute_shard(
            gate_path="/unused/fake-gate.json",
            dataset_root="/unused",
            shard_id_value="forged",
            output_root="/unused",
        )


def test_frozen_runtime_entrypoint_remains_blocked():
    with pytest.raises(
        RuntimeError,
        match=outputs.runtime.OUTER_EXECUTION_BLOCK,
    ):
        outputs.runtime.execute_shard(
            gate_path="/unused/fake-gate.json",
            shard_id_value="forged",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
        )


def test_module_has_no_model_forward_or_signal_reader_calls():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(outputs.__file__).read_text(encoding="utf-8")
    )

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    forbidden = {
        "load_model_bundle",
        "load_trial_segments_and_labels",
        "load_source_trial",
        "execute_fault_sequence",
        "condition_sensor_parent",
        "execute_bound_compute_fault_sequence",
        "run_model_member_stream",
        "load_bound_runtime_modules",
    }

    assert not calls.intersection(forbidden)
