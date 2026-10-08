"""Phase6O frozen record-contract safety and schema tests."""

import pytest

import csc_phase6o_synthetic_record_contract_v1 as contract


def test_execution_remains_prohibited():
    assert contract.EXECUTION_AUTHORIZED is False
    assert contract.MODEL_FORWARD_AUTHORIZED is False
    assert contract.PRODUCTION_BODY_RELEASED is False


def test_frozen_phase6h_schema_is_consistent():
    config = contract.check_pins()

    for kind, (section, field_name, version) in (
        contract.SCHEMAS.items()
    ):
        assert set(config[section]["required_fields"]) == set(
            getattr(contract.runtime, field_name)
        )
        assert config[section]["schema_version"] == version


@pytest.mark.parametrize("kind", list(contract.SCHEMAS))
def test_missing_required_field_is_rejected(kind):
    _, fields, version = contract.SCHEMAS[kind]

    row = {
        field: None
        for field in getattr(contract.runtime, fields)
    }
    row["schema_version"] = version

    contract.validate_record_shape(kind, row)

    del row[next(
        field for field in row if field != "schema_version"
    )]

    with pytest.raises(RuntimeError, match=contract.ABORT):
        contract.validate_record_shape(kind, row)


@pytest.mark.parametrize("kind", list(contract.SCHEMAS))
def test_wrong_schema_version_is_rejected(kind):
    _, fields, version = contract.SCHEMAS[kind]

    row = {
        field: None
        for field in getattr(contract.runtime, fields)
    }

    row["schema_version"] = "forged-schema"

    with pytest.raises(RuntimeError, match=contract.ABORT):
        contract.validate_record_shape(kind, row)


def test_unknown_record_kind_is_rejected():
    with pytest.raises(RuntimeError, match=contract.ABORT):
        contract.validate_record_shape("unknown", {})


def test_incorrect_binding_sha_fails_closed(monkeypatch):
    real = contract.sha

    def forged(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return real(path)

    with monkeypatch.context() as patch:
        patch.setattr(contract, "sha", forged)

        with pytest.raises(
            RuntimeError,
            match=contract.ABORT,
        ):
            contract.check_pins()


def test_forged_execution_gate_never_reaches_runtime():
    with pytest.raises(
        RuntimeError,
        match=contract.EXECUTION_BLOCK,
    ):
        contract.execute_shard(
            gate_path="forged-gate.json",
            shard_id_value="forged-shard",
            dataset_root="/unused",
            output_root="/unused",
        )


def test_frozen_runtime_execution_entrypoint_is_still_closed():
    with pytest.raises(
        RuntimeError,
        match=contract.runtime.OUTER_EXECUTION_BLOCK,
    ):
        contract.runtime.execute_shard(
            gate_path="/unused/gate.json",
            shard_id_value="forged",
            dataset_root="/unused",
            phase5_output_root="/unused",
            output_root="/unused",
        )


def test_module_does_not_call_model_or_real_fault_executor():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(contract.__file__).read_text(encoding="utf-8")
    )

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    assert not calls.intersection({
        "load_model_bundle",
        "load_bound_runtime_modules",
        "condition_sensor_parent",
        "execute_bound_compute_fault_sequence",
        "execute_fault_sequence",
        "load_source_trial",
        "prepare_phase6h_artifact",
        "commit_phase6h_artifact",
    })
