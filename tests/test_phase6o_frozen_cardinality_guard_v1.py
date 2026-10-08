"""Phase6O frozen-cardinality guard and synthetic attack tests."""

import pytest

import csc_phase6o_frozen_cardinality_guard_v1 as guard


@pytest.fixture(scope="module")
def plan():
    return guard.FrozenShardCardinalityPlan()


def test_execution_capabilities_remain_closed():
    assert guard.EXECUTION_AUTHORIZED is False
    assert guard.MODEL_FORWARD_AUTHORIZED is False
    assert guard.PRODUCTION_BODY_RELEASED is False
    assert guard.runtime.EXECUTION_AUTHORIZED is False


def test_all_732_frozen_shard_expectations_present(plan):
    assert len(plan.expected_by_shard) == 732
    assert plan.global_counts == {
        "pair_member_records": 21793038,
        "sensor_reference_records": 35167107,
        "csc_fault_records": 411540372,
        "simultaneous_overlap_records": 19926021,
    }


def test_valid_frozen_coverage_claim(plan):
    shard_id = next(iter(plan.expected_by_shard))
    expected = plan.expected_counts(shard_id)

    plan.check_frozen_claim(shard_id, expected)


@pytest.mark.parametrize("field", guard.COUNT_FIELDS)
def test_one_missing_or_decremented_count_rejected(plan, field):
    shard_id = next(iter(plan.expected_by_shard))
    expected = plan.expected_counts(shard_id)

    missing = dict(expected)
    del missing[field]

    with pytest.raises(RuntimeError, match=guard.ABORT):
        plan.check_frozen_claim(shard_id, missing)

    wrong = dict(expected)
    wrong[field] -= 1

    with pytest.raises(RuntimeError, match=guard.ABORT):
        plan.check_frozen_claim(shard_id, wrong)


def test_boolean_not_accepted_as_integer_count(plan):
    shard_id = next(iter(plan.expected_by_shard))
    wrong = plan.expected_counts(shard_id)

    wrong["pair_member_records"] = True

    with pytest.raises(RuntimeError, match=guard.ABORT):
        plan.check_frozen_claim(shard_id, wrong)


def test_synthetic_coverage_cannot_impersonate_frozen_shard(plan):
    shard_id = next(iter(plan.expected_by_shard))
    claimed = plan.expected_counts(shard_id)

    claimed["synthetic_fixture_only"] = True

    with pytest.raises(RuntimeError, match=guard.ABORT):
        plan.check_frozen_claim(shard_id, claimed)


def test_unknown_shard_fails_closed(plan):
    with pytest.raises(RuntimeError, match=guard.ABORT):
        plan.expected_counts("forged_shard_id")


def test_returned_expected_counts_are_defensive_copy(plan):
    shard_id = next(iter(plan.expected_by_shard))

    original = plan.expected_counts(shard_id)
    copy = plan.expected_counts(shard_id)

    copy["pair_member_records"] = 0

    assert plan.expected_counts(shard_id) == original


def test_disposable_fixture_detects_hash_valid_forgery():
    observed = guard.qualify_synthetic_fixture_attack_cases()

    checks = observed["synthetic_transition_checks"]

    assert checks["valid_fixture_pass"] == 1
    assert checks["forged_coverage_rejected"] == 1
    assert checks["physical_undercount_rejected"] == 1
    assert checks["noncanonical_line_endings_rejected"] == 1
    assert checks["restored_fixture_revalidated"] == 1

    assert observed["temporary_fixture_cleanup_verified"] is True
    assert observed["permanent_success_marker_created"] is False


def test_pinned_runtime_substitution_rejected(monkeypatch):
    real = guard.sha

    def fake(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return real(path)

    with monkeypatch.context() as patch:
        patch.setattr(guard, "sha", fake)

        with pytest.raises(RuntimeError, match=guard.ABORT):
            guard.FrozenShardCardinalityPlan()


def test_fake_execution_gate_is_ineffective():
    with pytest.raises(RuntimeError, match=guard.EXECUTION_BLOCK):
        guard.execute_shard(
            gate_path="/unused/fake-gate.json",
            shard_id_value="forged",
            dataset_root="/unused",
            output_root="/unused",
        )


def test_no_real_signal_or_model_execution_calls():
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path(guard.__file__).read_text(encoding="utf-8")
    )

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    forbidden = {
        "load_model_bundle",
        "load_bound_runtime_modules",
        "condition_sensor_parent",
        "execute_bound_compute_fault_sequence",
        "execute_fault_sequence",
        "load_source_trial",
        "load_trial_segments_and_labels",
        "run_model_member_stream",
        "synthetic_execute_pair_member",
    }

    assert not calls.intersection(forbidden)
