"""Prospective Phase6L selection and fail-closed delegation tests."""

from pathlib import Path

import pytest

import csc_phase6l_producer_aware_cache_v1 as adapter


@pytest.fixture(scope="module")
def inventory():
    return adapter.load_frozen_inventory()


def select(inventory, key, **kwargs):
    return adapter.select_frozen_cache(
        inventory,
        fold=key[0],
        subject=key[1],
        model_variant=key[2],
        checkpoint_seed=key[3],
        **kwargs,
    )


def test_execution_is_never_authorized():
    assert adapter.EXECUTION_AUTHORIZED is False
    assert adapter.MODEL_FORWARD_AUTHORIZED is False


def test_all_366_frozen_identity_routes(inventory):
    counts = {"canary": 0, "fleet": 0}
    seen_ids = set()

    for key, row in inventory["plan_by_member"].items():
        selected = select(inventory, key)

        assert selected["clean_cache_id"] == row["clean_cache_id"]
        assert selected["execution_authorized"] is False
        assert selected["clean_cache_id"] not in seen_ids

        seen_ids.add(selected["clean_cache_id"])
        counts[selected["producer_kind"]] += 1

    assert len(seen_ids) == 366
    assert counts == {"canary": 1, "fleet": 365}


def test_exact_canary_route(inventory):
    selected = select(inventory, adapter.CANARY_MEMBER)

    assert selected["clean_cache_id"] == adapter.CANARY_ID
    assert selected["producer_kind"] == "canary"
    assert selected["expected_executor_sha256"] == (
        adapter.PRODUCER_SHA["canary"]
    )


def test_fleet_route(inventory):
    selected = select(inventory, (5, 9, "fp32", 123))

    assert selected["producer_kind"] == "fleet"
    assert selected["expected_executor_sha256"] == (
        adapter.PRODUCER_SHA["fleet"]
    )


@pytest.mark.parametrize("key", [
    (5, 9, "unfrozen", 42),
    (5, 9, "fp32", 7),
    (5, 9999, "fp32", 42),
    (999, 9, "fp32", 42),
])
def test_unknown_member_fails_closed(inventory, key):
    with pytest.raises(RuntimeError, match=adapter.UNKNOWN_CACHE):
        select(inventory, key)


def test_wrong_canary_cache_id_fails_closed(inventory):
    with pytest.raises(RuntimeError, match=adapter.UNKNOWN_CACHE):
        select(
            inventory,
            adapter.CANARY_MEMBER,
            requested_cache_id="forged-canary-id",
        )


def test_wrong_fleet_cache_id_fails_closed(inventory):
    with pytest.raises(RuntimeError, match=adapter.UNKNOWN_CACHE):
        select(
            inventory,
            (5, 9, "fp32", 123),
            requested_cache_id=adapter.CANARY_ID,
        )


def test_modified_binding_fails_closed(inventory):
    key = adapter.CANARY_MEMBER
    original = inventory["binding_by_member"][key]
    modified = dict(original)
    modified["expected_executor_sha256"] = (
        adapter.PRODUCER_SHA["fleet"]
    )

    copied = dict(inventory)
    copied["binding_by_member"] = dict(inventory["binding_by_member"])
    copied["binding_by_member"][key] = modified

    with pytest.raises(RuntimeError, match=adapter.UNKNOWN_CACHE):
        select(copied, key)


def test_validation_delegates_correct_hash_without_model_loading(inventory):
    observed = []
    key = adapter.CANARY_MEMBER
    selected = select(inventory, key)

    def fake_frozen_resolver(**kwargs):
        observed.append(kwargs)
        return {
            "clean_cache_id": selected["clean_cache_id"],
            "success": {
                "status": "PASS",
                "artifact_id": selected["clean_cache_id"],
                "artifact_kind": "clean_cache",
                "plan_sha256": adapter.PLAN_SHA,
                "executor_sha256": adapter.PRODUCER_SHA["canary"],
                "coverage": {
                    "window_count": selected["window_count"],
                    "clean_model_window_evaluations":
                        selected["window_count"],
                    "labels_read": False,
                    "onfield_read": False,
                },
            },
        }

    result = adapter.validate_existing_cache(
        inventory,
        fold=key[0],
        subject=key[1],
        model_variant=key[2],
        checkpoint_seed=key[3],
        phase5_output_root=Path("/unused/synthetic/cache/root"),
        phase5_outer_core=object(),
        frozen_phase6j_resolver=fake_frozen_resolver,
    )

    assert len(observed) == 1
    assert observed[0]["expected_executor_sha256"] == (
        adapter.PRODUCER_SHA["canary"]
    )
    assert result["existing_cache_hash_validator_passed"] is True
    assert result["execution_authorized"] is False


def test_inconsistent_delegate_result_fails_closed(inventory):
    key = adapter.CANARY_MEMBER

    def bad_resolver(**kwargs):
        return {
            "clean_cache_id": adapter.CANARY_ID,
            "success": {
                "status": "PASS",
                "artifact_id": adapter.CANARY_ID,
                "artifact_kind": "clean_cache",
                "plan_sha256": adapter.PLAN_SHA,
                "executor_sha256": adapter.PRODUCER_SHA["fleet"],
                "coverage": {},
            },
        }

    with pytest.raises(RuntimeError, match=adapter.INVALID_RESOLUTION):
        adapter.validate_existing_cache(
            inventory,
            fold=key[0],
            subject=key[1],
            model_variant=key[2],
            checkpoint_seed=key[3],
            phase5_output_root="/unused",
            phase5_outer_core=object(),
            frozen_phase6j_resolver=bad_resolver,
        )


def test_missing_validator_fails_closed(inventory):
    key = adapter.CANARY_MEMBER

    with pytest.raises(RuntimeError, match=adapter.INVALID_RESOLUTION):
        adapter.validate_existing_cache(
            inventory,
            fold=key[0],
            subject=key[1],
            model_variant=key[2],
            checkpoint_seed=key[3],
            phase5_output_root="/unused",
            phase5_outer_core=object(),
            frozen_phase6j_resolver=None,
        )
