"""Metadata-only provenance and substitution tests for Phase6L v2."""

from pathlib import Path

import pytest

import csc_phase6l_producer_aware_cache_v1 as legacy
import csc_phase6l_producer_aware_cache_v2 as v2


@pytest.fixture(scope="module")
def validator():
    return v2.FrozenProducerAwareCacheValidator(
        phase5_output_root=Path("/unused/phase5/cache/root")
    )


def args(key):
    return {
        "fold": key[0],
        "subject": key[1],
        "model_variant": key[2],
        "checkpoint_seed": key[3],
    }


def test_v2_remains_execution_disabled():
    assert v2.EXECUTION_AUTHORIZED is False
    assert v2.MODEL_FORWARD_AUTHORIZED is False


def test_canonical_frozen_module_bindings():
    resolver, core = v2.verified_production_validators()

    import csc_execution_runtime_v1 as runtime
    import compute_fi_outer_executor_v1 as phase5core

    assert resolver is runtime.resolve_phase5_clean_cache
    assert core is phase5core
    assert runtime.EXECUTION_AUTHORIZED is False


def test_all_366_producer_routes(validator):
    from collections import Counter

    counts = Counter()
    ids = set()

    for key in validator._inventory["plan_by_member"]:
        selected = validator.select(**args(key))

        assert selected["clean_cache_id"] not in ids
        assert selected["execution_authorized"] is False
        assert selected["expected_executor_sha256"] == (
            legacy.PRODUCER_SHA[selected["producer_kind"]]
        )

        ids.add(selected["clean_cache_id"])
        counts[selected["producer_kind"]] += 1

    assert len(ids) == 366
    assert counts == {"canary": 1, "fleet": 365}


def test_accepted_canary_identity(validator):
    result = validator.select(**args(legacy.CANARY_MEMBER))

    assert result["producer_kind"] == "canary"
    assert result["clean_cache_id"] == legacy.CANARY_ID


def test_wrong_cache_identity_fails_closed(validator):
    with pytest.raises(RuntimeError, match=legacy.UNKNOWN_CACHE):
        validator.select(
            **args(legacy.CANARY_MEMBER),
            requested_cache_id="forged-cache",
        )


def test_unknown_member_fails_closed(validator):
    with pytest.raises(RuntimeError, match=legacy.UNKNOWN_CACHE):
        validator.select(
            fold=5,
            subject=9,
            model_variant="fp32",
            checkpoint_seed=9999,
        )


def test_callers_cannot_inject_a_resolver(validator):
    with pytest.raises(TypeError):
        validator.validate(
            **args(legacy.CANARY_MEMBER),
            frozen_phase6j_resolver=lambda **kw: {},
        )


def test_callers_cannot_inject_phase5_core(validator):
    with pytest.raises(TypeError):
        validator.validate(
            **args(legacy.CANARY_MEMBER),
            phase5_outer_core=object(),
        )


def test_callers_cannot_override_producer_hash(validator):
    with pytest.raises(TypeError):
        validator.validate(
            **args(legacy.CANARY_MEMBER),
            expected_executor_sha256=legacy.PRODUCER_SHA["fleet"],
        )


def test_monkeypatched_runtime_resolver_rejected(validator, monkeypatch):
    import csc_execution_runtime_v1 as runtime

    observed = []

    def forged_resolver(**kwargs):
        observed.append(kwargs)
        return {}

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime,
            "resolve_phase5_clean_cache",
            forged_resolver,
        )

        with pytest.raises(
            RuntimeError,
            match=v2.PROVENANCE_ABORT,
        ):
            validator.validate(**args(legacy.CANARY_MEMBER))

    assert observed == []


def test_monkeypatched_phase5_validator_rejected(validator, monkeypatch):
    import compute_fi_outer_executor_v1 as core

    observed = []

    def forged_validator(**kwargs):
        observed.append(kwargs)
        return {}

    with monkeypatch.context() as patch:
        patch.setattr(
            core,
            "validate_success_marker",
            forged_validator,
        )

        with pytest.raises(
            RuntimeError,
            match=v2.PROVENANCE_ABORT,
        ):
            validator.validate(**args(legacy.CANARY_MEMBER))

    assert observed == []


def test_monkeypatched_legacy_delegate_rejected(validator, monkeypatch):
    observed = []

    def forged_delegate(**kwargs):
        observed.append(kwargs)
        return {}

    with monkeypatch.context() as patch:
        patch.setattr(
            legacy,
            "validate_existing_cache",
            forged_delegate,
        )

        with pytest.raises(
            RuntimeError,
            match=v2.PROVENANCE_ABORT,
        ):
            validator.validate(**args(legacy.CANARY_MEMBER))

    assert observed == []


def test_source_hash_mismatch_rejected_without_disk_mutation(
    validator, monkeypatch
):
    original = v2._sha256_file

    def altered_hash(path):
        if path.name == "csc_execution_runtime_v1.py":
            return "0" * 64
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(v2, "_sha256_file", altered_hash)

        with pytest.raises(
            RuntimeError,
            match=v2.PROVENANCE_ABORT,
        ):
            validator.validate(**args(legacy.CANARY_MEMBER))
