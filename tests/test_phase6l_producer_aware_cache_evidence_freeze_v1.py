"""Integrity tests for the Phase6L evidence-only freeze."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FREEZE = (
    ROOT
    / "manifests/phase_6l_producer_aware_cache_evidence_freeze_v1.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_file_sha256_bindings():
    freeze = load(FREEZE)

    for group in (
        "qualification_candidate_files",
        "frozen_dependency_files",
        "freeze_document_and_tests",
    ):
        assert freeze[group]

        for relative, expected in freeze[group].items():
            observed = hashlib.sha256(
                (ROOT / relative).read_bytes()
            ).hexdigest()
            assert observed == expected, relative


def test_execution_is_not_authorized():
    freeze = load(FREEZE)

    assert freeze["freeze_scope"] == "EVIDENCE_ONLY"
    assert freeze["execution_authorized"] is False
    assert freeze["execution_gate_created"] is False
    assert freeze["model_forward_performed"] is False
    assert freeze["real_csc_execution_performed"] is False
    assert freeze["phase6j_execution_body_integrated"] is False
    assert freeze["historical_phase6e_digests_reproduced"] is False
    assert freeze["adversarial_in_memory_substitution_fully_excluded"] is False


def test_historical_candidate_remains_unchanged():
    freeze = load(FREEZE)

    candidate = load(
        ROOT /
        "manifests/phase_6l_producer_aware_cache_qualification_candidate_v1.json"
    )

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert candidate["execution_authorized"] is False

    assert (
        freeze["qualification_candidate_files"][
            "manifests/phase_6l_producer_aware_cache_qualification_candidate_v1.json"
        ]
        == hashlib.sha256(
            (
                ROOT /
                "manifests/phase_6l_producer_aware_cache_qualification_candidate_v1.json"
            ).read_bytes()
        ).hexdigest()
    )


def test_archived_real_resolver_census():
    freeze = load(FREEZE)

    assert freeze["verified_counts"]["clean_caches"] == 366
    assert freeze["verified_counts"]["canary_caches"] == 1
    assert freeze["verified_counts"]["fleet_caches"] == 365
    assert freeze["verified_counts"]["subjects"] == 61
    assert freeze["verified_counts"]["output_hashes_per_audit"] == 732

    archive = (
        ROOT /
        "manifests/phase_6l_producer_aware_cache_evidence_v1"
    )

    v1 = load(
        archive / "phase6l_actual_frozen_cache_resolver_audit_v1.json"
    )
    v2 = load(
        archive / "phase6l_pinned_validator_cache_resolver_audit_v2.json"
    )

    assert len(v1["rows"]) == len(v2["rows"]) == 366
    assert v1["execution_authorized"] is False
    assert v2["execution_authorized"] is False
    assert v2["phase6j_execution_body_integrated"] is False
