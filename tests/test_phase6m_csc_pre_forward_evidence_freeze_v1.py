"""Phase6M evidence-only formal freeze integrity tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FREEZE = (
    ROOT /
    "manifests/phase_6m_csc_pre_forward_evidence_freeze_v1.json"
)

CANDIDATE = (
    ROOT /
    "manifests/phase_6m_csc_pre_forward_qualification_candidate_v1.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_bindings_are_byte_exact():
    freeze = load(FREEZE)

    for field in (
        "preserved_candidate_files_sha256",
        "frozen_upstream_files_sha256",
        "freeze_document_and_test_sha256",
    ):
        assert freeze[field]

        for relative, expected in freeze[field].items():
            assert sha(ROOT / relative) == expected, relative


def test_freeze_is_evidence_only():
    freeze = load(FREEZE)

    assert freeze["freeze_scope"] == "EVIDENCE_ONLY"

    for field in (
        "execution_authorized",
        "execution_gate_created",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "production_execution_body_released",
        "historical_phase6e_digests_reproduced",
        "full_fleet_canonical_execution_qualified",
    ):
        assert freeze[field] is False, field


def test_freeze_preserves_historical_candidate():
    freeze = load(FREEZE)
    candidate = load(CANDIDATE)

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert freeze["historical_candidate_status_preserved"] is True
    assert freeze["preserved_candidate_files_sha256"][
        str(CANDIDATE.relative_to(ROOT))
    ] == sha(CANDIDATE)


def test_frozen_scope_is_not_misrepresented():
    freeze = load(FREEZE)

    counts = freeze["verified_counts"]

    assert counts["subject_family_shards"] == 732
    assert counts["validated_clean_caches"] == 366
    assert counts["representative_requests"] == 189
    assert counts["synthetic_model_member_streams"] == 69
    assert counts["frozen_model_variant_seed_pair_members"] == 21793038

    assert freeze["full_fleet_canonical_execution_qualified"] is False
    assert freeze["real_csc_execution_performed"] is False
