"""Phase6N formal evidence-only freeze regression tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FREEZE = (
    ROOT /
    "manifests/phase_6n_canonical_member_evidence_freeze_v1.json"
)

CANDIDATE = (
    ROOT /
    "manifests/phase_6n_canonical_member_qualification_candidate_v1.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_freeze_file_hashes():
    freeze = load(FREEZE)

    for group in (
        "preserved_qualification_files_sha256",
        "frozen_upstream_files_sha256",
        "freeze_document_and_test_sha256",
    ):
        assert freeze[group]

        for relative, expected in freeze[group].items():
            assert sha(ROOT / relative) == expected, relative


def test_freeze_never_authorizes_csc():
    freeze = load(FREEZE)

    assert freeze["freeze_scope"] == "EVIDENCE_ONLY"

    for field in (
        "execution_authorized",
        "execution_gate_created",
        "production_execution_body_released",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "historical_phase6e_digests_reproduced",
        "all_full_phase6g_requests_individually_validated",
        "real_execution_integrated_and_qualified",
    ):
        assert freeze[field] is False, field


def test_preserved_historical_candidate_status():
    freeze = load(FREEZE)
    candidate = load(CANDIDATE)

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert freeze["historical_candidate_status_preserved"] is True
    assert freeze["preserved_qualification_files_sha256"][
        str(CANDIDATE.relative_to(ROOT))
    ] == sha(CANDIDATE)


def test_freeze_is_complete_metadata_not_model_inference():
    freeze = load(FREEZE)
    counts = freeze["verified_counts"]

    assert counts["outer_subjects"] == 61
    assert counts["subject_family_shards"] == 732
    assert counts["model_independent_pairs"] == 4237835
    assert counts["eligible_variant_seed_pair_members"] == 21793038
    assert counts["canonical_model_member_streams"] == 4392
    assert counts["phase6g_sample_requests_validated"] == 8784

    assert freeze["real_csc_execution_performed"] is False
    assert freeze["all_full_phase6g_requests_individually_validated"] is False
