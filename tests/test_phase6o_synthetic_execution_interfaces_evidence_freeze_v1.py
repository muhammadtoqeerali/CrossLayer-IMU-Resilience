"""Formal Phase6O evidence-only freeze tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FREEZE = ROOT / (
    "manifests/"
    "phase_6o_synthetic_execution_interfaces_evidence_freeze_v1.json"
)
CANDIDATE = ROOT / (
    "manifests/"
    "phase_6o_synthetic_execution_interfaces_qualification_candidate_v1.json"
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_frozen_file_bindings():
    freeze = read(FREEZE)

    for section in (
        "preserved_phase6o_files_sha256",
        "frozen_upstream_files_sha256",
        "freeze_document_and_test_sha256",
    ):
        assert freeze[section]

        for relative, expected in freeze[section].items():
            assert sha(ROOT / relative) == expected, relative


def test_freeze_is_not_execution_authorization():
    freeze = read(FREEZE)

    assert freeze["freeze_scope"] == "EVIDENCE_ONLY"

    for key in (
        "execution_authorized",
        "execution_gate_created",
        "production_execution_body_released",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "production_output_guard_connected",
        "actual_frozen_csc_outputs_validated",
        "historical_phase6e_digests_reproduced",
    ):
        assert freeze[key] is False, key


def test_historical_candidate_status_preserved():
    freeze = read(FREEZE)
    candidate = read(CANDIDATE)

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert freeze["historical_candidate_status_preserved"] is True

    relative = str(CANDIDATE.relative_to(ROOT))
    assert freeze["preserved_phase6o_files_sha256"][relative] == (
        sha(CANDIDATE)
    )


def test_qualification_scope_is_precise():
    freeze = read(FREEZE)
    counts = freeze["verified_counts"]

    assert counts["frozen_shards"] == 732
    assert counts["phase6o_representative_requests"] == 189
    assert counts["synthetic_model_member_streams"] == 69
    assert counts["correct_frozen_coverage_claims"] == 732
    assert counts["incorrect_frozen_coverage_claims_rejected"] == 732

    assert freeze["real_csc_execution_performed"] is False
    assert freeze["production_output_guard_connected"] is False
