"""Regression checks for archived Phase6K metadata evidence."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "manifests/phase_6k_csc_numeric_evidence_candidate_v1.json"
)


def read_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_all_archived_evidence_hashes_match():
    manifest = read_manifest()
    assert len(manifest["evidence_files"]) == 4

    for item in manifest["evidence_files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            item["sha256"]
        )


def test_no_authorization_or_historical_digest_claim():
    manifest = read_manifest()
    assert manifest["execution_authorized"] is False
    assert manifest["real_csc_execution_performed"] is False
    assert manifest["historical_phase6e_digest_serializer_recovered"] is False
    assert manifest["historical_phase6e_binding_digests_reproduced"] is False
    assert manifest["phase6j_producer_aware_integration_pending"] is True
    assert manifest["freeze_status"] == "UNCOMMITTED_CANDIDATE"


def test_archived_workload_and_cache_ownership():
    manifest = read_manifest()

    evidence = {}
    for item in manifest["evidence_files"]:
        evidence[Path(item["path"]).name] = json.loads(
            (ROOT / item["path"]).read_text(encoding="utf-8")
        )

    numeric = evidence["phase6k_numeric_workload_report_v1.json"]
    binding = evidence[
        "phase6k_mixed_producer_cache_binding_audit_v1.json"
    ]
    crosswalk = evidence[
        "phase6k_shard_clean_cache_ownership_crosswalk_v1.json"
    ]

    assert numeric["subject_count"] == 61
    assert numeric["shard_count"] == 732
    assert numeric["totals"]["pair_count"] == 4237835
    assert numeric["totals"]["pair_member_count"] == 21793038
    assert numeric["totals"]["member_union_window_sum"] == 426781458
    assert binding["verified_cache_count"] == 366

    producers = Counter(
        row["producer_kind"]
        for row in binding["rows"]
    )
    assert producers == {"canary": 1, "fleet": 365}

    ownership = crosswalk["shard_cache_ownership"]
    assert len(ownership) == 732

    subjects = Counter(row["subject"] for row in ownership)
    assert len(subjects) == 61
    assert set(subjects.values()) == {12}

    assert all(
        len(row["candidate_clean_caches"]) == 6
        for row in ownership
    )
    assert crosswalk[
        "candidate_cache_references_not_all_eligible_pair_members"
    ] is True
