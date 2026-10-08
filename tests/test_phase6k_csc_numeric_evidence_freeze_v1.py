"""Phase6K metadata-evidence freeze integrity regression tests."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "manifests/phase_6k_csc_numeric_evidence_freeze_v1.json"


def load():
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def test_frozen_file_hash_bindings():
    manifest = load()

    for group in (
        "qualification_candidate_files",
        "frozen_dependency_files",
        "freeze_document_and_tests",
    ):
        assert manifest[group]

        for relative, expected in manifest[group].items():
            actual = hashlib.sha256(
                (ROOT / relative).read_bytes()
            ).hexdigest()

            assert actual == expected, relative


def test_honest_scientific_and_execution_boundaries():
    freeze = load()

    assert freeze["qualification_status"] == (
        "METADATA_NUMERIC_AND_CACHE_EVIDENCE_PASS"
    )
    assert freeze["freeze_scope"] == "EVIDENCE_ONLY"
    assert freeze["execution_authorized"] is False
    assert freeze["real_csc_execution_performed"] is False
    assert freeze["historical_phase6e_digests_reproduced"] is False
    assert freeze["phase6j_producer_aware_runtime_qualified"] is False
    assert freeze["future_execution_gate_created"] is False
    assert freeze["threshold_selection_performed"] is False

    candidate_path = (
        ROOT
        / "manifests/phase_6k_csc_numeric_evidence_candidate_v1.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert candidate["execution_authorized"] is False


def test_evidence_cardinalities_and_crosswalk():
    freeze = load()
    counts = freeze["verified_counts"]

    assert counts["subjects"] == 61
    assert counts["shards"] == 732
    assert counts["model_independent_pairs"] == 4237835
    assert counts["pair_members"] == 21793038
    assert counts["compute_faulted_member_windows"] == 411540372
    assert counts["union_member_windows"] == 426781458
    assert counts["clean_caches"] == 366

    archive = ROOT / "manifests/phase_6k_csc_numeric_evidence_v1"
    numerical = json.loads(
        (archive / "phase6k_numeric_workload_report_v1.json")
        .read_text(encoding="utf-8")
    )
    binding = json.loads(
        (archive / "phase6k_mixed_producer_cache_binding_audit_v1.json")
        .read_text(encoding="utf-8")
    )
    crosswalk = json.loads(
        (
            archive
            / "phase6k_shard_clean_cache_ownership_crosswalk_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert numerical["totals"]["pair_count"] == counts[
        "model_independent_pairs"
    ]
    assert numerical["totals"]["pair_member_count"] == counts[
        "pair_members"
    ]
    assert numerical["totals"]["member_union_window_sum"] == counts[
        "union_member_windows"
    ]

    producers = Counter(
        row["producer_kind"]
        for row in binding["rows"]
    )
    assert producers == {"canary": 1, "fleet": 365}

    assert len(crosswalk["shard_cache_ownership"]) == 732
    assert all(
        row["candidate_clean_cache_count"] == 6
        for row in crosswalk["shard_cache_ownership"]
    )
    assert crosswalk[
        "candidate_cache_references_not_all_eligible_pair_members"
    ] is True


def test_hash_change_is_detectable_without_writing():
    freeze = load()
    relative, expected = next(
        iter(sorted(freeze["qualification_candidate_files"].items()))
    )
    original = (ROOT / relative).read_bytes()

    assert hashlib.sha256(original).hexdigest() == expected
    assert hashlib.sha256(original + b"\n").hexdigest() != expected
