"""Phase6N full-fleet metadata qualification candidate tests."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT /
    "manifests/phase_6n_canonical_member_qualification_candidate_v1.json"
)


def load(relative):
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8")
    )


def sha(relative):
    return hashlib.sha256(
        (ROOT / relative).read_bytes()
    ).hexdigest()


def test_all_candidate_file_bindings():
    candidate = load(CANDIDATE.relative_to(ROOT))

    for section in (
        "preserved_phase6n_files_sha256",
        "pinned_upstream_dependencies_sha256",
    ):
        for path, expected in candidate[section].items():
            assert sha(path) == expected, path


def test_candidate_does_not_authorize_execution():
    candidate = load(CANDIDATE.relative_to(ROOT))

    assert candidate["freeze_scope"] == "EVIDENCE_ONLY"
    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"

    for field in (
        "execution_authorized",
        "execution_gate_created",
        "production_execution_body_released",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "real_sensor_conditioning_performed",
        "real_compute_fault_injected",
        "historical_phase6e_digests_reproduced",
        "all_full_phase6g_requests_individually_validated",
        "full_fleet_model_forward_qualified",
    ):
        assert candidate[field] is False, field


def test_two_archived_reports_have_exact_scope():
    candidate = load(CANDIDATE.relative_to(ROOT))
    paths = candidate["preserved_phase6n_files_sha256"]

    subject_path = next(
        path for path in paths
        if path.endswith(
            "phase6n_complete_subject9_canonical_member_stream_audit_v1.json"
        )
    )

    fleet_path = next(
        path for path in paths
        if path.endswith(
            "phase6n_full_fleet_canonical_member_index_audit_v1.json"
        )
    )

    subject9 = load(subject_path)
    fleet = load(fleet_path)

    assert subject9["complete_model_independent_pair_count"] == 38107
    assert subject9["complete_pair_member_request_count"] == 195768
    assert subject9["model_member_stream_count"] == 72

    assert fleet["subject_count"] == 61
    assert fleet["shard_count"] == 732
    assert fleet["model_member_stream_count"] == 4392
    assert fleet["model_independent_pair_count"] == 4237835
    assert fleet["canonical_pair_member_count"] == 21793038
    assert fleet["phase6g_sample_request_count"] == 8784

    assert fleet["execution_authorized"] is False
    assert subject9["execution_authorized"] is False
    assert fleet["real_csc_execution_performed"] is False


def test_six_streams_per_shard():
    candidate = load(CANDIDATE.relative_to(ROOT))

    fleet_path = next(
        path for path in candidate["preserved_phase6n_files_sha256"]
        if path.endswith(
            "phase6n_full_fleet_canonical_member_index_audit_v1.json"
        )
    )

    fleet = load(fleet_path)

    counts = Counter(
        row["runtime_shard_id"]
        for row in fleet["streams"]
    )

    assert len(counts) == 732
    assert set(counts.values()) == {6}
    assert sum(counts.values()) == 4392


def test_subject9_stream_counts_reconcile():
    candidate = load(CANDIDATE.relative_to(ROOT))

    paths = candidate["preserved_phase6n_files_sha256"]

    subject_path = next(
        path for path in paths
        if path.endswith(
            "phase6n_complete_subject9_canonical_member_stream_audit_v1.json"
        )
    )

    fleet_path = next(
        path for path in paths
        if path.endswith(
            "phase6n_full_fleet_canonical_member_index_audit_v1.json"
        )
    )

    old = load(subject_path)
    new = load(fleet_path)

    def indexed(rows):
        return {
            (
                row["runtime_shard_id"],
                row["model_variant"],
                int(row["checkpoint_seed"]),
            ): row
            for row in rows
        }

    old_streams = indexed(old["streams"])
    new_streams = indexed(
        row for row in new["streams"]
        if int(row["subject"]) == 9
    )

    assert len(old_streams) == len(new_streams) == 72
    assert set(old_streams) == set(new_streams)

    for key in old_streams:
        for field in (
            "clean_cache_id",
            "pair_member_count",
            "sensor_reference_member_windows",
            "compute_faulted_member_windows",
            "simultaneous_overlap_member_windows",
            "union_member_windows",
        ):
            assert old_streams[key][field] == new_streams[key][field]
