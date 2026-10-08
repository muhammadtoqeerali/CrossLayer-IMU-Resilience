"""Phase6M evidence-only candidate binding tests."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = (
    ROOT /
    "manifests/phase_6m_csc_pre_forward_qualification_candidate_v1.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_sha256_bindings():
    candidate = load(CANDIDATE)

    for field in (
        "preserved_phase6m_files_sha256",
        "pinned_upstream_dependencies_sha256",
    ):
        assert candidate[field]

        for relative, expected in candidate[field].items():
            assert sha(ROOT / relative) == expected, relative


def test_candidate_execution_hard_boundaries():
    candidate = load(CANDIDATE)

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert candidate["freeze_scope"] == "EVIDENCE_ONLY"

    for field in (
        "execution_authorized",
        "execution_gate_created",
        "production_execution_body_released",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "real_checkpoint_loaded",
        "real_sensor_conditioning_performed",
        "real_compute_fault_injected",
        "historical_phase6e_digests_reproduced",
        "full_fleet_canonical_pair_members_emitted",
        "full_fleet_pair_member_stream_qualified",
        "full_phase6g_payloads_reconstructed_by_stream_stage",
    ):
        assert candidate[field] is False, field


def test_three_archived_report_scopes():
    candidate = load(CANDIDATE)
    paths = candidate["preserved_phase6m_files_sha256"]

    reports = {
        Path(relative).name: load(ROOT / relative)
        for relative in paths
        if relative.endswith(".json")
    }

    assert len(reports) == 3

    preflight = reports[
        "phase6m_prospective_732_shard_pre_forward_qualification_v1.json"
    ]
    witness = reports[
        "phase6m_canonical_request_synthetic_witness_v1.json"
    ]
    streams = reports[
        "phase6m_synthetic_member_stream_qualification_v1.json"
    ]

    assert preflight["shard_count"] == 732
    assert preflight["clean_cache_count"] == 366
    assert preflight["real_cache_validation"][
        "validated_output_file_hash_count"
    ] == 732

    assert len(witness["requests"]) == 189
    assert witness["distinct_pair_witness_count"] == 37

    assert len(streams["streams"]) == 69
    assert streams["synthetic_bundle_load_count"] == 69
    assert streams["synthetic_bundle_release_count"] == 69
    assert streams["distinct_request_ids_seen"] == 189

    for report in (preflight, witness, streams):
        assert report["execution_authorized"] is False
        assert report["real_csc_execution_performed"] is False


def test_stream_census_matches_witness_groups():
    candidate = load(CANDIDATE)
    paths = candidate["preserved_phase6m_files_sha256"]

    witness = load(ROOT / next(
        path for path in paths
        if path.endswith("canonical_request_synthetic_witness_v1.json")
    ))

    streams = load(ROOT / next(
        path for path in paths
        if path.endswith("synthetic_member_stream_qualification_v1.json")
    ))

    expected = Counter(
        (row["sensor_family"], row["model_variant"], row["checkpoint_seed"])
        for row in witness["requests"]
    )

    actual = {
        (row["sensor_family"], row["model_variant"], row["checkpoint_seed"]):
            row["synthetic_request_count"]
        for row in streams["streams"]
    }

    assert dict(expected) == actual
    assert sum(actual.values()) == 189
