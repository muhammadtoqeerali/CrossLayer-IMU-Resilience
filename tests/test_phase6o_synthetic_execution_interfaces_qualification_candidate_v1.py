"""Phase6O qualification candidate evidence tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / (
    "manifests/"
    "phase_6o_synthetic_execution_interfaces_qualification_candidate_v1.json"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_candidate_file_hashes():
    candidate = read(CANDIDATE)

    for section in (
        "preserved_phase6o_files_sha256",
        "pinned_upstream_dependencies_sha256",
    ):
        assert candidate[section]

        for relative, expected in candidate[section].items():
            assert sha(ROOT / relative) == expected, relative


def test_candidate_is_evidence_only():
    candidate = read(CANDIDATE)

    assert candidate["freeze_scope"] == "EVIDENCE_ONLY"
    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"

    for key in (
        "execution_authorized",
        "execution_gate_created",
        "production_execution_body_released",
        "real_csc_execution_performed",
        "real_model_forward_performed",
        "real_sensor_fault_applied",
        "real_compute_fault_injected",
        "real_ptq_restoration_qualified",
        "historical_phase6e_digests_reproduced",
        "full_scientific_output_validation_qualified",
        "production_output_guard_connected",
        "actual_frozen_csc_outputs_validated",
    ):
        assert candidate[key] is False, key


def test_four_qualification_report_counts():
    candidate = read(CANDIDATE)
    files = candidate["preserved_phase6o_files_sha256"]

    reports = {
        Path(relative).name: read(ROOT / relative)
        for relative in files
        if relative.endswith("_qualification_v1.json")
    }

    assert len(reports) == 4

    bridge = reports[
        "phase6o_synthetic_interface_bridge_qualification_v1.json"
    ]
    output = reports[
        "phase6o_synthetic_output_lifecycle_qualification_v1.json"
    ]
    record = reports[
        "phase6o_synthetic_record_contract_qualification_v1.json"
    ]
    guard = reports[
        "phase6o_frozen_cardinality_guard_qualification_v1.json"
    ]

    assert bridge["prior_frozen_request_ids_reconciled"] == 189
    assert bridge["synthetic_member_streams"] == 69
    assert bridge["synthetic_bundle_loads"] == 69
    assert bridge["synthetic_bundle_releases"] == 69

    assert output["transition_counters"]["synthetic_success_commits"] == 2
    assert output["temporary_fixture_cleanup_verified"] is True

    assert record["validated_request_count"] == 189
    assert record["validated_record_counts"] == {
        "pair_members": 189,
        "sensor_references": 5310,
        "csc_fault_windows": 13989,
    }

    assert guard["frozen_shard_count"] == 732
    assert guard["correct_frozen_claims_validated"] == 732
    assert guard["incorrect_frozen_claims_rejected"] == 732

    for report in reports.values():
        assert report["execution_authorized"] is False
        assert report["real_csc_execution_performed"] is False


def test_frozen_732_shard_count_reconciliation():
    candidate = read(CANDIDATE)

    path = next(
        path for path in candidate["preserved_phase6o_files_sha256"]
        if path.endswith(
            "phase6o_frozen_cardinality_guard_qualification_v1.json"
        )
    )

    guard = read(ROOT / path)
    rows = guard["per_shard_expected_counts"]

    assert len(rows) == 732

    assert len({
        row["runtime_shard_id"]
        for row in rows
    }) == 732

    for field in (
        "pair_member_records",
        "sensor_reference_records",
        "csc_fault_records",
        "simultaneous_overlap_records",
    ):
        assert sum(row[field] for row in rows) == (
            candidate["frozen_global_record_expectations"][field]
        )
