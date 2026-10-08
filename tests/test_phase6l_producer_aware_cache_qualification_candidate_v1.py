"""Integrity tests for Phase6L uncommitted evidence candidate."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT /
    "manifests/phase_6l_producer_aware_cache_qualification_candidate_v1.json"
)


def load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_pinned_file_sha256_bindings():
    candidate = load()

    for field in (
        "source_dependency_sha256",
        "archived_evidence_sha256",
    ):
        assert candidate[field]
        for relative, expected in candidate[field].items():
            assert hashlib.sha256(
                (ROOT / relative).read_bytes()
            ).hexdigest() == expected, relative


def test_no_execution_authorization_or_historical_digest_claim():
    candidate = load()

    assert candidate["freeze_status"] == "UNCOMMITTED_CANDIDATE"
    assert candidate["execution_authorized"] is False
    assert candidate["execution_gate_created"] is False
    assert candidate["model_forward_performed"] is False
    assert candidate["real_csc_execution_performed"] is False
    assert candidate["prediction_rows_parsed"] is False
    assert candidate["historical_phase6e_digests_reproduced"] is False
    assert candidate["phase6j_execution_body_integrated"] is False
    assert candidate["phase6j_execution_integration_qualified"] is False
    assert candidate[
        "adversarial_in_memory_code_substitution_fully_excluded"
    ] is False


def test_archived_v1_v2_cache_equivalence():
    candidate = load()
    reports = {}

    for relative in candidate["archived_evidence_sha256"]:
        reports[Path(relative).name] = json.loads(
            (ROOT / relative).read_text(encoding="utf-8")
        )

    first = reports[
        "phase6l_actual_frozen_cache_resolver_audit_v1.json"
    ]
    second = reports[
        "phase6l_pinned_validator_cache_resolver_audit_v2.json"
    ]

    assert first["cache_count"] == second["cache_count"] == 366
    assert first["validated_output_file_count"] == 732
    assert second["validated_output_file_count"] == 732

    first_by_id = {
        row["clean_cache_id"]: row for row in first["rows"]
    }
    second_by_id = {
        row["clean_cache_id"]: row for row in second["rows"]
    }

    assert len(first_by_id) == len(second_by_id) == 366
    assert set(first_by_id) == set(second_by_id)

    counts = Counter()
    windows = 0

    for cache_id, before in first_by_id.items():
        after = second_by_id[cache_id]

        for field in (
            "fold", "subject", "model_variant",
            "checkpoint_seed", "producer_kind", "window_count"
        ):
            assert before[field] == after[field]

        assert (
            before["expected_executor_sha256"]
            == after["producer_sha256"]
        )
        assert before["execution_authorized"] is False
        assert after["execution_authorized"] is False

        counts[before["producer_kind"]] += 1
        windows += before["window_count"]

    assert counts == {"canary": 1, "fleet": 365}
    assert windows == 1642980


def test_v2_audit_does_not_claim_execution_integration():
    candidate = load()
    relative = next(
        path for path in candidate["archived_evidence_sha256"]
        if path.endswith("_v2.json")
    )
    report = json.loads((ROOT / relative).read_text(encoding="utf-8"))

    assert report["phase6j_execution_body_integrated"] is False
    assert report["caller_supplied_resolver_accepted_by_v2"] is False
    assert report["caller_supplied_phase5_core_accepted_by_v2"] is False
    assert report["execution_authorized"] is False
