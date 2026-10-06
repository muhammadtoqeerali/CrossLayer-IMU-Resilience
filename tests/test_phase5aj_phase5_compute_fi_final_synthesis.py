from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5aj_phase5_compute_fi_final_synthesis_v1.json"
)

RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5aj_phase5_compute_fi_final_synthesis_v1/final_synthesis.json")
SUMMARY_MD = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5aj_phase5_compute_fi_final_synthesis_v1/PHASE5_FINAL_SYNTHESIS.md")

AF_SUCCESS = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5af_compute_fi_cc_outcome_execution_v1/_SUCCESS.json")
AG_RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ag_compute_fi_cc_outcome_technical_acceptance_v1/technical_acceptance.json")
AH_RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ah_compute_fi_cc_scientific_interpretation_v1/interpretation.json")
AI_RESULT = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ai_compute_fi_cc_scientific_report_v1/scientific_report.json")


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_phase5_status():
    x = load(RESULT)

    assert x[
        "status"
    ] == "PHASE5_COMPLETE_FROZEN"

    assert (
        x[
            "phase5_governance_status"
        ]
        == "FROZEN_PHASE5_COMPUTE_FAULT_STUDY_COMPLETE"
    )


def test_exact_frozen_source_hashes():
    x = load(RESULT)[
        "source_hashes"
    ]

    assert (
        x[
            "phase5af_final_success_sha256"
        ]
        == sha(AF_SUCCESS)
        == "0746c4c6d52a215dd85b7d943957aaf17dd0ad92eb739c892c80f4b2bb1742ac"
    )

    assert (
        x[
            "phase5ag_technical_acceptance_sha256"
        ]
        == sha(AG_RESULT)
        == "9bd147a666db08fa369a1f3f7a144a71415fe0fb71b6f513603a326343c63b19"
    )

    assert (
        x[
            "phase5ah_scientific_interpretation_sha256"
        ]
        == sha(AH_RESULT)
        == "3841ceb4a1af3ca2f1c32a7d1126ab3f94e6debbb46222f2635e694cec6012fb"
    )

    assert (
        x[
            "phase5ai_scientific_report_sha256"
        ]
        == sha(AI_RESULT)
        == "8a72edb086d255b45c9ad0fd6f9b0d207854611bfa052e6959e11f50a2f6e63f"
    )


def test_final_estate_cardinalities():
    x = load(RESULT)[
        "accepted_estate"
    ]

    assert x["subjects"] == 61
    assert x["trials"] == 6309
    assert x["stored_windows"] == 273830
    assert x["clean_caches"] == 366
    assert x["fault_shards"] == 732

    assert x[
        "grandfathered_R3_V2_transient_shards"
    ] == 1

    assert x[
        "R5_V3_shards"
    ] == 731

    assert x["outer_instance_ids"] == 20170008
    assert x["fault_records"] == 29798820


def test_final_scientific_findings():
    x = load(RESULT)[
        "scientific_findings"
    ]

    assert x["CC_available_strata"] == 576

    assert x[
        "CC_timing_unavailable_without_imputation"
    ] == 144

    assert x[
        "persistent_vs_transient_available_comparisons"
    ] == 24

    assert x[
        "persistent_more_adverse_descriptive"
    ] == 24

    assert x[
        "PTQ_less_adverse_descriptive"
    ] == 21

    assert x[
        "PTQ_more_adverse_descriptive"
    ] == 3

    assert x[
        "universal_variant_superiority_claim"
    ] is False


def test_completion_checks():
    x = load(RESULT)[
        "phase5_completion_checks"
    ]

    assert x[
        "prospective_compute_fault_estate_complete"
    ] is True

    assert x[
        "all_732_fault_shards_accepted"
    ] is True

    assert x[
        "technical_acceptance_complete"
    ] is True

    assert x[
        "scientific_interpretation_complete"
    ] is True

    assert x[
        "scientific_report_complete"
    ] is True

    assert x[
        "unavailable_timing_preserved"
    ] is True

    assert x[
        "outer_result_feedback_occurred"
    ] is False

    assert x[
        "scientific_protocol_changed_after_boundary"
    ] is False


def test_forbidden_outer_feedback():
    x = load(CONFIG)[
        "irreversible_governance"
    ]

    for key, value in x.items():
        assert value is False, key


def test_no_csc_onfield_or_physical_claim():
    x = load(CONFIG)[
        "phase5_scope"
    ]

    assert x["regimes_completed"] == [
        "C0",
        "CC",
    ]

    assert x["CSC_completed"] is False
    assert x["OnField_used"] is False
    assert x["physical_or_MCU_validation_claim"] is False


def test_commit_push_not_performed_by_phase5aj():
    x = load(RESULT)[
        "commit_push_authorization"
    ]

    assert x[
        "major_phase5_scope_complete"
    ] is True

    assert x[
        "commit_may_be_created_after_successful_phase5aj_regression"
    ] is True

    assert x[
        "push_may_follow_that_commit"
    ] is True

    assert x[
        "commit_created_by_phase5aj"
    ] is False

    assert x[
        "push_performed_by_phase5aj"
    ] is False


def test_final_summary_exists():
    assert SUMMARY_MD.is_file()
    assert SUMMARY_MD.stat().st_size > 0


def test_summary_refuses_universal_variant_claim():
    text = SUMMARY_MD.read_text()

    normalized = " ".join(
        text.split()
    )

    assert (
        "does not claim that one model variant is universally more robust"
        in normalized
    )

    assert (
        "No subject-primary median-lead robustness conclusion is made"
        in normalized
    )
