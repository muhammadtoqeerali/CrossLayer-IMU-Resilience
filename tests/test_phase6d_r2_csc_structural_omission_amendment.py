from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PHASE6D_CONFIG = (
    ROOT
    / "configs/evaluation/phase6d_csc_pairing_protocol_v1.json"
)

PHASE6D_R1_CONFIG = (
    ROOT
    / "configs/evaluation/phase6d_r1_csc_pairing_clarification_v1.json"
)

R2_CONFIG = (
    ROOT
    / "configs/evaluation/"
      "phase6d_r2_csc_structural_omission_amendment_v1.json"
)

R2_DOC = (
    ROOT
    / "docs/"
      "PHASE_6D_R2_CSC_STRUCTURAL_OMISSION_AMENDMENT_V1.md"
)

R2_TEST = Path(__file__).resolve()

R2_MANIFEST = (
    ROOT
    / "manifests/"
      "phase_6d_r2_csc_structural_omission_amendment_freeze_v1.json"
)

EXPECTED_PHASE6D_SHA = (
    "4f4585e410782ed039c4cd7f21bce418ab13ac7e1ee4c9aa4750cba15677617e"
)

EXPECTED_R1_SHA = (
    "60b557b16cb7c9d75ea7fd67299c264624b024fe962d436ce336827031faeea0"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _load(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def test_original_phase6d_and_r1_are_byte_identical():
    assert _sha(
        PHASE6D_CONFIG
    ) == EXPECTED_PHASE6D_SHA

    assert _sha(
        PHASE6D_R1_CONFIG
    ) == EXPECTED_R1_SHA


def test_status_and_scientific_change_are_explicit():
    cfg = _load(
        R2_CONFIG
    )

    assert cfg[
        "phase"
    ] == "6D-R2"

    assert cfg[
        "status"
    ] == (
        "FROZEN_PRE_PLAN_PROSPECTIVE_CSC_"
        "STRUCTURAL_OMISSION_AMENDMENT"
    )

    assert cfg[
        "scientific_change"
    ] is True


def test_rule_is_geometry_only_and_precedes_compute_assignment():
    cfg = _load(
        R2_CONFIG
    )

    domain = cfg[
        "qualification_domain"
    ]

    assert domain[
        "partition"
    ] == "training_calibration"

    assert domain[
        "outer_test_used"
    ] is False

    assert domain[
        "model_outputs_used"
    ] is False

    assert domain[
        "performance_outcomes_used"
    ] is False

    order = cfg[
        "source_trial_selection_rule"
    ][
        "order"
    ]

    assert "retained-window exposure" in order[
        1
    ]

    assert "eligible set is nonempty" in order[
        3
    ]

    assert "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR" in order[
        4
    ]

    assert "assign frozen Phase-6D compute stratum" in order[
        5
    ]


def test_empty_eligible_group_is_omitted_without_replacement():
    cfg = _load(
        R2_CONFIG
    )

    eligibility = cfg[
        "source_trial_candidate_eligibility"
    ]

    selection = cfg[
        "source_trial_selection_rule"
    ]

    assert eligibility[
        "new_sensor_instances_generated"
    ] is False

    assert eligibility[
        "candidate_resampling_allowed"
    ] is False

    assert eligibility[
        "temporal_relocation_allowed"
    ] is False

    assert eligibility[
        "replacement_allowed"
    ] is False

    assert selection[
        "rebalancing_after_omission_allowed"
    ] is False

    assert selection[
        "omitted_group_compute_target_assignment"
    ] == "NOT_PERFORMED"

    assert selection[
        "omitted_group_CSC_pair"
    ] == "NONE"


def test_hash_min_identity_is_unchanged_for_eligible_candidates():
    cfg = _load(
        R2_CONFIG
    )

    rule = cfg[
        "source_trial_selection_rule"
    ]

    assert rule[
        "hash_namespace_changed"
    ] is False

    assert rule[
        "hash_payload_changed"
    ] is False

    assert rule[
        "hash_algorithm_changed"
    ] is False

    assert rule[
        "hash_ranking_changed"
    ] is False


def test_r1_abort_is_retained_as_internal_consistency_gate():
    cfg = _load(
        R2_CONFIG
    )

    interaction = cfg[
        "interaction_with_r1"
    ]

    assert interaction[
        "r1_window_geometry_unchanged"
    ] is True

    assert interaction[
        "r1_sensor_exposure_predicate_unchanged"
    ] is True

    assert interaction[
        "r1_transient_window_hash_rule_unchanged"
    ] is True

    assert interaction[
        "r1_persistent_overlap_rule_unchanged"
    ] is True

    text = interaction[
        "r1_zero_exposure_abort_role_after_r2"
    ]

    assert "ABORT_PLAN_DERIVATION_NO_FALLBACK" in text
    assert "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR" in text


def test_devcal_qualification_evidence_is_exact():
    cfg = _load(
        R2_CONFIG
    )

    evidence = cfg[
        "qualification_evidence"
    ]

    parent = evidence[
        "devcal_parent_universe"
    ]

    assert parent[
        "unique_parent_count"
    ] == 5346

    assert parent[
        "fold_parent_memberships"
    ] == 11221

    assert parent[
        "sha256"
    ] == (
        "8c506d09d053cdfd07c01351ab2f2ca5"
        "ddc6b10cba8db94b61b2775463fff234"
    )

    r2c = evidence[
        "r2c_observability_census"
    ]

    assert r2c[
        "qualification_group_count"
    ] == 235641

    assert r2c[
        "zero_eligible_group_count"
    ] == 2250

    assert r2c[
        "census_sha256"
    ] == (
        "db46998af6d00137529b89a7cfb7519b"
        "674def7de7d99ebe703d5e6cda00e0fc"
    )

    r2d = evidence[
        "r2d_structural_omission_qualification"
    ]

    assert r2d[
        "eligible_source_trial_groups"
    ] == 233391

    assert r2d[
        "structurally_ineligible_groups"
    ] == 2250

    assert r2d[
        "candidate_rule_digest_sha256"
    ] == (
        "740b444fa8b9a08fbc5aa154b9f0ea2d"
        "4f77d38f33bd4921b363d49bf008b412"
    )


def test_all_predeclared_coverage_gates_passed():
    cfg = _load(
        R2_CONFIG
    )

    gates = cfg[
        "qualification_evidence"
    ][
        "r2d_structural_omission_qualification"
    ][
        "coverage_gates"
    ]

    assert gates == {
        "all_falling_family_severity_have_event_coverage":
            True,

        "all_fold_family_severity_strata_retained":
            True,

        "all_fold_parent_severity_have_some_sequence_pair":
            True,

        "all_source_family_severity_surfaces_retained":
            True,
    }

    surface = cfg[
        "scientific_surface"
    ]

    assert surface[
        "all_12_sensor_families_retained_as_analysis_surface"
    ] is True

    assert surface[
        "all_3_sensor_severities_retained_as_analysis_surface"
    ] is True

    assert surface[
        "structural_missingness_must_not_be_imputed"
    ] is True


def test_old_nominal_pair_count_is_explicitly_not_authoritative():
    cfg = _load(
        R2_CONFIG
    )

    governance = cfg[
        "pair_count_governance"
    ]

    assert governance[
        "phase6d_original_nominal_pair_count"
    ] == 4239939

    assert governance[
        "phase6d_original_nominal_pair_count_still_authoritative"
    ] is False

    assert governance[
        "exact_post_r2_pair_count_known_before_outer_structural_recheck"
    ] is False

    assert governance[
        "outer_structural_recheck_may_use_performance_outcomes"
    ] is False

    assert governance[
        "outer_structural_recheck_may_execute_models"
    ] is False


def test_manifest_binds_config_doc_test_and_no_outcome_feedback():
    manifest = _load(
        R2_MANIFEST
    )

    assert manifest[
        "scientific_change"
    ] is True

    assert manifest[
        "config"
    ][
        "sha256"
    ] == _sha(
        R2_CONFIG
    )

    assert manifest[
        "documentation"
    ][
        "sha256"
    ] == _sha(
        R2_DOC
    )

    assert manifest[
        "regression_test"
    ][
        "sha256"
    ] == _sha(
        R2_TEST
    )

    governance = manifest[
        "execution_plan_governance"
    ]

    assert governance[
        "CS_performance_outcomes_used"
    ] is False

    assert governance[
        "CC_performance_outcomes_used"
    ] is False

    assert governance[
        "CSC_model_forward_before_this_freeze"
    ] is False

    assert governance[
        "CSC_pair_materialization_before_this_freeze"
    ] is False

    assert governance[
        "phase6e_must_depend_on_r2"
    ] is True
