from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs/evaluation/phase6e_csc_execution_plan_v1.json"
DOC = ROOT / "docs/PHASE_6E_CSC_EXECUTION_PLAN_V1.md"
TEST = ROOT / "tests/test_phase6e_csc_execution_plan.py"
MANIFEST = ROOT / "manifests/phase_6e_csc_execution_plan_freeze_v1.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_status_and_schema():
    cfg = load(CONFIG)

    assert cfg["schema_version"] == "phase6e_csc_execution_plan_v1"
    assert cfg["phase"] == "6E"
    assert cfg["status"] == "FROZEN_PRE_EXECUTION_CSC_PLAN"
    assert cfg["partition"] == "outer_test"
    assert cfg["evidence_tier"] == "P0"

    assert (
        cfg["execution_gate"]["status"]
        == "PLAN_FROZEN_EXECUTION_NOT_YET_PERFORMED"
    )


def test_upstream_hashes():
    cfg = load(CONFIG)

    for row in cfg["frozen_dependencies"].values():
        if not isinstance(row, dict):
            continue

        if "path" not in row:
            continue

        path = ROOT / row["path"]

        assert path.is_file()
        assert sha(path) == row["sha256"]


def test_pair_surface_counts():
    cfg = load(CONFIG)
    surface = cfg["pair_surface"]

    assert surface["original_phase6d_nominal_pair_count"] == 4239939
    assert surface["source_trial_original_group_count"] == 132489
    assert surface["source_trial_retained_pair_count"] == 130385
    assert surface["source_trial_structural_omission_count"] == 2104
    assert surface["stored_window_pair_count"] == 4107450
    assert surface["model_independent_csc_pair_count"] == 4237835
    assert surface["r2_pair_count_reduction"] == 2104

    assert (
        surface["source_trial_retained_pair_count"]
        + surface["stored_window_pair_count"]
        == surface["model_independent_csc_pair_count"]
    )

    assert (
        surface["original_phase6d_nominal_pair_count"]
        - surface["model_independent_csc_pair_count"]
        == surface["source_trial_structural_omission_count"]
    )

    assert surface["subject_family_shard_count"] == 732
    assert surface["nonzero_subject_family_shard_count"] == 732
    assert surface["zero_pair_shard_count"] == 0


def test_persistence_surface_counts():
    cfg = load(CONFIG)
    p = cfg["persistence_surface"]

    assert (
        p["source_trial"]["transient_pair_count"]
        + p["source_trial"]["persistent_pair_count"]
        == 130385
    )

    assert (
        p["stored_window"]["transient_pair_count"]
        + p["stored_window"]["persistent_pair_count"]
        == 4107450
    )

    assert (
        p["combined"]["transient_pair_count"]
        + p["combined"]["persistent_pair_count"]
        == 4237835
    )

    assert p["combined"]["transient_pair_count"] == 2117146
    assert p["combined"]["persistent_pair_count"] == 2120689


def test_model_variant_surface():
    cfg = load(CONFIG)
    surface = cfg["model_variant_surface"]

    assert surface["fp32"]["pair_count"] == 3026511
    assert surface["ptq_v7"]["pair_count"] == 4237835

    assert surface["fp32"]["seed_expanded_pair_member_count"] == 9079533
    assert surface["ptq_v7"]["seed_expanded_pair_member_count"] == 12713505

    assert surface["combined_model_variant_expanded_pair_count"] == 7264346
    assert surface["combined_seed_variant_expanded_pair_member_count"] == 21793038

    assert (
        surface["fp32"]["seed_expanded_pair_member_count"]
        + surface["ptq_v7"]["seed_expanded_pair_member_count"]
        == surface["combined_seed_variant_expanded_pair_member_count"]
    )


def test_temporal_workload_totals():
    cfg = load(CONFIG)
    w = cfg["temporal_workload"]

    assert w["frozen_c0_clean_cache_window_cardinality"] == 1642980
    assert w["sensor_reference_member_window_count"] == 35167107
    assert w["compute_faulted_member_window_count"] == 411540372
    assert w["simultaneous_csc_overlap_member_window_count"] == 19926021
    assert w["sensor_compute_union_member_window_count"] == 426781458
    assert w["reference_plus_faulted_accounting_sum"] == 446707479

    assert (
        w["sensor_reference_member_window_count"]
        + w["compute_faulted_member_window_count"]
        == w["reference_plus_faulted_accounting_sum"]
    )

    assert (
        w["reference_plus_faulted_accounting_sum_is_executor_architecture_claim"]
        is False
    )


def test_zero_overlap_is_retained_not_filtered():
    cfg = load(CONFIG)
    w = cfg["temporal_workload"]
    semantics = cfg["execution_semantics"]

    assert w["zero_temporal_overlap_pair_count"] == 1018215
    assert w["zero_temporal_overlap_pair_member_count"] == 5234355

    assert (
        w["zero_overlap_by_parent_kind"]["source_trial"]["pair_count"]
        == 13275
    )

    assert (
        w["zero_overlap_by_parent_kind"]["stored_window"]["pair_count"]
        == 1004940
    )

    assert (
        w["zero_overlap_by_parent_kind"]["source_trial"]["pair_member_count"]
        == 68277
    )

    assert (
        w["zero_overlap_by_parent_kind"]["stored_window"]["pair_member_count"]
        == 5166078
    )

    assert "retain pair" in semantics["zero_persistent_overlap_policy"]
    assert "do not filter" in semantics["zero_persistent_overlap_policy"]
    assert "resample" in semantics["zero_persistent_overlap_policy"]


def test_exact_28_strata_surface():
    cfg = load(CONFIG)
    strata = cfg["compute_strata"]

    assert len(strata) == 28
    assert [row["index"] for row in strata] == list(range(28))
    assert all(row["pair_count"] > 0 for row in strata)

    assert sum(
        row["pair_count"]
        for row in strata
    ) == 4237835

    assert sum(
        row["pair_member_count"]
        for row in strata
    ) == 21793038

    assert sum(
        row["member_compute_faulted_window_sum"]
        for row in strata
    ) == 411540372

    assert sum(
        row["member_overlap_window_sum"]
        for row in strata
    ) == 19926021

    assert all(
        row["eligible_model_variants"] == ["fp32", "ptq_v7"]
        for row in strata[:20]
    )

    assert all(
        row["eligible_model_variants"] == ["ptq_v7"]
        for row in strata[20:]
    )

    assert all(
        row["persistence"] == (
            "transient_one_inference"
            if row["index"] % 2 == 0
            else "persistent_from_onset_until_trial_end"
        )
        for row in strata
    )


def test_binding_hashes_are_frozen():
    cfg = load(CONFIG)
    b = cfg["derivation_bindings"]

    assert (
        b["source_r2_structural_census_sha256"]
        == "275c45f0522626cc65835332612efc1480bff6fbf73f927532c2586589cb5b13"
    )

    assert (
        b["pair_surface_binding_sha256"]
        == "3b39a909ea140db75da52315abcf5efe2748fcdb2d31f0fc98ddf766b000af53"
    )

    assert (
        b["combined_compute_coordinate_binding_sha256"]
        == "2117c7b07084566607ccac46db150faae81d28bef51d0dcab7c700627d8eb435"
    )

    assert (
        b["temporal_workload_binding_sha256"]
        == "5b447bca2f45d4889bcf0dc86d6170c40a1028171a9f1b0ea7a28e2884f12ba8"
    )


def test_governance_has_no_outcome_feedback_or_execution():
    cfg = load(CONFIG)
    g = cfg["governance"]

    false_fields = [
        "scientific_change_at_phase6e_freeze",
        "outer_performance_outcomes_used",
        "cs_performance_outcomes_used",
        "cc_performance_outcomes_used",
        "csc_performance_outcomes_used",
        "validation_performance_used",
        "onfield_used",
        "threshold_retuning",
        "checkpoint_selection",
        "sensor_family_selection",
        "sensor_severity_selection",
        "compute_target_selection",
        "compute_persistence_selection",
        "resampling",
        "rebalancing",
        "pair_materialization_before_freeze",
        "csc_model_forward_before_freeze",
        "physical_or_mcu_claim",
    ]

    for field in false_fields:
        assert g[field] is False


def test_manifest_integrity():
    manifest = load(MANIFEST)

    assert manifest["schema_version"] == "phase6e_csc_execution_plan_freeze_v1"
    assert manifest["status"] == "FROZEN_PRE_EXECUTION_CSC_PLAN"

    artifacts = manifest["artifacts"]

    assert artifacts["config"]["sha256"] == sha(CONFIG)
    assert artifacts["documentation"]["sha256"] == sha(DOC)
    assert artifacts["regression_test"]["sha256"] == sha(TEST)

    assert (
        manifest["bindings"]["pair_surface_binding_sha256"]
        == "3b39a909ea140db75da52315abcf5efe2748fcdb2d31f0fc98ddf766b000af53"
    )

    assert (
        manifest["bindings"]["combined_compute_coordinate_binding_sha256"]
        == "2117c7b07084566607ccac46db150faae81d28bef51d0dcab7c700627d8eb435"
    )

    assert (
        manifest["bindings"]["temporal_workload_binding_sha256"]
        == "5b447bca2f45d4889bcf0dc86d6170c40a1028171a9f1b0ea7a28e2884f12ba8"
    )
