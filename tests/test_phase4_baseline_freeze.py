import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    return json.loads((ROOT / name).read_text())

def test_phase4f_fp32_config_is_frozen():
    d = load("configs/baseline/prospective_300ms_fp32_v1.json")
    assert d["status"] == "FROZEN"
    assert d["provenance_tier"] == "P0"
    assert d["workload"]["primary"] is True
    assert d["workload"]["window_ms"] == 300
    assert d["workload"]["sampling_hz"] == 100
    assert d["workload"]["samples_per_window"] == 30
    assert d["workload"]["overlap_fraction"] == 0.5
    assert d["workload"]["stride_samples"] == 15
    assert d["workload"]["stride_ms"] == 150
    assert d["training"]["seeds"] == [42, 123, 2025]
    assert d["training"]["folds"] == [1, 2, 3, 4, 5]
    assert d["selection_policy"]["outer_test"] == "evaluation only"
    assert d["selection_policy"]["single_best_seed_fold_selection"] == "prohibited"
    assert d["int8_transition_constraint"]["calibration_partition"] == "frozen training-only calibration IDs"

def test_phase4f_freeze_manifest_is_authoritative():
    d = load("manifests/phase_4f_prospective_300ms_fp32_freeze_v1.json")
    assert d["status"] == "FROZEN"
    assert d["schema_version"] == "phase4f_prospective_300ms_fp32_freeze_v1"
    assert d["workload"]["window_ms"] == 300
    assert d["workload"]["overlap_fraction"] == 0.5
    assert d["workload"]["stride_samples"] == 15
    assert d["fold_protocol"]["seeds"] == [42, 123, 2025]
    assert d["fold_protocol"]["folds"] == [1, 2, 3, 4, 5]
    assert d["fold_protocol"]["expected_checkpoint_count"] == 15
    assert d["fold_protocol"]["checkpoint_row_count"] == 15
    assert d["fold_protocol"]["threshold_row_count"] == 45
    assert d["fold_protocol"]["within_fold_subject_overlap"] == 0
    assert d["fold_protocol"]["outer_test_union_subject_count"] == 61
    assert d["fold_protocol"]["each_primary_subject_outer_test_exactly_once"] is True
    assert d["fold_protocol"]["actual_membership_exactly_matches_phase3_freeze"] is True
    assert d["fold_protocol"]["outer_test"] == "evaluation only"
    assert d["fold_protocol"]["threshold_selection"] == "validation only"
    assert d["fold_protocol"]["int8_calibration"] == "training partition only"

def test_phase4f_checkpoint_estate_is_complete_and_not_cherrypicked():
    d = load("manifests/phase_4f_prospective_300ms_fp32_freeze_v1.json")
    ckpts = d["checkpoints"]
    assert len(ckpts) == 15
    pairs = {(x["seed"], x["fold"]) for x in ckpts}
    assert pairs == {(s, f) for s in (42, 123, 2025) for f in range(1, 6)}
    assert all(len(x["sha256"]) == 64 for x in ckpts)
    assert d["checkpoint_policy"]["best_seed_or_fold_selected_from_outer_test"] is False
    assert d["checkpoint_policy"]["single_deployment_checkpoint_selected"] is False
    assert d["checkpoint_policy"]["retained_primary_evidence"] == "full 3-seed x 5-fold estate"

def test_phase4f_forbidden_roles_remain_excluded():
    d = load("manifests/phase_4f_prospective_300ms_fp32_freeze_v1.json")
    assert d["population"]["rejected_OnField"] == ["999", "1000"]
    assert d["scientific_constraints"]["outer_test_used_for_tuning"] is False
    assert d["scientific_constraints"]["OnField_used_for_training_or_tuning"] is False
    assert d["scientific_constraints"]["rejected_999_1000_used"] is False
    assert d["scientific_constraints"]["resilience_hypotheses_confirmed_by_phase4f"] is False

def test_phase4f_dirty_source_provenance_is_explicit():
    d = load("manifests/phase_4f_prospective_300ms_fp32_freeze_v1.json")
    s = d["source_lineage"]
    assert len(s["source_inventory_sha256"]) == 64
    assert s["source_inventory_file_count"] == len(s["source_inventory"])
    assert "models/CNN.py" in s["critical_source"]
    assert "train.py" in s["critical_source"]
    assert len(s["critical_source"]["models/CNN.py"]["sha256"]) == 64
    assert len(s["critical_source"]["train.py"]["sha256"]) == 64

def test_phase4f_clean_artifact_hashes_are_frozen():
    d = load("manifests/phase_4f_prospective_300ms_fp32_freeze_v1.json")
    expected = {
        "run_manifest.json",
        "checkpoint_manifest.csv",
        "validation_selected_thresholds.csv",
        "per_fold_event_metrics.csv",
        "test_trial_predictions.csv",
        "validation_trial_predictions.csv",
        "segment_diagnostics_manifest.csv",
    }
    assert set(d["clean_evidence_artifacts"]) == expected
    for x in d["clean_evidence_artifacts"].values():
        assert len(x["sha256"]) == 64
        assert x["bytes"] > 0
