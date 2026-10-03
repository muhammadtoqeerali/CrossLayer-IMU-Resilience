import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUAL = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_instance_sampling_v3_qualification_v1.json"
)

V2 = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v2_fault_id.json"
)

V3 = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v3_causal_frame_loss.json"
)

EXPECTED_V2_SHA = (
    "6004868ec664981f5feb8fc03d70f7263b41e94be597805fa7dba5f9831a3cdd"
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load():
    return json.loads(
        QUAL.read_text()
    )


def test_status():
    d = load()

    assert d["status"] == (
        "QUALIFIED_METADATA_REPLAY_CAUSAL_FRAME_LOSS"
    )


def test_v2_preserved():
    d = load()

    assert d["parent_v2"]["preserved_unchanged"] is True
    assert sha(V2) == EXPECTED_V2_SHA


def test_v3_pre_result_preserved():
    d = load()

    assert d["v3_pre_result_config"]["preserved_unchanged"] is True
    assert d["v3_pre_result_config"]["sha256"] == sha(V3)


def test_exhaustive_acceptance():
    d = load()
    a = d["acceptance"]

    assert a["fold_count"] == 5
    assert a["pass_count"] == 5
    assert a["fail_count"] == 0
    assert a["all_folds_pass"] is True
    assert a["v2_frame_loss_onset0_observed"] == 142
    assert a["v3_frame_loss_onset0_observed"] == 0
    assert a["non_frame_loss_sampling_mismatch_count"] == 0
    assert a["frame_loss_seed_mismatch_count"] == 0
    assert a["frame_loss_fault_id_mismatch_count"] == 0
    assert a["contract_failure_count"] == 0
    assert a["bounds_failure_count"] == 0


def test_no_results_or_heldout_used():
    d = load()
    b = d["scientific_boundary"]

    assert b["metadata_only"] is True
    assert b["sensor_values_mutated"] is False
    assert b["fault_injection_executed"] is False
    assert b["model_loaded"] is False
    assert b["model_predictions_computed"] is False
    assert b["validation_used"] is False
    assert b["outer_test_used"] is False
    assert b["onfield_used"] is False
    assert b["v2_rewritten"] is False
    assert b["severity_changed"] is False
    assert b["non_frame_loss_sampling_changed"] is False
    assert b["seed_derivation_changed"] is False


def test_rebind_and_jitter_semantics_remain_before_operators():
    d = load()

    assert d["downstream_governance_status"] == "REBIND_REQUIRED"
    assert len(
        d["remaining_items_before_fault_operator_implementation"]
    ) == 2
