import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = (
    ROOT
    / "configs/faults/phase4h_sensor_fi_input_contract_v1.json"
)

EVIDENCE = (
    ROOT
    / "manifests/phase_4h_sensor_input_provenance_evidence_v1.json"
)

FREEZE = (
    ROOT
    / "manifests/phase_4h_sensor_fi_input_contract_freeze_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/sensor_fi_contract.py"
)


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_contract",
        IMPL,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_phase4h_contract_is_pre_severity():
    d = json.loads(CONTRACT.read_text())

    assert d["status"] == "FROZEN_PRE_SEVERITY"
    assert d["severity_policy"]["severity_values_frozen"] is False
    assert d["severity_policy"]["severity_grid"] is None
    assert d["severity_policy"]["default_severities"] is None
    assert d["severity_policy"]["physical_realism_claim_allowed"] is False


def test_phase4h_exact_family_set():
    d = json.loads(CONTRACT.read_text())

    assert set(d["fault_families"]) == {
        "bias",
        "drift",
        "scale_factor",
        "noise",
        "clipping_saturation",
        "stuck_channel",
        "axis_loss",
        "dropout",
        "frame_loss",
        "jitter",
        "delay",
        "orientation",
    }


def test_phase4h_layer_partition():
    d = json.loads(CONTRACT.read_text())
    f = d["fault_families"]

    window = {
        name
        for name, meta in f.items()
        if meta["temporal_class"] == "window_value"
    }

    sequence = {
        name
        for name, meta in f.items()
        if meta["temporal_class"] == "sequence"
    }

    assert window == {
        "bias",
        "scale_factor",
        "noise",
        "clipping_saturation",
        "axis_loss",
    }

    assert sequence == {
        "drift",
        "stuck_channel",
        "dropout",
        "frame_loss",
        "jitter",
        "delay",
        "orientation",
    }


def test_phase4h_input_and_units():
    d = json.loads(CONTRACT.read_text())
    s = d["stored_input_contract"]

    assert s["shape"] == [30, 9]
    assert s["sampling_rate_hz"] == 100
    assert s["stride_samples"] == 15

    assert s["channel_order"] == [
        "AccX",
        "AccY",
        "AccZ",
        "GyrX",
        "GyrY",
        "GyrZ",
        "EulerX",
        "EulerY",
        "EulerZ",
    ]

    assert [s["software_units"][str(i)] for i in range(6)] == [
        "mg",
        "mg",
        "mg",
        "mdps",
        "mdps",
        "mdps",
    ]


def test_phase4h_effective_channels_only():
    d = json.loads(CONTRACT.read_text())
    m = d["model_effective_input_contract"]

    assert m["effective_channel_indices"] == [0, 1, 2, 3, 4, 5]
    assert m["excluded_primary_fault_target_indices"] == [6, 7, 8]


def test_phase4h_exact_provenance_checks():
    d = json.loads(EVIDENCE.read_text())

    assert d["status"] == "FROZEN"

    assert d["kfall_raw_to_oriented"]["status"] == (
        "EXACT_CURRENT_TRANSFORM_MATCH"
    )

    assert d["kfall_oriented_to_primary_300ms"]["status"] == (
        "EXACT_FILTERED_WINDOW_REPRODUCTION"
    )

    u = d["univr_oriented_to_primary_300ms"]

    assert u["all_exact"] is True
    assert u["all_unique"] is True
    assert u["expected_15_sample_stride"] is True
    assert u["maximum_best_absolute_difference"] == 0.0


def valid_instance(m):
    x = {
        "fault_id": "toy-bias-001",
        "family": "bias",
        "domain": "sensor",
        "evidence_tier": "P0",
        "injection_layer": m.WINDOW_LAYER,
        "target_channels": [0],
        "onset_sample": 0,
        "duration_samples": 30,
        "persistence": "transient",
        "severity": {
            "offset": 1.0,
            "unit": "mg",
        },
        "severity_provenance": {
            "evidence_tier": "P0",
            "physical_realism_claim": False,
            "source": "synthetic_test_only",
        },
        "seed": 42,
        "parent_sequence_id": "toy-sequence",
        "partition": "development",
        "replay_id": "",
    }

    x["replay_id"] = m.compute_replay_id(x)
    return x


def test_phase4h_replay_identity_is_deterministic():
    m = load_impl()
    x = valid_instance(m)

    assert m.validate_fault_instance(x) is True

    a = m.compute_replay_id(x)
    b = m.compute_replay_id(x)

    assert a == b == x["replay_id"]

    y = dict(x)
    y["seed"] = 43
    y["replay_id"] = m.compute_replay_id(y)

    assert y["replay_id"] != x["replay_id"]


def test_phase4h_rejects_euler_primary_target():
    m = load_impl()
    x = valid_instance(m)

    x["target_channels"] = [6]
    x["replay_id"] = m.compute_replay_id(x)

    with pytest.raises(ValueError):
        m.validate_fault_instance(x)


def test_phase4h_rejects_wrong_layer():
    m = load_impl()
    x = valid_instance(m)

    x["injection_layer"] = m.SEQUENCE_LAYER
    x["replay_id"] = m.compute_replay_id(x)

    with pytest.raises(ValueError):
        m.validate_fault_instance(x)


def test_phase4h_rejects_physical_realism_claim():
    m = load_impl()
    x = valid_instance(m)

    x["severity_provenance"]["physical_realism_claim"] = True
    x["replay_id"] = m.compute_replay_id(x)

    with pytest.raises(ValueError):
        m.validate_fault_instance(x)


def test_phase4h_leakage_boundaries():
    d = json.loads(CONTRACT.read_text())
    p = d["partition_policy"]
    r = d["replay_contract"]

    assert r["split_before_stochastic_fault_injection"] is True
    assert p["validation_outcomes_may_select_severity"] is False
    assert p["outer_test_may_select_severity"] is False
    assert p["onfield_may_select_severity"] is False


def test_phase4h_freeze_manifest_has_no_fault_execution():
    d = json.loads(FREEZE.read_text())

    assert d["status"] == "FROZEN_PRE_SEVERITY"
    assert d["fault_family_count"] == 12
    assert d["severity_values_frozen"] is False
    assert d["fault_execution_performed"] is False
    assert d["scientific_boundary"]["physical_realism_claim"] is False
