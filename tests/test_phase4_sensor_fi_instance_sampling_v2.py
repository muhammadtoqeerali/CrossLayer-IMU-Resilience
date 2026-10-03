import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V1_PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v1.json"
)

V2_PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v2_fault_id.json"
)

V2_IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling_v2.py"
)

SEVERITY = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_sampling_v2",
        V2_IMPL,
    )

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m


def load(path):
    return json.loads(path.read_text())


def test_v2_is_single_change_protocol():
    v2 = load(V2_PROTOCOL)
    c = v2["single_change_from_v1"]

    assert c["field"] == "fault_id"

    for key in [
        "severity_changed",
        "target_policy_changed",
        "onset_policy_changed",
        "duration_policy_changed",
        "replicate_policy_changed",
        "seed_namespace_changed",
        "seed_derivation_changed",
        "partition_policy_changed",
        "replay_canonicalization_changed",
    ]:
        assert c[key] is False


def test_seed_namespace_unchanged():
    v1 = load(V1_PROTOCOL)
    v2 = load(V2_PROTOCOL)

    assert v2["seed_namespace"] == v1["seed_namespace"]
    assert v2["seed_derivation"] == v1["seed_derivation"]


def test_scientific_sampling_rules_unchanged():
    v1 = load(V1_PROTOCOL)
    v2 = load(V2_PROTOCOL)

    for key in [
        "partition_policy",
        "parent_units",
        "window_parent_identity",
        "sequence_parent_identity",
        "target_policy",
        "replication_policy",
        "temporal_placement",
        "variant_policy",
        "expected_instances_per_parent",
        "instance_required_fields",
        "governance",
        "not_yet_frozen",
    ]:
        assert v2[key] == v1[key]


def test_fault_id_contains_partition_and_fold():
    m = load_impl()
    severity = load(SEVERITY)

    a = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1",
        parent_length=3647,
        severity_protocol=severity,
    )[0]

    b = m.generate_sequence_instances(
        fold=2,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1",
        parent_length=3647,
        severity_protocol=severity,
    )[0]

    assert a["fault_id"] != b["fault_id"]
    assert "training_calibration-f1-" in a["fault_id"]
    assert "training_calibration-f2-" in b["fault_id"]


def test_seed_behavior_remains_fold_specific_and_deterministic():
    m = load_impl()

    kwargs = dict(
        partition="training_calibration",
        parent_kind="source_trial",
        parent_sequence_id="subject=9|task=1|trial=1",
        family="dropout",
        severity_level="L2",
        variant_id="ch0",
        replicate_index=1,
    )

    a = m.derive_seed(
        fold=1,
        **kwargs,
    )

    b = m.derive_seed(
        fold=1,
        **kwargs,
    )

    c = m.derive_seed(
        fold=2,
        **kwargs,
    )

    assert a == b
    assert a != c


def test_counts_unchanged():
    m = load_impl()
    severity = load(SEVERITY)

    w = m.generate_window_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1|window=0",
        severity_protocol=severity,
    )

    s = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1",
        parent_length=3647,
        severity_protocol=severity,
    )

    assert len(w) == 153
    assert len(s) == 138


def test_same_fold_replay_deterministic():
    m = load_impl()
    severity = load(SEVERITY)

    a = m.generate_sequence_instances(
        fold=3,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1",
        parent_length=3647,
        severity_protocol=severity,
    )

    b = m.generate_sequence_instances(
        fold=3,
        partition="training_calibration",
        parent_sequence_id="subject=9|task=1|trial=1",
        parent_length=3647,
        severity_protocol=severity,
    )

    assert a == b


def test_onfield_still_rejected():
    m = load_impl()
    severity = load(SEVERITY)

    try:
        m.generate_sequence_instances(
            fold=1,
            partition="onfield",
            parent_sequence_id="forbidden",
            parent_length=1000,
            severity_protocol=severity,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "OnField generation was not rejected"
        )
