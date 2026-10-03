import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V2_CONFIG = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v2_fault_id.json"
)

V2_IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling_v2.py"
)

V3_CONFIG = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v3_causal_frame_loss.json"
)

V3_IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling_v3.py"
)

SEVERITY = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(path):
    return json.loads(path.read_text())


def test_single_change_is_frame_loss_minimum_onset():
    d = load(V3_CONFIG)
    c = d["single_change_from_v2"]

    assert c["family"] == "frame_loss"
    assert c["field"] == "minimum_eligible_onset_sample"
    assert c["v2_value"] == 0
    assert c["v3_value"] == 1

    for key in [
        "fault_family_changed",
        "severity_changed",
        "duration_changed",
        "target_policy_changed",
        "replicate_count_changed",
        "replicate_strata_count_changed",
        "seed_namespace_changed",
        "seed_derivation_changed",
        "partition_policy_changed",
        "physical_realism_claim_changed",
    ]:
        assert c[key] is False


def test_seed_contract_unchanged():
    v2 = load(V2_CONFIG)
    v3 = load(V3_CONFIG)

    assert v3["seed_namespace"] == v2["seed_namespace"]
    assert v3["seed_derivation"] == v2["seed_derivation"]
    assert v3["replication_policy"] == v2["replication_policy"]
    assert v3["target_policy"] == v2["target_policy"]


def test_dropout_behavior_is_identical_to_v2():
    v2 = load_module(
        "sampling_v2",
        V2_IMPL,
    )
    v3 = load_module(
        "sampling_v3",
        V3_IMPL,
    )
    severity = load(SEVERITY)

    kwargs = dict(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=10|task=10|trial=1",
        parent_length=1000,
        severity_protocol=severity,
    )

    a = [
        x
        for x in v2.generate_sequence_instances(**kwargs)
        if x["family"] == "dropout"
    ]

    b = [
        x
        for x in v3.generate_sequence_instances(**kwargs)
        if x["family"] == "dropout"
    ]

    assert a == b


def test_all_non_frame_loss_instances_identical_to_v2():
    v2 = load_module(
        "sampling_v2_nonframe",
        V2_IMPL,
    )
    v3 = load_module(
        "sampling_v3_nonframe",
        V3_IMPL,
    )
    severity = load(SEVERITY)

    kwargs = dict(
        fold=3,
        partition="training_calibration",
        parent_sequence_id="subject=106|task=27|trial=5",
        parent_length=3072,
        severity_protocol=severity,
    )

    a = [
        x
        for x in v2.generate_sequence_instances(**kwargs)
        if x["family"] != "frame_loss"
    ]

    b = [
        x
        for x in v3.generate_sequence_instances(**kwargs)
        if x["family"] != "frame_loss"
    ]

    assert a == b


def test_frame_loss_onset_is_strictly_positive():
    m = load_module(
        "sampling_v3_positive",
        V3_IMPL,
    )
    severity = load(SEVERITY)

    x = [
        row
        for row in m.generate_sequence_instances(
            fold=1,
            partition="training_calibration",
            parent_sequence_id="subject=10|task=10|trial=1",
            parent_length=1000,
            severity_protocol=severity,
        )
        if row["family"] == "frame_loss"
    ]

    assert len(x) == 9
    assert min(row["onset_sample"] for row in x) >= 1


def test_frame_loss_seed_is_unchanged_from_v2():
    v2 = load_module(
        "sampling_v2_seed",
        V2_IMPL,
    )
    v3 = load_module(
        "sampling_v3_seed",
        V3_IMPL,
    )
    severity = load(SEVERITY)

    kwargs = dict(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=10|task=10|trial=1",
        parent_length=1000,
        severity_protocol=severity,
    )

    a = [
        x
        for x in v2.generate_sequence_instances(**kwargs)
        if x["family"] == "frame_loss"
    ]

    b = [
        x
        for x in v3.generate_sequence_instances(**kwargs)
        if x["family"] == "frame_loss"
    ]

    assert [x["seed"] for x in a] == [x["seed"] for x in b]
    assert [x["fault_id"] for x in a] == [x["fault_id"] for x in b]


def test_instance_counts_unchanged():
    m = load_module(
        "sampling_v3_counts",
        V3_IMPL,
    )
    severity = load(SEVERITY)

    s = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=10|task=10|trial=1",
        parent_length=1000,
        severity_protocol=severity,
    )

    w = m.generate_window_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=10|task=10|trial=1|window=0",
        severity_protocol=severity,
    )

    assert len(s) == 138
    assert len(w) == 153


def test_onfield_still_rejected():
    m = load_module(
        "sampling_v3_onfield",
        V3_IMPL,
    )
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


def test_downstream_rebinding_is_required():
    d = load(V3_CONFIG)
    g = d["downstream_governance"]

    assert g["aggregation_v1_scientific_rules_changed"] is False
    assert g["reporting_v1_scientific_rules_changed"] is False
    assert g["existing_downstream_records_reference_sampling_v2"] is True
    assert g["rebinding_to_sampling_v3_required_before_fault_execution"] is True
    assert g["preexecution_governance_temporarily_reopened"] is True
