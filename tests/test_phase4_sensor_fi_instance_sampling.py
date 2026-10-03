import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_instance_sampling_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_sampling.py"
)

SEVERITY = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_sampling",
        IMPL,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_json(path):
    return json.loads(path.read_text())


def test_protocol_pre_result_boundaries():
    d = load_json(PROTOCOL)

    assert d["status"] == "FROZEN_PRE_SANITY_RESULT"
    assert d["evidence_tier"] == "P0"
    assert d["domain"] == "sensor"
    assert d["physical_realism_claim"] is False

    g = d["governance"]

    assert g["sampling_tuned_on_model_outcomes"] is False
    assert g["validation_used"] is False
    assert g["outer_test_used"] is False
    assert g["onfield_used"] is False
    assert g["faults_executed"] is False
    assert g["model_predictions_computed"] is False


def test_model_identity_excluded_from_seed_and_replay():
    d = load_json(PROTOCOL)

    assert d["seed_derivation"]["model_identity_in_seed"] is False

    r = d["replay_identity"]

    assert r["model_identity_in_replay_id"] is False
    assert r["same_instance_reused_across_backbones"] is True
    assert r["same_instance_reused_across_precision_variants"] is True


def test_exact_replication_policy():
    d = load_json(PROTOCOL)

    assert d["replication_policy"]["stochastic_families"] == {
        "noise": 3,
        "dropout": 3,
        "frame_loss": 3,
        "jitter": 3,
    }

    assert d["replication_policy"]["replicates_are_nested_within_parent_sequence"] is True
    assert d["replication_policy"]["replicates_are_independent_subjects_or_events"] is False


def test_expected_instance_counts():
    d = load_json(PROTOCOL)

    assert d["expected_instances_per_parent"]["stored_window"] == 153
    assert d["expected_instances_per_parent"]["source_trial"] == 138

    assert sum(
        d[
            "expected_instances_per_parent"
        ][
            "derivation"
        ][
            "stored_window"
        ].values()
    ) == 153

    assert sum(
        d[
            "expected_instances_per_parent"
        ][
            "derivation"
        ][
            "source_trial"
        ].values()
    ) == 138


def test_generator_counts_and_determinism():
    m = load_impl()
    severity = load_json(SEVERITY)

    a = m.generate_window_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1|window=0",
        severity_protocol=severity,
    )

    b = m.generate_window_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1|window=0",
        severity_protocol=severity,
    )

    assert len(a) == 153
    assert a == b

    s1 = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1",
        parent_length=212,
        severity_protocol=severity,
    )

    s2 = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1",
        parent_length=212,
        severity_protocol=severity,
    )

    assert len(s1) == 138
    assert s1 == s2


def test_replay_ids_unique_within_parent():
    m = load_impl()
    severity = load_json(SEVERITY)

    instances = (
        m.generate_window_instances(
            fold=1,
            partition="training_calibration",
            parent_sequence_id="subject=1|task=1|trial=1|window=0",
            severity_protocol=severity,
        )
        + m.generate_sequence_instances(
            fold=1,
            partition="training_calibration",
            parent_sequence_id="subject=1|task=1|trial=1",
            parent_length=212,
            severity_protocol=severity,
        )
    )

    ids = [
        x["replay_id"]
        for x in instances
    ]

    assert len(ids) == len(set(ids))


def test_all_sequence_bounds_valid_at_212_samples():
    m = load_impl()
    severity = load_json(SEVERITY)

    instances = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1",
        parent_length=212,
        severity_protocol=severity,
    )

    for x in instances:
        assert x["onset_sample"] >= 0

        if x["duration_samples"] is not None:
            assert (
                x["onset_sample"]
                + x["duration_samples"]
                <= 212
            )


def test_dropout_and_frame_loss_replicates_cover_three_strata():
    m = load_impl()
    severity = load_json(SEVERITY)

    instances = m.generate_sequence_instances(
        fold=1,
        partition="training_calibration",
        parent_sequence_id="subject=1|task=1|trial=1",
        parent_length=300,
        severity_protocol=severity,
    )

    for family in [
        "dropout",
        "frame_loss",
    ]:
        subset = [
            x
            for x in instances
            if (
                x["family"] == family
                and x["severity"]["level"] == "L2"
            )
        ]

        strata = {
            x["severity"]["onset_stratum"]
            for x in subset
        }

        assert strata == {0, 1, 2}


def test_onfield_generation_prohibited():
    m = load_impl()
    severity = load_json(SEVERITY)

    try:
        m.generate_window_instances(
            fold=1,
            partition="onfield",
            parent_sequence_id="forbidden",
            severity_protocol=severity,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "OnField generation was not rejected"
        )


def test_only_two_governance_items_remain_unfrozen():
    d = load_json(PROTOCOL)

    assert set(d["not_yet_frozen"]) == {
        "evaluation aggregation protocol",
        "robustness acceptance criteria",
    }
