import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = (
    ROOT
    / "configs/faults/"
    "phase4h_sensor_fi_severity_protocol_v1.json"
)

FREEZE = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_severity_protocol_freeze_v1.json"
)

IMPL = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_severity.py"
)


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "sensor_fi_severity",
        IMPL,
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def test_protocol_is_pre_result_p0_only():
    d = load_protocol()

    assert d["status"] == "FROZEN_PRE_SANITY_RESULT"
    assert d["protocol_class"] == "P0_SOFTWARE_STRESS_SEVERITY_LADDER"
    assert d["physical_realism_claim"] is False

    b = d["scope_boundaries"]

    assert b["model_robustness_measured"] is False
    assert b["model_predictions_used_for_design"] is False
    assert b["validation_outcomes_used"] is False
    assert b["outer_test_outcomes_used"] is False
    assert b["onfield_outcomes_used"] is False


def test_exact_severity_level_names():
    d = load_protocol()

    assert d["severity_levels"]["ordered"] == [
        "L1",
        "L2",
        "L3",
    ]


def test_exact_value_fault_levels():
    d = load_protocol()
    f = d["families"]

    assert f["bias"]["levels"] == {
        "L1": 0.05,
        "L2": 0.10,
        "L3": 0.20,
    }

    assert f["drift"]["levels"] == {
        "L1": 0.05,
        "L2": 0.10,
        "L3": 0.20,
    }

    assert f["scale_factor"]["levels"] == {
        "L1": 0.05,
        "L2": 0.10,
        "L3": 0.20,
    }

    assert f["noise"]["levels"] == {
        "L1": 0.50,
        "L2": 1.00,
        "L3": 2.00,
    }

    assert f["clipping_saturation"]["levels"] == {
        "L1": 1.00,
        "L2": 0.75,
        "L3": 0.50,
    }


def test_exact_temporal_levels():
    d = load_protocol()
    f = d["families"]

    assert f["stuck_channel"]["levels"] == {
        "L1": 5,
        "L2": 15,
        "L3": 30,
    }

    assert f["dropout"]["levels"] == {
        "L1": 5,
        "L2": 15,
        "L3": 30,
    }

    assert f["frame_loss"]["levels"] == {
        "L1": 1,
        "L2": 3,
        "L3": 5,
    }

    assert f["delay"]["levels"] == {
        "L1": 1,
        "L2": 5,
        "L3": 10,
    }

    assert f["jitter"]["levels"] == {
        "L1": 1.0,
        "L2": 2.0,
        "L3": 4.0,
    }

    assert f["orientation"]["levels"] == {
        "L1": 5.0,
        "L2": 15.0,
        "L3": 30.0,
    }


def test_axis_loss_levels():
    d = load_protocol()
    x = d["families"]["axis_loss"]["levels"]

    assert [
        x["L1"]["channel_count"],
        x["L2"]["channel_count"],
        x["L3"]["channel_count"],
    ] == [1, 3, 6]

    assert x["L3"]["target_variants"] == [
        [0, 1, 2, 3, 4, 5]
    ]


def test_no_cross_fold_pooled_reference_selection():
    d = load_protocol()
    r = d["fold_local_materialization_rule"]

    assert r["enabled"] is True
    assert r["cross_fold_pooling_for_scale_selection"] is False
    assert r["same_dimensionless_level_across_folds"] is True


def test_materialized_reference_levels_monotonic():
    m = load_impl()

    ref = np.asarray([
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
    ])

    for fn in [
        m.bias_magnitudes,
        m.drift_endpoint_magnitudes,
        m.noise_sigmas,
    ]:
        x = fn(ref)

        l1 = np.asarray(x["L1"])
        l2 = np.asarray(x["L2"])
        l3 = np.asarray(x["L3"])

        assert np.all(l1 < l2)
        assert np.all(l2 < l3)

    clip = m.clipping_thresholds(ref)

    assert np.all(
        np.asarray(clip["L1"])
        > np.asarray(clip["L2"])
    )

    assert np.all(
        np.asarray(clip["L2"])
        > np.asarray(clip["L3"])
    )


def test_scale_gains_remain_positive():
    m = load_impl()

    gains = m.scale_gains()

    for level in m.LEVELS:
        lo, hi = gains[level]

        assert 0.0 < lo < 1.0
        assert hi > 1.0


def test_orientation_matrices_are_proper_rotations():
    m = load_impl()

    for axis in ["x", "y", "z"]:
        for degrees in [5.0, 15.0, 30.0]:
            r = m.rotation_matrix(
                axis,
                degrees,
            )

            np.testing.assert_allclose(
                r.T @ r,
                np.eye(3),
                atol=1e-12,
                rtol=0.0,
            )

            np.testing.assert_allclose(
                np.linalg.det(r),
                1.0,
                atol=1e-12,
                rtol=0.0,
            )


def test_jitter_preserves_predeclared_ordering_margin():
    d = load_protocol()

    j = d["families"]["jitter"]

    assert j["sample_period_ms"] == 10.0
    assert max(j["levels"].values()) < 5.0


def test_not_yet_frozen_items_are_explicit():
    d = load_protocol()

    assert set(d["not_yet_frozen"]) == {
        "fault-instance onset sampling policy",
        "number of stochastic fault instances per parent sequence",
        "fault-instance seed namespace",
        "evaluation aggregation protocol",
        "robustness acceptance criteria",
    }


def test_freeze_manifest_has_no_model_results():
    d = json.loads(FREEZE.read_text())

    assert d["status"] == "FROZEN_PRE_SANITY_RESULT"
    assert d["prospective_before_model_robustness_results"] is True
    assert d["physical_realism_claim"] is False
    assert d["model_robustness_results_seen"] is False
    assert d["validation_outcomes_seen"] is False
    assert d["outer_test_outcomes_seen"] is False
    assert d["onfield_outcomes_seen"] is False
    assert d["faults_executed"] is False
