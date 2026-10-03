import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / (
    "phase_4h_sensor_fi_outer_execution_completion_v1.json"
)
PLAN = ROOT / "manifests" / (
    "phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def test_completion_manifest_status_and_scope():
    m = load(MANIFEST)

    assert m["schema"] == (
        "crosslayer_phase4h_sensor_fi_outer_execution_completion_v1"
    )
    assert m["status"] == "COMPLETE_IMMUTABLE_OUTER_EXECUTION"
    assert m["scientific_scope"]["partition"] == "outer_test"
    assert m["scientific_scope"]["regimes_executed"] == ["C0", "CS"]
    assert m["scientific_scope"]["onfield_used"] is False
    assert m["scientific_scope"]["outer_results_used_for_tuning"] is False
    assert (
        m["scientific_scope"][
            "performance_values_used_during_execution_gating"
        ]
        is False
    )


def test_completion_totals_equal_frozen_plan():
    m = load(MANIFEST)
    plan = load(PLAN)

    shards = plan["shards"]

    assert len(shards) == 793

    assert m["plan_totals_source"]["json_path"] == (
        "$.expected_totals_from_shards"
    )

    expected = plan["expected_totals_from_shards"]

    assert sum(
        int(x["expected_condition_rows"])
        for x in shards
    ) == int(expected["subject_condition_rows"]) == 320616

    assert sum(
        int(x["expected_unique_fault_instances"])
        for x in shards
    ) == int(expected["unique_fault_instances"]) == 42766632

    assert sum(
        int(x["expected_model_window_evaluations"])
        for x in shards
    ) == int(expected["model_window_evaluations"]) == 479750160

    totals = m["completion_totals"]

    assert totals["shards"] == 793
    assert totals["subjects"] == 61
    assert totals["subject_condition_rows"] == 320616
    assert totals["unique_fault_instances"] == 42766632
    assert totals["model_window_evaluations"] == 479750160
    assert totals["regime_shards"] == {"C0": 61, "CS": 732}
    assert totals["fold_subject_counts"] == {
        "1": 13,
        "2": 12,
        "3": 12,
        "4": 12,
        "5": 12,
    }
    assert totals["dataset_subject_counts"] == {
        "KFALL": 32,
        "UNIVR": 29,
    }


def test_every_family_has_one_shard_per_subject():
    m = load(MANIFEST)

    expected_families = {
        "C0",
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

    family_shards = m["completion_totals"]["family_shards"]

    assert set(family_shards) == expected_families
    assert all(value == 61 for value in family_shards.values())


def test_completion_integrity_digest_and_retention_policy():
    m = load(MANIFEST)
    integrity = m["artifact_integrity"]

    assert integrity["all_success_markers_pass"] is True
    assert integrity["all_declared_output_hashes_verified"] is True
    assert integrity["temporary_shard_directories_remaining"] == 0
    assert integrity["raw_probabilities_persisted"] is False
    assert integrity[
        "all_shard_success_marker_index_digest_sha256"
    ] == (
        "0683f407f68e6a9c1f1a3a93e5a2a6a5"
        "c2387cdeeea867c859dacc440b017dee"
    )


def test_operational_recovery_is_non_scientific():
    m = load(MANIFEST)
    incidents = m["operational_recovery_incidents"]

    assert len(incidents) == 1

    incident = incidents[0]

    assert incident["plan_index"] == 292
    assert incident["shard_id"] == (
        "p4h-o2-f2-s120-stuck_channel-3dc8bd0e9771"
    )
    assert incident["recovery_mechanism"] == (
        "frozen_executor_--recompute-partial"
    )
    assert incident["scientific_parameters_changed"] is False
    assert incident["result_values_used_to_choose_recovery"] is False
    assert incident["final_success_marker_sha256"] == (
        "bf75fdd20e3b1d398b0d6e6293663dd"
        "354d8c9510ed59ee52918f6927568e49e"
    )


def test_frozen_lineage_hashes():
    m = load(MANIFEST)

    expected = {
        "configs/evaluation/phase4h_sensor_fi_outer_executor_v1.json":
            "bb73d35e2a8458d490d2e605551eb1cb"
            "fae3112fad74cfdea322175de52c1ef9",

        "experiments/phase_04/sensor_fi_outer_executor_v1.py":
            "350f3356b5eed65d7e318effdb2c152f4"
            "f022e117fac73fcc0a8ee4d1871e72d",

        "manifests/phase_4h_sensor_fi_outer_executor_v1_freeze.json":
            "3dcaf7f560d516be675e95236917e4cbd"
            "874d415a5dcce3bcf7f9e19b2022162",

        "manifests/phase_4h_sensor_fi_outer_executor_v1_qualification.json":
            "5f2a5ad39a5de850dfc6fe2a66045028"
            "b2981f385189525ae8c108cb765720e7",

        "manifests/phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json":
            "a43df702c2694854e1ce79712af50a9bf"
            "91d0f24b40989241f604405f460024d",

        "manifests/phase_4h_sensor_fi_outer_execution_plan_v2_qualification_v1.json":
            "921ca4d3e4497e5517a57cc0df03d1a1"
            "6d9b037fd89030bfe33ebab3419ecb74",
    }

    assert set(m["frozen_lineage"]) == set(expected)

    for rel, wanted in expected.items():
        path = ROOT / rel
        assert sha256(path) == wanted
        assert m["frozen_lineage"][rel]["sha256"] == wanted


def test_completion_gate_only_authorizes_frozen_analysis():
    gate = load(MANIFEST)["completion_gate"]

    assert gate["state"] == (
        "QUALIFIED_COMPLETE_OUTER_EXECUTION_INTEGRITY"
    )
    assert gate["aggregation_authorized"] is True
    assert gate["reporting_authorized_after_frozen_aggregation"] is True
    assert gate["scientific_retuning_authorized"] is False
