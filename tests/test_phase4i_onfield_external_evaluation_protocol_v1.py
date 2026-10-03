import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CFG = (
    ROOT
    / "configs/evaluation/"
    "phase4i_onfield_external_evaluation_protocol_v1.json"
)

FREEZE = (
    ROOT
    / "manifests/"
    "phase_4i_onfield_external_evaluation_protocol_v1_freeze.json"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def test_protocol_frozen_pre_inference():
    x = load(CFG)
    f = load(FREEZE)

    assert x[
        "status"
    ] == "FROZEN_PRE_ONFIELD_MODEL_INFERENCE"

    assert f[
        "status"
    ] == "FROZEN_PRE_ONFIELD_MODEL_INFERENCE"

    assert f[
        "pre_inference_assertions"
    ][
        "onfield_model_forward_passes_executed"
    ] is False

    assert f[
        "pre_inference_assertions"
    ][
        "onfield_performance_values_read"
    ] is False


def test_external_cohort_is_activity_only_and_fixed():
    x = load(CFG)

    assert x["dataset"][
        "retained_storage_ids"
    ] == [
        str(i)
        for i in range(
            1001,
            1011,
        )
    ]

    assert x["dataset"][
        "retained_subject_count"
    ] == 10

    assert x["dataset"][
        "retained_trial_count"
    ] == 16

    assert x["dataset"][
        "retained_window_count"
    ] == 1023337

    assert x["dataset"][
        "label_inventory"
    ] == {
        "Activity": 1023337,
        "Falling": 0,
    }

    assert x["dataset"][
        "rejected_storage_ids"
    ] == [
        "999",
        "1000",
    ]


def test_all_15_checkpoints_are_mandatory_and_unselected():
    x = load(CFG)

    estate = x[
        "model_estate"
    ]

    assert estate[
        "seeds"
    ] == [
        42,
        123,
        2025,
    ]

    assert estate[
        "folds"
    ] == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert estate[
        "checkpoint_count_per_model_variant"
    ] == 15

    assert len(
        estate[
            "fp32_checkpoints"
        ]
    ) == 15

    assert estate[
        "single_deployment_checkpoint_selected"
    ] is False

    assert estate[
        "best_seed_or_fold_selection"
    ] == "PROHIBITED"

    assert estate[
        "probability_level_ensemble"
    ] == "PROHIBITED_NOT_PART_OF_FROZEN_MODEL"


def test_all_three_validation_selected_operating_points_required():
    x = load(CFG)

    op = x[
        "operating_points"
    ]

    assert op[
        "names"
    ] == [
        "balanced",
        "low_false_alarm",
        "timely_150ms",
    ]

    assert op[
        "threshold_row_count"
    ] == 45

    assert len(
        op["rows"]
    ) == 45

    assert op[
        "onfield_threshold_tuning"
    ] is False

    assert op[
        "onfield_operating_point_selection"
    ] is False

    assert op[
        "all_three_operating_points_must_be_reported"
    ] is True


def test_checkpoint_aggregation_is_equal_and_nonselective():
    x = load(CFG)

    a = x[
        "checkpoint_aggregation"
    ]

    assert a[
        "checkpoint_specific_results_retained"
    ] is True

    assert a[
        "within_subject_hierarchy"
    ] == [
        "compute metric independently for each checkpoint",
        "equal mean across 5 folds within each seed",
        "equal mean across the 3 seeds",
    ]

    assert a[
        "complete_rectangular_estate_equivalent_weight"
    ] == "1/15 per checkpoint"

    assert a[
        "checkpoint_performance_used_for_weighting"
    ] is False

    assert a[
        "checkpoint_performance_used_for_selection"
    ] is False


def test_subject_is_uncertainty_unit_not_windows():
    x = load(CFG)

    u = x[
        "uncertainty"
    ]

    assert u[
        "sampling_unit"
    ] == "external subject"

    assert u[
        "subject_count"
    ] == 10

    assert u[
        "bootstrap_replicates"
    ] == 10000

    assert u[
        "windows_bootstrapped"
    ] is False

    assert u[
        "trials_bootstrapped_as_independent_units"
    ] is False

    assert x[
        "activity_only_metrics"
    ][
        "overlapping_windows_treated_as_independent_in_uncertainty"
    ] is False


def test_activity_only_claim_boundary():
    x = load(CFG)

    allowed = set(
        x[
            "claim_boundary"
        ][
            "allowed"
        ]
    )

    prohibited = set(
        x[
            "claim_boundary"
        ][
            "prohibited"
        ]
    )

    assert (
        "activity_specificity"
        in allowed
    )

    assert (
        "false_triggers_per_activity_hour"
        in allowed
    )

    assert (
        "fall_recall"
        in prohibited
    )

    assert (
        "sensor_lead_time"
        in prohibited
    )

    assert (
        "FP32/PTQ equivalence"
        in prohibited
    )


def test_expected_execution_cardinalities():
    x = load(CFG)

    c = x[
        "expected_execution_counts"
    ]

    assert c[
        "model_window_evaluations"
    ] == 30700110

    assert c[
        "threshold_applications"
    ] == 92100330

    assert c[
        "subject_checkpoint_operating_point_rows"
    ] == 900

    assert c[
        "trial_checkpoint_operating_point_rows"
    ] == 1440

    assert c[
        "subject_estate_summary_rows"
    ] == 60

    assert c[
        "cohort_summary_rows"
    ] == 6


def test_governance_has_no_onfield_selection_feedback():
    x = load(CFG)

    g = x[
        "governance"
    ]

    assert all(
        value is False
        for value in g.values()
    )
