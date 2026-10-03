import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMPLETION = (
    ROOT
    / "manifests/"
    "phase_4i_onfield_external_execution_completion_v1.json"
)


def load():
    return json.loads(
        COMPLETION.read_text()
    )


def test_completion_status_and_gate():
    x = load()

    assert x[
        "status"
    ] == (
        "COMPLETE_IMMUTABLE_ONFIELD_ACTIVITY_ONLY_EXTERNAL_EXECUTION"
    )

    assert x[
        "completion_gate"
    ] == (
        "QUALIFIED_COMPLETE_EXTERNAL_ACTIVITY_ONLY_EXECUTION"
    )


def test_execution_inventory():
    x = load()

    e = x[
        "execution_inventory"
    ]

    assert e[
        "retained_subject_count"
    ] == 10

    assert e[
        "retained_trial_count"
    ] == 16

    assert e[
        "retained_window_count"
    ] == 1023337

    assert e[
        "model_window_evaluations"
    ] == 30700110

    assert e[
        "threshold_applications"
    ] == 92100330

    assert e[
        "trial_checkpoint_operating_point_rows"
    ] == 1440

    assert e[
        "subject_checkpoint_operating_point_rows"
    ] == 900

    assert e[
        "subject_estate_summary_rows"
    ] == 60

    assert e[
        "cohort_summary_rows"
    ] == 6

    assert e[
        "subject_bootstrap_replicates"
    ] == 10000


def test_execution_guarantees():
    x = load()

    g = x[
        "execution_guarantees"
    ]

    assert g[
        "all_15_checkpoints_evaluated"
    ] is True

    assert g[
        "all_3_operating_points_evaluated"
    ] is True

    assert g[
        "subject_level_bootstrap_complete"
    ] is True

    assert g[
        "raw_probabilities_stored"
    ] is False

    assert g[
        "fall_side_metrics_generated"
    ] is False

    assert g[
        "checkpoint_selection"
    ] is False

    assert g[
        "threshold_tuning"
    ] is False

    assert g[
        "operating_point_selection"
    ] is False

    assert g[
        "global_binary_robustness_label"
    ] is False


def test_activity_only_claim_boundary():
    x = load()

    scope = x[
        "scientific_scope"
    ]

    assert scope[
        "external_role"
    ] == "ACTIVITY_ONLY"

    assert (
        "activity_specificity"
        in scope[
            "allowed_interpretation_after_completion"
        ]
    )

    assert (
        "false_triggers_per_activity_hour"
        in scope[
            "allowed_interpretation_after_completion"
        ]
    )

    assert (
        "fall_recall"
        in scope[
            "prohibited_claims"
        ]
    )

    assert (
        "sensor_lead_time"
        in scope[
            "prohibited_claims"
        ]
    )


def test_no_external_feedback():
    x = load()

    scope = x[
        "scientific_scope"
    ]

    assert scope[
        "onfield_results_used_for_tuning"
    ] is False

    assert scope[
        "onfield_results_used_for_checkpoint_selection"
    ] is False

    assert scope[
        "onfield_results_used_for_operating_point_selection"
    ] is False

    assert scope[
        "onfield_results_used_for_threshold_selection"
    ] is False

    assert scope[
        "performance_values_read_while_creating_completion_record"
    ] is False


def test_interpretation_authorization_is_activity_only():
    x = load()

    a = x[
        "authorization"
    ]

    assert a[
        "activity_only_result_interpretation"
    ] is True

    assert a[
        "fall_side_interpretation"
    ] is False

    assert a[
        "scientific_retuning"
    ] is False

    assert a[
        "checkpoint_reselection"
    ] is False

    assert a[
        "threshold_reselection"
    ] is False

    assert a[
        "operating_point_reselection"
    ] is False
