import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INTERP = (
    ROOT
    / "manifests/"
    "phase_4i_onfield_external_interpretation_v1.json"
)


def load():
    return json.loads(
        INTERP.read_text()
    )


def results():
    x = load()

    return {
        (
            row[
                "model_variant"
            ],
            row[
                "operating_point"
            ],
        ):
            row
        for row in x[
            "frozen_cohort_results"
        ]
    }


def test_status_and_external_scope():
    x = load()

    assert x[
        "status"
    ] == "FROZEN_ACTIVITY_ONLY_EXTERNAL_INTERPRETATION"

    assert x[
        "external_scope"
    ][
        "role"
    ] == "ACTIVITY_ONLY"

    assert x[
        "external_scope"
    ][
        "subjects"
    ] == 10

    assert x[
        "external_scope"
    ][
        "trials"
    ] == 16

    assert x[
        "external_scope"
    ][
        "windows"
    ] == 1023337

    assert x[
        "external_scope"
    ][
        "falling_windows"
    ] == 0

    assert x[
        "external_scope"
    ][
        "subject_bootstrap_replicates"
    ] == 10000


def test_six_frozen_result_rows():
    r = results()

    assert set(r) == {
        (
            "prospective_fp32_300ms",
            "balanced",
        ),
        (
            "prospective_fp32_300ms",
            "low_false_alarm",
        ),
        (
            "prospective_fp32_300ms",
            "timely_150ms",
        ),
        (
            "qualified_static_ptq_v7",
            "balanced",
        ),
        (
            "qualified_static_ptq_v7",
            "low_false_alarm",
        ),
        (
            "qualified_static_ptq_v7",
            "timely_150ms",
        ),
    }


def test_exact_reported_point_estimates():
    r = results()

    expected = {
        (
            "prospective_fp32_300ms",
            "balanced",
        ):
            (
                0.999010,
                18.328044,
            ),

        (
            "prospective_fp32_300ms",
            "low_false_alarm",
        ):
            (
                0.999005,
                4.044666,
            ),

        (
            "prospective_fp32_300ms",
            "timely_150ms",
        ):
            (
                0.995899,
                70.148137,
            ),

        (
            "qualified_static_ptq_v7",
            "balanced",
        ):
            (
                0.998950,
                19.446624,
            ),

        (
            "qualified_static_ptq_v7",
            "low_false_alarm",
        ):
            (
                0.998947,
                4.289012,
            ),

        (
            "qualified_static_ptq_v7",
            "timely_150ms",
        ):
            (
                0.995657,
                74.611724,
            ),
    }

    for key, values in expected.items():
        row = r[
            key
        ]

        assert round(
            row[
                "activity_specificity"
            ][
                "point_estimate"
            ],
            6,
        ) == values[
            0
        ]

        assert round(
            row[
                "false_triggers_per_activity_hour"
            ][
                "point_estimate"
            ],
            6,
        ) == values[
            1
        ]


def test_operating_point_order_is_descriptive_not_selection():
    x = load()

    d = x[
        "descriptive_operating_point_ordering"
    ]

    for model in (
        "prospective_fp32_300ms",
        "qualified_static_ptq_v7",
    ):
        assert d[
            model
        ][
            "false_triggers_per_activity_hour_low_to_high"
        ] == [
            "low_false_alarm",
            "balanced",
            "timely_150ms",
        ]

        assert d[
            model
        ][
            "selection_authorized"
        ] is False


def test_fp32_ptq_no_superiority_or_equivalence_claim():
    x = load()

    g = x[
        "governance_boundary"
    ]

    assert g[
        "fp32_ptq_superiority_claim"
    ] is False

    assert g[
        "fp32_ptq_equivalence_claim"
    ] is False


def test_activity_only_governance_boundary():
    x = load()

    g = x[
        "governance_boundary"
    ]

    assert g[
        "results_may_support_activity_side_reporting"
    ] is True

    assert g[
        "results_may_support_fall_side_reporting"
    ] is False

    assert g[
        "operating_point_selection_from_onfield"
    ] is False

    assert g[
        "checkpoint_selection_from_onfield"
    ] is False

    assert g[
        "threshold_selection_from_onfield"
    ] is False

    assert g[
        "model_weighting_from_onfield"
    ] is False

    assert g[
        "scientific_retuning_from_onfield"
    ] is False

    assert g[
        "binary_global_robustness_label"
    ] is False
