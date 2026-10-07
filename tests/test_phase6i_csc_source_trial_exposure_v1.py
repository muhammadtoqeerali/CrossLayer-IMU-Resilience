from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

import csc_outer_executor_v1 as csc
import csc_source_trial_exposure_v1 as exposure


ROOT = Path(__file__).resolve().parents[1]

HELPER = (
    ROOT
    / "experiments/phase_06/csc_source_trial_exposure_v1.py"
)


def instance(
    *,
    fault_id: str = "fault-1",
    family: str = "dropout",
    onset: int = 20,
    duration: int | None = 10,
    persistence: str = "transient",
):
    return {
        "fault_id":
            fault_id,

        "replay_id":
            f"replay-{fault_id}",

        "parent_kind":
            "source_trial",

        "family":
            family,

        "onset_sample":
            onset,

        "duration_samples":
            duration,

        "persistence":
            persistence,
    }


def test_pinned_contract_is_exact():
    observed = (
        exposure.validate_pinned_exposure_contract()
    )

    assert (
        observed[
            "status"
        ]
        == "PINNED_EXPOSURE_CONTRACT_VALID"
    )

    assert (
        observed[
            "window_length_samples"
        ]
        == 30
    )

    assert (
        observed[
            "eligibility_predicate"
        ]
        == "sensor_exposed_window_count >= 1"
    )


def test_helper_is_metadata_only_by_import_surface():
    tree = ast.parse(
        HELPER.read_text(
            encoding="utf-8"
        )
    )

    imported = set()

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imported.update(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imported.add(
                    node.module
                )

    assert "torch" not in imported
    assert "numpy" not in imported
    assert "sensor_fi_operators" not in imported
    assert "compute_fi_outer_fleet_executor_v1" not in imported


def test_finite_active_support_is_half_open_interval():
    assert (
        exposure.source_trial_fault_active_support(
            instance=instance(
                onset=20,
                duration=10,
            ),
            source_length=100,
        )
        == (
            20,
            30,
        )
    )


def test_until_end_active_support_uses_source_length():
    row = instance(
        family="jitter",
        onset=0,
        duration=None,
        persistence="until_end",
    )

    assert (
        exposure.source_trial_fault_active_support(
            instance=row,
            source_length=101,
        )
        == (
            0,
            101,
        )
    )


def test_exact_strict_overlap_boundary_semantics():
    row = instance(
        onset=30,
        duration=15,
    )

    observed = (
        exposure.source_trial_exposed_window_indices(
            instance=row,
            historical_window_ends=[
                30,
                45,
                60,
            ],
            source_length=100,
        )
    )

    # [30,45) does not overlap [0,30), but overlaps both
    # [15,45) and [30,60) under the frozen strict predicate.
    assert observed == [
        1,
        2,
    ]


def test_finite_episode_can_have_zero_retained_window_exposure():
    row = instance(
        onset=90,
        duration=5,
    )

    observed = (
        exposure.source_trial_exposed_window_indices(
            instance=row,
            historical_window_ends=[
                30,
                45,
                60,
            ],
            source_length=100,
        )
    )

    assert observed == []


def test_until_end_family_exposes_all_historical_windows():
    row = instance(
        family="orientation",
        onset=0,
        duration=None,
        persistence="until_end",
    )

    assert (
        exposure.source_trial_exposed_window_indices(
            instance=row,
            historical_window_ends=[
                30,
                45,
                60,
                75,
            ],
            source_length=100,
        )
        == [
            0,
            1,
            2,
            3,
        ]
    )


def test_census_produces_exact_r2_input_maps():
    candidates = [
        instance(
            fault_id="observable",
            onset=20,
            duration=10,
        ),
        instance(
            fault_id="unobservable",
            onset=90,
            duration=5,
        ),
    ]

    observed = exposure.source_trial_exposure_census(
        candidates=candidates,
        historical_window_ends=[
            30,
            45,
            60,
        ],
        source_length=100,
    )

    assert observed[
        "sensor_exposed_window_indices_by_fault_id"
    ] == {
        "observable":
            [
                0,
                1,
            ],

        "unobservable":
            [],
    }

    assert observed[
        "exposure_count_by_fault_id"
    ] == {
        "observable":
            2,

        "unobservable":
            0,
    }

    assert observed[
        "execution_performed"
    ] is False


def test_census_feeds_frozen_r2_observable_first_selector():
    validated = csc.validate_frozen_plan()

    candidates = [
        {
            **instance(
                fault_id="fault-unobservable",
                onset=90,
                duration=5,
            ),

            "parent_sequence_id":
                "subject=9|task=1|trial=1",
        },
        {
            **instance(
                fault_id="fault-observable",
                onset=20,
                duration=10,
            ),

            "parent_sequence_id":
                "subject=9|task=1|trial=1",
        },
    ]

    census = exposure.source_trial_exposure_census(
        candidates=candidates,
        historical_window_ends=[
            30,
            45,
            60,
        ],
        source_length=100,
    )

    selected = csc.select_source_trial_candidate(
        candidates=candidates,
        exposure_count_by_fault_id=census[
            "exposure_count_by_fault_id"
        ],
        fold=5,
        subject=9,
        family="dropout",
        severity="L1",
        phase6d=validated[
            "phase6d"
        ],
    )

    assert selected[
        "status"
    ] == "ELIGIBLE_SELECTED"

    assert selected[
        "selected"
    ][
        "fault_id"
    ] == "fault-observable"


def test_census_preserves_structural_omission_without_fallback():
    validated = csc.validate_frozen_plan()

    candidates = [
        {
            **instance(
                fault_id="fault-1",
                onset=90,
                duration=5,
            ),

            "parent_sequence_id":
                "subject=9|task=1|trial=1",
        },
        {
            **instance(
                fault_id="fault-2",
                onset=95,
                duration=5,
            ),

            "parent_sequence_id":
                "subject=9|task=1|trial=1",
        },
    ]

    census = exposure.source_trial_exposure_census(
        candidates=candidates,
        historical_window_ends=[
            30,
            45,
            60,
        ],
        source_length=100,
    )

    selected = csc.select_source_trial_candidate(
        candidates=candidates,
        exposure_count_by_fault_id=census[
            "exposure_count_by_fault_id"
        ],
        fold=5,
        subject=9,
        family="dropout",
        severity="L2",
        phase6d=validated[
            "phase6d"
        ],
    )

    assert selected == {
        "status":
            "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR",

        "selected":
            None,
    }


def test_invalid_family_is_rejected():
    with pytest.raises(
        ValueError,
        match="unsupported source-trial family",
    ):
        exposure.source_trial_fault_active_support(
            instance=instance(
                family="bias",
            ),
            source_length=100,
        )


def test_until_end_contract_is_not_silently_relaxed():
    with pytest.raises(
        ValueError,
        match="must have null duration",
    ):
        exposure.source_trial_fault_active_support(
            instance=instance(
                family="delay",
                onset=0,
                duration=10,
                persistence="until_end",
            ),
            source_length=100,
        )


def test_duplicate_fault_ids_are_rejected():
    rows = [
        instance(
            fault_id="duplicate",
        ),
        instance(
            fault_id="duplicate",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate source-trial fault_id",
    ):
        exposure.source_trial_exposure_census(
            candidates=rows,
            historical_window_ends=[
                30,
                45,
            ],
            source_length=100,
        )


def test_metadata_helper_does_not_modify_frozen_executor():
    path = (
        ROOT
        / "experiments/phase_06/csc_outer_executor_v1.py"
    )

    before = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    exposure.source_trial_exposure_census(
        candidates=[
            instance()
        ],
        historical_window_ends=[
            30,
            45,
            60,
        ],
        source_length=100,
    )

    after = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert after == before
