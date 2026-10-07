from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = (
    ROOT
    / "configs/evaluation/phase6d_csc_pairing_protocol_v1.json"
)

BASE_DOC = (
    ROOT
    / "docs/PHASE_6D_CSC_PAIRING_PROTOCOL_V1.md"
)

BASE_TEST = (
    ROOT
    / "tests/test_phase6d_csc_pairing_protocol.py"
)

BASE_MANIFEST = (
    ROOT
    / "manifests/phase_6d_csc_pairing_protocol_freeze_v1.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/phase6d_r1_csc_pairing_clarification_v1.json"
)


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_original_phase6d_is_byte_identical():
    assert sha(BASE_CONFIG) == "4f4585e410782ed039c4cd7f21bce418ab13ac7e1ee4c9aa4750cba15677617e"
    assert sha(BASE_DOC) == "1dbaa4ff24e2133202992c91850dfe88833393d9ec83caf28635e9de3322b9c1"
    assert sha(BASE_TEST) == "0961f42b90df11ed08430f7d001b3c5c96070b5a02ac0dbc23480df9400a3b68"
    assert sha(BASE_MANIFEST) == "f4f9f79485d60948aa4e88f405b5453b63f83760e44a4e2f787392aca799bea8"


def test_status_and_no_scientific_change():
    x = load(CONFIG)

    assert x["phase"] == "6D-R1"

    assert (
        x["status"]
        == "FROZEN_PRE_PLAN_CSC_PAIRING_CLARIFICATION"
    )

    assert x["scientific_change"] is False


def test_exact_window_geometry():
    x = load(CONFIG)[
        "source_trial_window_geometry"
    ]

    assert x[
        "window_length_samples"
    ] == 30

    assert x[
        "window_stride_samples"
    ] == 15

    assert x[
        "historical_window_end_semantics"
    ] == "exclusive_stop"

    assert x[
        "window_support_inclusive"
    ] == (
        "[historical_window_end-30,"
        "historical_window_end-1]"
    )


def test_exact_sensor_support():
    x = load(CONFIG)[
        "sensor_active_support"
    ]

    assert x[
        "finite_duration"
    ][
        "support_inclusive"
    ] == (
        "[onset_sample,"
        "onset_sample+duration_samples-1]"
    )

    assert x[
        "until_end"
    ][
        "support_inclusive"
    ] == (
        "[onset_sample,parent_length-1]"
    )


def test_exact_exposure_boundary_cases():
    def exposed(
        historical_end,
        onset,
        duration,
        parent_length,
    ):
        window_start = historical_end - 30
        window_stop = historical_end

        fault_start = onset

        fault_stop = (
            parent_length
            if duration is None
            else onset + duration
        )

        return (
            window_start < fault_stop
            and fault_start < window_stop
        )

    assert exposed(
        30,
        0,
        1,
        100,
    )

    assert not exposed(
        30,
        30,
        1,
        100,
    )

    assert exposed(
        45,
        44,
        1,
        100,
    )

    assert not exposed(
        45,
        45,
        1,
        100,
    )

    assert exposed(
        45,
        40,
        None,
        100,
    )


def test_source_trial_transient_selection_vector():
    x = load(CONFIG)

    rule = x[
        "source_trial_transient_compute_window_selection"
    ]

    fixture = x[
        "qualification_vector"
    ]

    assert rule[
        "canonical_payload_fields"
    ] == [
        "namespace",
        "sensor_replay_id",
        "target_name",
    ]

    encoded = canonical_json(
        fixture[
            "payload"
        ]
    )

    assert (
        encoded
        == fixture[
            "canonical_json"
        ]
    )

    digest = hashlib.sha256(
        encoded.encode(
            "utf-8"
        )
    ).digest()

    assert (
        hashlib.sha256(
            encoded.encode(
                "utf-8"
            )
        ).hexdigest()
        == fixture[
            "sha256"
        ]
    )

    u64 = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    assert (
        u64
        % fixture[
            "exposed_window_count"
        ]
        == fixture[
            "selected_index"
        ]
    )


def test_zero_exposure_is_strict_abort():
    x = load(CONFIG)[
        "source_trial_transient_compute_window_selection"
    ]

    assert (
        x[
            "zero_exposed_window_policy"
        ]
        == "ABORT_PLAN_DERIVATION_NO_FALLBACK"
    )

    assert x[
        "rebalancing_allowed"
    ] is False

    assert x[
        "resampling_allowed"
    ] is False

    assert x[
        "outcome_dependent_reselection_allowed"
    ] is False


def test_persistent_overlap_is_recorded_not_filtered():
    x = load(CONFIG)[
        "persistent_compute_temporal_overlap"
    ]

    assert x[
        "temporal_overlap"
    ] == "len(overlap_set) > 0"

    assert x[
        "temporal_overlap_window_count"
    ] == "len(overlap_set)"

    assert x[
        "filter_pair_when_no_overlap"
    ] is False

    assert x[
        "resample_when_no_overlap"
    ] is False


def test_no_execution_or_outcome_feedback():
    x = load(CONFIG)[
        "execution_plan_governance"
    ]

    assert x[
        "phase6e_must_depend_on_this_clarification"
    ] is True

    forbidden_true = (
        "pair_materialization_before_this_freeze",
        "CSC_model_forward_before_this_freeze",
        "CS_performance_outcomes_used",
        "CC_performance_outcomes_used",
        "threshold_retuning",
        "checkpoint_selection",
        "sensor_family_selection",
        "sensor_severity_selection",
        "compute_target_selection",
        "compute_persistence_selection",
        "OnField_used",
        "physical_or_MCU_claim",
    )

    for key in forbidden_true:
        assert x[key] is False
