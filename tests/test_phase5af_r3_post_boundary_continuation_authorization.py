from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AUTH = (
    ROOT
    / "configs/evaluation/"
    "phase5af_r3_post_boundary_continuation_authorization_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status_and_boundary_state():
    x = load(
        AUTH
    )

    assert (
        x["status"]
        == "FROZEN_POST_BOUNDARY_CC_CONTINUATION_AUTHORIZATION"
    )

    assert (
        x["boundary_state"]
        == "ONE_WAY_OUTCOME_BOUNDARY_ALREADY_CROSSED"
    )


def test_all_hash_bound_file_dependencies_exist():
    x = load(
        AUTH
    )

    sections = (
        x[
            "original_scientific_authority"
        ],
        x[
            "original_execution_lineage"
        ],
        x[
            "repair_evidence"
        ],
        x[
            "authorized_continuation_chain"
        ],
    )

    for section in sections:
        for rec in section.values():
            if not (
                isinstance(
                    rec,
                    dict,
                )
                and "path" in rec
                and "sha256" in rec
            ):
                continue

            path = Path(
                rec[
                    "path"
                ]
            )

            if not path.is_absolute():
                path = ROOT / path

            assert path.is_file()

            assert (
                sha(
                    path
                )
                == rec[
                    "sha256"
                ]
            )


def test_exact_v2_chain_is_required():
    x = load(
        AUTH
    )

    a = x[
        "continuation_authorization"
    ]

    assert a[
        "exact_v2_chain_required"
    ] is True

    assert a[
        "exact_r2_binding_required"
    ] is True

    assert a[
        "v1_executor_may_resume_production"
    ] is False


def test_original_execution_start_is_immutable():
    x = load(
        AUTH
    )

    assert x[
        "original_execution_lineage"
    ][
        "execution_start_must_remain_byte_identical"
    ] is True

    assert x[
        "continuation_authorization"
    ][
        "original_execution_start_may_be_modified"
    ] is False

    assert x[
        "original_execution_lineage"
    ][
        "completed_shards_before_repair"
    ] == 0


def test_only_frozen_continuation_actions_are_authorized():
    x = load(
        AUTH
    )

    a = x[
        "continuation_authorization"
    ]

    assert a[
        "phase5af_resume_authorized"
    ] is True

    assert a[
        "accepted_prediction_read_authorized"
    ] is True

    assert a[
        "accepted_outer_label_load_authorized"
    ] is True

    assert a[
        "frozen_threshold_application_authorized"
    ] is True

    assert a[
        "CC_metric_computation_authorized"
    ] is True

    assert a[
        "aggregate_CC_result_generation_authorized"
    ] is True


def test_scientific_changes_are_not_authorized():
    x = load(
        AUTH
    )

    a = x[
        "continuation_authorization"
    ]

    prohibited = (
        "threshold_retuning_authorized",
        "checkpoint_selection_authorized",
        "fault_resampling_authorized",
        "metric_change_authorized",
        "aggregation_change_authorized",
        "uncertainty_change_authorized",
        "stratum_change_authorized",
        "fault_membership_change_authorized",
        "protocol_change_authorized",
        "CSC_generation_authorized",
        "OnField_access_authorized",
        "model_forward_authorized",
        "new_fault_execution_authorized",
    )

    for key in prohibited:
        assert a[
            key
        ] is False


def test_resume_success_markers_must_bind_r3_and_v2():
    x = load(
        AUTH
    )

    r = x[
        "required_continuation_behavior"
    ]

    assert r[
        "write_separate_continuation_start_marker"
    ] is True

    assert r[
        "reuse_only_v2_successful_shard_outputs"
    ] is True

    assert r[
        "reuse_requires_exact_r3_authorization_sha"
    ] is True

    assert r[
        "reuse_requires_exact_v2_executor_sha"
    ] is True

    assert r[
        "unverified_partial_output_reuse"
    ] is False


def test_scientific_invariants_are_fixed():
    x = load(
        AUTH
    )

    s = x[
        "scientific_invariants"
    ]

    assert s[
        "regimes"
    ] == [
        "C0",
        "CC",
    ]

    assert s[
        "CSC"
    ] is False

    assert s[
        "OnField"
    ] is False

    assert s[
        "threshold_comparator"
    ] == ">="

    assert s[
        "nonfinite_imputation"
    ] is False

    assert s[
        "bootstrap_replicates"
    ] == 10000

    assert s[
        "scientific_rules_mutable"
    ] is False


def test_r3_itself_does_not_resume_or_compute():
    x = load(
        AUTH
    )

    assert all(
        value is False
        for value in x[
            "r3_boundary"
        ].values()
    )
