from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AUTH = (
    ROOT
    / "configs/evaluation/"
    "phase5af_r5_v3_continuation_authorization_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status():
    x = load(
        AUTH
    )

    assert (
        x["status"]
        == "FROZEN_POST_BOUNDARY_V3_CC_CONTINUATION_AUTHORIZATION"
    )


def test_hash_bound_repo_dependencies():
    x = load(
        AUTH
    )

    sections = (
        x[
            "scientific_authority"
        ],
        x[
            "repair_evidence"
        ],
        x[
            "authorized_v3_chain"
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
                rec["path"]
            )

            if not path.is_absolute():
                path = ROOT / path

            assert path.is_file()

            assert sha(
                path
            ) == rec[
                "sha256"
            ]


def test_exactly_one_grandfathered_shard():
    x = load(
        AUTH
    )

    g = x[
        "grandfathered_completed_shard"
    ]

    assert g[
        "count"
    ] == 1

    assert g[
        "persistence"
    ] == "transient_one_inference"

    assert g[
        "reuse_authorized"
    ] is True

    assert g[
        "delete_authorized"
    ] is False

    assert g[
        "rerun_authorized"
    ] is False


def test_grandfathered_shard_hashes_are_exact():
    x = load(
        AUTH
    )

    g = x[
        "grandfathered_completed_shard"
    ]

    for key in (
        "outcome",
        "success_marker",
    ):
        rec = g[
            key
        ]

        path = Path(
            rec[
                "path"
            ]
        )

        assert path.is_file()

        assert sha(
            path
        ) == rec[
            "sha256"
        ]


def test_remaining_shard_accounting():
    x = load(
        AUTH
    )

    r = x[
        "remaining_execution"
    ]

    assert r[
        "total_phase5e_shards"
    ] == 732

    assert r[
        "grandfathered_completed_shards"
    ] == 1

    assert r[
        "remaining_shards"
    ] == 731

    assert r[
        "remaining_shards_must_use_v3"
    ] is True

    assert r[
        "skip_only_exact_grandfathered_shard"
    ] is True


def test_only_v3_is_authorized_for_remaining_shards():
    x = load(
        AUTH
    )

    a = x[
        "continuation_authorization"
    ]

    assert a[
        "exact_v3_chain_required"
    ] is True

    assert a[
        "grandfathered_transient_shard_reuse_authorized"
    ] is True

    assert a[
        "grandfathered_transient_shard_rerun_authorized"
    ] is False

    assert a[
        "any_other_v2_shard_reuse_authorized"
    ] is False


def test_scientific_changes_remain_prohibited():
    x = load(
        AUTH
    )

    a = x[
        "continuation_authorization"
    ]

    for key in (
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
    ):
        assert a[
            key
        ] is False


def test_resume_marker_requirements():
    x = load(
        AUTH
    )

    r = x[
        "required_resume_behavior"
    ]

    assert r[
        "write_separate_r5_v3_continuation_marker"
    ] is True

    assert r[
        "continuation_marker_must_bind_r5_authorization_sha"
    ] is True

    assert r[
        "continuation_marker_must_bind_v3_executor_sha"
    ] is True

    assert r[
        "continuation_marker_must_bind_preserved_shard_hashes"
    ] is True

    assert r[
        "do_not_reexecute_preserved_shard"
    ] is True

    assert r[
        "process_remaining_731_shards"
    ] is True


def test_r5_itself_did_not_resume():
    x = load(
        AUTH
    )

    assert all(
        value is False
        for value in x[
            "r5_boundary"
        ].values()
    )
