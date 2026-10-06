from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5ag_compute_fi_cc_outcome_technical_acceptance_v1.json"
)

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ag_compute_fi_cc_outcome_technical_acceptance_v1/technical_acceptance.json"
)

EXECUTION_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5af_compute_fi_cc_outcome_execution_v1"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_status_and_check_count():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "TECHNICALLY_ACCEPTED_COMPLETE_PHASE5AF_CC_OUTCOME_EXECUTION"
    )

    assert x[
        "technical_check_count"
    ] == 12

    assert all(
        x[
            "technical_checks"
        ].values()
    )


def test_exact_required_check_set():
    config = load(
        CONFIG
    )

    result = load(
        RESULT
    )

    assert set(
        result[
            "technical_checks"
        ]
    ) == set(
        config[
            "required_checks"
        ]
    )


def test_exact_execution_cardinalities():
    x = load(
        RESULT
    )[
        "accepted_execution"
    ]

    assert x[
        "fault_shards"
    ] == 732

    assert x[
        "grandfathered_r3_v2_transient_shards"
    ] == 1

    assert x[
        "r5_v3_shards"
    ] == 731

    assert x[
        "outer_instance_ids"
    ] == 20170008

    assert x[
        "fault_records"
    ] == 29798820


def test_final_output_hashes_match_acceptance():
    x = load(
        RESULT
    )[
        "accepted_hashes"
    ]

    assert sha(
        EXECUTION_ROOT
        / "aggregate_cc_outcome.json"
    ) == x[
        "aggregate_cc_outcome"
    ]

    assert sha(
        EXECUTION_ROOT
        / "execution_summary.json"
    ) == x[
        "execution_summary"
    ]

    assert sha(
        EXECUTION_ROOT
        / "_SUCCESS.json"
    ) == x[
        "execution_success"
    ]


def test_grandfathered_shard_hashes_match():
    x = load(
        RESULT
    )[
        "accepted_hashes"
    ]

    shard = (
        EXECUTION_ROOT
        / "shards"
        / "p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9"
    )

    assert sha(
        shard
        / "outcome.json"
    ) == x[
        "grandfathered_outcome"
    ]

    assert sha(
        shard
        / "_SUCCESS.json"
    ) == x[
        "grandfathered_success"
    ]


def test_no_temporary_shard_directories():
    shards = (
        EXECUTION_ROOT
        / "shards"
    )

    assert not list(
        shards.glob(
            "*.tmp"
        )
    )


def test_exact_732_completed_shard_directories():
    shards = (
        EXECUTION_ROOT
        / "shards"
    )

    completed = [
        p
        for p in shards.iterdir()
        if p.is_dir()
        and not p.name.endswith(
            ".tmp"
        )
    ]

    assert len(
        completed
    ) == 732


def test_technical_boundary_forbids_interpretation():
    b = load(
        RESULT
    )[
        "technical_acceptance_boundary"
    ]

    assert b[
        "aggregate_payload_deserialized_for_structure"
    ] is True

    assert b[
        "scientific_metric_values_printed"
    ] is False

    assert b[
        "scientific_metric_interpretation"
    ] is False

    assert b[
        "scientific_ranking"
    ] is False


def test_technical_boundary_forbids_selection_and_retuning():
    b = load(
        RESULT
    )[
        "technical_acceptance_boundary"
    ]

    assert b[
        "best_target_selection"
    ] is False

    assert b[
        "best_family_selection"
    ] is False

    assert b[
        "best_variant_selection"
    ] is False

    assert b[
        "threshold_retuning"
    ] is False

    assert b[
        "checkpoint_selection"
    ] is False

    assert b[
        "fault_resampling"
    ] is False


def test_scope_excludes_csc_onfield_and_new_execution():
    b = load(
        RESULT
    )[
        "technical_acceptance_boundary"
    ]

    assert b[
        "CSC_generated"
    ] is False

    assert b[
        "OnField_used"
    ] is False

    assert b[
        "model_forward_executed"
    ] is False

    assert b[
        "new_fault_execution"
    ] is False
