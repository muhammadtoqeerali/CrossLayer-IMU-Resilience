from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori"
    "/results/phase5o_compute_fi_outer_canary_technical_acceptance_v1"
    "/technical_acceptance.json"
)

PHASE5L = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5l_compute_fi_outer_canary_execution_v1.json"
)

PHASE5M = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_outer_canary_executor_v1.py"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5O_COMPUTE_FI_OUTER_CANARY_TECHNICAL_ACCEPTANCE_V1.md"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_status():
    x = load(RESULT)

    assert (
        x["status"]
        == "TECHNICALLY_ACCEPTED_SINGLE_OUTER_CANARY"
    )


def test_acceptance_scope_is_technical_only():
    x = load(RESULT)

    assert (
        x["acceptance_scope"]
        == "TECHNICAL_INTEGRITY_ONLY_NO_OUTCOME_INTERPRETATION"
    )

    assert x[
        "scientific_interpretation"
    ] == "NONE"


def test_lineage_hashes():
    x = load(RESULT)[
        "frozen_lineage"
    ]

    assert x[
        "phase5e_plan_sha256"
    ] == sha(PLAN)

    assert x[
        "phase5l_config_sha256"
    ] == sha(PHASE5L)

    assert x[
        "phase5m_executor_sha256"
    ] == sha(PHASE5M)


def test_exact_artifact_cardinalities():
    x = load(RESULT)[
        "artifact_integrity"
    ]

    assert x[
        "clean_cache"
    ][
        "record_count"
    ] == 2481

    assert x[
        "fault_shard"
    ][
        "record_count"
    ] == 24810


def test_all_technical_checks_pass():
    x = load(RESULT)[
        "technical_acceptance"
    ]

    assert x

    assert all(
        value is True
        for value in x.values()
    )


def test_no_prediction_or_metric_selection():
    x = load(RESULT)[
        "prohibited_post_canary_uses"
    ]

    assert x

    assert all(
        value is False
        for value in x.values()
    )


def test_full_fleet_still_not_authorized():
    x = load(RESULT)[
        "execution_state"
    ]

    assert x[
        "outer_canary_executed"
    ] is True

    assert x[
        "executed_fault_shards"
    ] == 1

    assert x[
        "remaining_fault_shards_not_executed"
    ] == 731

    assert x[
        "full_fleet_execution_authorized"
    ] is False

    assert x[
        "aggregate_CC_result_generated"
    ] is False

    assert x[
        "CSC_result_generated"
    ] is False


def test_documentation_exists():
    assert DOC.is_file()
