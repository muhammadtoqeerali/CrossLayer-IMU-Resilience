from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

PHASE05 = (
    ROOT
    / "experiments/phase_05"
)

sys.path.insert(
    0,
    str(PHASE05),
)

import cc_outcome_executor_v1 as executor  # noqa: E402


RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5ad_compute_fi_cc_outcome_executor_qualification_v1/qualification.json"
)

EXECUTOR = (
    ROOT
    / "experiments/phase_05/"
    "cc_outcome_executor_v1.py"
)

QUALIFIER = (
    ROOT
    / "experiments/phase_05/"
    "qualify_cc_outcome_executor_v1.py"
)

PROTOCOL = (
    ROOT
    / "configs/evaluation/"
    "phase5z_compute_fi_cc_outcome_analysis_protocol_v1.json"
)

GATE = (
    ROOT
    / "configs/evaluation/"
    "phase5ac_compute_fi_cc_pre_outcome_execution_gate_v1.json"
)

PLAN = (
    ROOT
    / "manifests/"
    "phase_5e_compute_fi_outer_execution_plan_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_qualification_result():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "QUALIFIED_FINAL_CC_OUTCOME_EXECUTOR_PRE_OUTCOME"
    )

    assert (
        x["qualification_check_count"]
        == len(
            x[
                "qualification_checks"
            ]
        )
    )

    assert (
        x["qualification_check_count"]
        >= 10
    )

    assert all(
        x[
            "qualification_checks"
        ].values()
    )


def test_result_binds_current_executor_and_qualifier():
    x = load(
        RESULT
    )

    assert (
        x["executor_sha256"]
        == sha(
            EXECUTOR
        )
    )

    assert (
        x["qualifier_sha256"]
        == sha(
            QUALIFIER
        )
    )


def test_production_root_mismatch_fails_before_execution():
    with pytest.raises(
        executor.OutcomeExecutorError,
        match="production outer root differs",
    ):
        executor.validate_lineage(
            protocol_path=PROTOCOL,
            gate_path=GATE,
            plan_path=PLAN,
            outer_root="/not-the-frozen-root",
            dataset_root="/not-the-frozen-dataset",
            qualification_mode=False,
        )


def test_qualification_boundary():
    x = load(
        RESULT
    )

    boundary = x[
        "qualification_boundary"
    ]

    assert boundary[
        "synthetic_estates_only"
    ] is True

    for key, value in boundary.items():
        if key == "synthetic_estates_only":
            continue

        assert value is False


def test_executor_implementation_boundary():
    x = load(
        RESULT
    )

    boundary = x[
        "implementation_boundary"
    ]

    assert boundary[
        "final_outcome_executor_implemented"
    ] is True

    assert boundary[
        "final_outcome_executor_qualified"
    ] is True

    assert boundary[
        "production_execution_performed"
    ] is False

    assert boundary[
        "accepted_outcome_analysis_executed"
    ] is False


def test_expected_major_checks_exist():
    x = load(
        RESULT
    )

    checks = x[
        "qualification_checks"
    ]

    required = {
        "production_path_binding_without_payload_access",
        "phase5m_style_frozen_hash_path",
        "phase5r_integrity_path",
        "twelve_synthetic_subject_seed_persistence_jobs",
        "transient_end_to_end",
        "persistent_end_to_end",
        "three_operating_points",
        "paired_clean_faulted_and_degradation_outputs",
        "equal_seed_then_subject_aggregation",
        "subject_cluster_bootstrap",
        "clean_baseline_deduplicated_across_persistence",
        "missing_timing_never_imputed",
        "synthetic_label_and_risk_join",
        "exact_lineage_binding",
    }

    assert required.issubset(
        set(
            checks
        )
    )


def test_executor_has_no_hardcoded_accepted_root():
    source = EXECUTOR.read_text(
        encoding="utf-8"
    )

    assert (
        "phase5_outer_compute_fi_prospective_v1"
        not in source
    )
