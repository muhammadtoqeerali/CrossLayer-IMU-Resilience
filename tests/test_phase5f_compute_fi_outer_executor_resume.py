from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PHASE5 = (
    ROOT
    / "experiments"
    / "phase_05"
)

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_outer_executor_v1 import (  # noqa: E402
    artifact_is_reusable,
    build_outer_instance_id,
    canonical_json,
    commit_atomic_artifact,
    prepare_atomic_artifact,
    sha256_file,
    write_json,
)


EXECUTOR = (
    PHASE5
    / "compute_fi_outer_executor_v1.py"
)

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5f_compute_fi_outer_executor_qualification_v1.json"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5F_COMPUTE_FI_OUTER_EXECUTOR_RESUME_QUALIFICATION_V1.md"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_prequalification_contract():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x["status"]
        == "FROZEN_PRE_QUALIFICATION_EXECUTOR_RESUME_CONTRACT"
    )

    scope = x[
        "qualification_scope"
    ]

    assert scope[
        "synthetic_fixture"
    ] is True

    assert scope[
        "training_calibration_fixture"
    ] is True

    assert scope[
        "outer_test_payload_allowed"
    ] is False

    assert scope[
        "outer_test_model_forward_allowed"
    ] is False

    assert scope[
        "outer_shard_execution_allowed"
    ] is False


def test_phase5e_plan_hash_binding():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x[
            "frozen_dependencies"
        ][
            "phase5e_plan_sha256"
        ]
        == sha(
            PLAN
        )
    )


def test_executor_hash_binding():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x[
            "frozen_dependencies"
        ][
            "executor_sha256"
        ]
        == sha(
            EXECUTOR
        )
    )


def test_resume_contract_is_strict():
    x = json.loads(
        CONFIG.read_text()
    )[
        "resume_contract"
    ]

    assert x[
        "success_marker_written_last"
    ] is True

    assert x[
        "reuse_requires_plan_hash"
    ] is True

    assert x[
        "reuse_requires_executor_hash"
    ] is True

    assert x[
        "reuse_requires_all_output_hashes"
    ] is True

    assert x[
        "partial_final_reuse"
    ] is False

    assert x[
        "partial_temp_reuse"
    ] is False


def test_atomic_first_write_and_reuse(tmp_path):
    plan_sha = sha(
        PLAN
    )

    executor_sha = sha(
        EXECUTOR
    )

    state, temp, success = prepare_atomic_artifact(
        output_root=tmp_path,
        artifact_id="fixture",
        artifact_kind="fault_shard",
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        recompute_partial=False,
    )

    assert state == "compute"
    assert temp is not None
    assert success is None

    output = (
        temp
        / "coverage.json"
    )

    output_sha = write_json(
        output,
        {
            "complete":
                True,
        },
    )

    commit_atomic_artifact(
        output_root=tmp_path,
        artifact_id="fixture",
        artifact_kind="fault_shard",
        temp_dir=temp,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        output_hashes={
            "coverage.json":
                output_sha,
        },
        coverage={
            "complete":
                True,
        },
    )

    assert artifact_is_reusable(
        output_root=tmp_path,
        artifact_id="fixture",
        artifact_kind="fault_shard",
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
    )


def test_partial_final_rejected(tmp_path):
    artifact = (
        tmp_path
        / "fixture"
    )

    artifact.mkdir()

    rejected = False

    try:
        prepare_atomic_artifact(
            output_root=tmp_path,
            artifact_id="fixture",
            artifact_kind="fault_shard",
            plan_sha256=sha(
                PLAN
            ),
            executor_sha256=sha(
                EXECUTOR
            ),
            recompute_partial=False,
        )

    except ValueError:
        rejected = True

    assert rejected


def test_partial_temp_rejected(tmp_path):
    temp = (
        tmp_path
        / "fixture.partial"
    )

    temp.mkdir()

    rejected = False

    try:
        prepare_atomic_artifact(
            output_root=tmp_path,
            artifact_id="fixture",
            artifact_kind="fault_shard",
            plan_sha256=sha(
                PLAN
            ),
            executor_sha256=sha(
                EXECUTOR
            ),
            recompute_partial=False,
        )

    except ValueError:
        rejected = True

    assert rejected


def test_documentation_exists():
    assert DOC.is_file()
