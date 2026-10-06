from __future__ import annotations

import ast
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

from compute_fi_outer_shard_executor_v1 import (  # noqa: E402
    execute_shard,
    validate_plan_structure,
)


CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5g_compute_fi_outer_shard_executor_v1.json"
)

EXECUTOR = (
    PHASE5
    / "compute_fi_outer_shard_executor_v1.py"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5G_COMPUTE_FI_OUTER_SHARD_EXECUTOR_DRY_RUN_V1.md"
)


def load_json(path):
    return json.loads(
        path.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_execution_gate_is_false():
    cfg = load_json(
        CONFIG
    )

    assert cfg[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ] is False

    assert cfg[
        "execution_gate"
    ][
        "dry_run_enabled"
    ] is True

    assert cfg[
        "execution_gate"
    ][
        "execute_shard_enabled"
    ] is False


def test_execute_shard_is_hard_blocked():
    blocked = False

    try:
        execute_shard(
            CONFIG,
            "fixture",
        )

    except RuntimeError as exc:
        blocked = (
            "gate is FALSE"
            in str(
                exc
            )
        )

    assert blocked


def test_executor_has_no_np_or_torch_load():
    source = EXECUTOR.read_text()
    tree = ast.parse(
        source
    )

    prohibited = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if isinstance(
            func,
            ast.Attribute,
        ):
            parts = []
            cur = func

            while isinstance(
                cur,
                ast.Attribute,
            ):
                parts.append(
                    cur.attr
                )
                cur = cur.value

            if isinstance(
                cur,
                ast.Name,
            ):
                parts.append(
                    cur.id
                )

            name = ".".join(
                reversed(
                    parts
                )
            )

        elif isinstance(
            func,
            ast.Name,
        ):
            name = func.id

        else:
            continue

        if name in {
            "np.load",
            "numpy.load",
            "torch.load",
            "np.memmap",
            "numpy.memmap",
        }:
            prohibited.append(
                name
            )

    assert prohibited == []


def test_plan_structure_all_732():
    plan = load_json(
        PLAN
    )

    summary = validate_plan_structure(
        plan
    )

    assert summary[
        "shard_count"
    ] == 732

    assert summary[
        "clean_cache_count"
    ] == 366

    assert summary[
        "fp32_shards"
    ] == 366

    assert summary[
        "ptq_shards"
    ] == 366

    assert summary[
        "transient_shards"
    ] == 366

    assert summary[
        "persistent_shards"
    ] == 366


def test_execution_cardinality_is_exact():
    summary = validate_plan_structure(
        load_json(
            PLAN
        )
    )

    assert summary[
        "total_outer_instance_ids"
    ] == 20170008

    assert summary[
        "clean_model_window_evaluations"
    ] == 1642980

    assert summary[
        "transient_faulted_model_window_evaluations"
    ] == 19715760

    assert summary[
        "persistent_faulted_model_window_evaluations"
    ] == 10083060

    assert summary[
        "total_faulted_model_window_evaluations"
    ] == 29798820

    assert summary[
        "total_model_window_evaluations_including_clean"
    ] == 31441800


def test_config_binds_executor_hash():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg[
            "frozen_dependencies"
        ][
            "executor_implementation"
        ][
            "sha256"
        ]
        == sha(
            EXECUTOR
        )
    )


def test_config_binds_plan_hash():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg[
            "frozen_dependencies"
        ][
            "phase5e_plan"
        ][
            "sha256"
        ]
        == sha(
            PLAN
        )
    )


def test_documentation_exists():
    assert DOC.is_file()
