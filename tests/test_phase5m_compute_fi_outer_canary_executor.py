from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

EXECUTOR = (
    PHASE5
    / "compute_fi_outer_canary_executor_v1.py"
)

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5l_compute_fi_outer_canary_execution_v1.json"
)

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5M_COMPUTE_FI_OUTER_CANARY_EXECUTOR_STATIC_QUALIFICATION_V1.md"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "phase5m_executor",
        EXECUTOR,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    return module


def test_expected_config_hash_is_hard_bound():
    module = load_module()

    assert (
        module.EXPECTED_PHASE5L_CONFIG_SHA256
        == hashlib.sha256(
            CONFIG.read_bytes()
        ).hexdigest()
    )


def test_validate_configuration_exact_canary():
    module = load_module()

    validated = module.validate_configuration(
        CONFIG
    )

    assert len(
        validated[
            "trial_rows"
        ]
    ) == 44

    assert sum(
        int(
            row[
                "window_count"
            ]
        )
        for row in validated[
            "trial_rows"
        ]
    ) == 2481

    assert (
        validated[
            "config"
        ][
            "authorization"
        ][
            "authorized_shard_count"
        ]
        == 1
    )


def test_authorized_shard_is_plan_zero():
    module = load_module()

    validated = module.validate_configuration(
        CONFIG
    )

    plan = json.loads(
        PLAN.read_text()
    )

    shard = module.validate_requested_shard(
        validated,
        plan[
            "shards"
        ][
            0
        ][
            "shard_id"
        ],
    )

    assert shard == plan[
        "shards"
    ][
        0
    ]


def test_second_shard_rejected():
    module = load_module()

    validated = module.validate_configuration(
        CONFIG
    )

    plan = json.loads(
        PLAN.read_text()
    )

    with pytest.raises(
        ValueError,
        match="not authorized",
    ):
        module.validate_requested_shard(
            validated,
            plan[
                "shards"
            ][
                1
            ][
                "shard_id"
            ],
        )


def test_validation_functions_cannot_load_outer_arrays_or_model():
    source = EXECUTOR.read_text()
    tree = ast.parse(
        source
    )

    functions = {
        node.name:
            node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }

    prohibited = {
        "np.load",
        "torch.load",
        "load_frozen_fp32_model",
        "run_fp32_fault_only",
        "_load_trial_signal",
        "execute_canary",
    }

    def names(node):
        result = set()

        for child in ast.walk(
            node
        ):
            if not isinstance(
                child,
                ast.Call,
            ):
                continue

            func = child.func

            if isinstance(
                func,
                ast.Name,
            ):
                result.add(
                    func.id
                )

            elif isinstance(
                func,
                ast.Attribute,
            ) and isinstance(
                func.value,
                ast.Name,
            ):
                result.add(
                    f"{func.value.id}.{func.attr}"
                )

        return result

    for name in (
        "validate_configuration",
        "validate_requested_shard",
    ):
        assert not (
            names(
                functions[
                    name
                ]
            )
            & prohibited
        )


def test_only_execution_loader_uses_numpy_load():
    source = EXECUTOR.read_text()
    tree = ast.parse(
        source
    )

    owners = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        for child in ast.walk(
            node
        ):
            if not isinstance(
                child,
                ast.Call,
            ):
                continue

            func = child.func

            if (
                isinstance(
                    func,
                    ast.Attribute,
                )
                and isinstance(
                    func.value,
                    ast.Name,
                )
                and func.value.id == "np"
                and func.attr == "load"
            ):
                owners.append(
                    node.name
                )

    assert owners == [
        "_load_trial_signal"
    ]


def test_no_label_file_access_or_paired_runner():
    source = EXECUTOR.read_text()

    assert "labels.npy" not in source
    assert "run_fp32_paired" not in source
    assert "run_ptq_activation_buffer_paired" not in source
    assert ".run_paired(" not in source


def test_execution_path_uses_phase5k_and_phase5f_atomic_core():
    source = EXECUTOR.read_text()

    assert "run_fp32_fault_only(" in source
    assert "prepare_atomic_artifact(" in source
    assert "commit_atomic_artifact(" in source
    assert "artifact_is_reusable(" in source


def test_forward_budget_constants():
    module = load_module()

    assert module.EXPECTED_TRIAL_COUNT == 44
    assert module.EXPECTED_WINDOW_COUNT == 2481
    assert module.EXPECTED_TARGET_COUNT == 10
    assert module.EXPECTED_CLEAN_FORWARD_COUNT == 2481
    assert module.EXPECTED_FAULT_FORWARD_COUNT == 24810
    assert module.EXPECTED_TOTAL_FORWARD_COUNT == 27291


def test_documentation_exists():
    assert DOC.is_file()
