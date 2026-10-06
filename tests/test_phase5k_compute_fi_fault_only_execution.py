from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

MODULE = (
    PHASE5
    / "compute_fi_fault_only_execution_v1.py"
)

CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5k_compute_fi_fault_only_execution_qualification_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5K_COMPUTE_FI_FAULT_ONLY_EXECUTION_QUALIFICATION_V1.md"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_config_scope():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x["status"]
        == "FROZEN_PRE_OUTER_FAULT_ONLY_EXECUTION_QUALIFICATION"
    )

    assert x[
        "partition"
    ] == "training_calibration"

    assert x[
        "qualification_matrix"
    ][
        "total_cases"
    ] == 14


def test_module_hash_binding():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x[
            "frozen_dependencies"
        ][
            "fault_only_module_sha256"
        ]
        == sha(
            MODULE
        )
    )


def test_no_paired_runner_delegation():
    source = MODULE.read_text()

    assert "run_fp32_paired" not in source
    assert "run_ptq_activation_buffer_paired" not in source
    assert ".run_paired(" not in source
    assert "clean_reference_model" not in source


def test_expected_public_api():
    tree = ast.parse(
        MODULE.read_text()
    )

    functions = {
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }

    classes = {
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    }

    assert {
        "run_fp32_fault_only",
        "run_ptq_activation_buffer_fault_only",
    }.issubset(
        functions
    )

    assert "PTQWeightFaultOnlySession" in classes


def test_no_outer_partition_literal():
    source = MODULE.read_text().lower()

    assert "outer_test" not in source
    assert "onfield" not in source


def test_documentation_exists():
    assert DOC.is_file()
