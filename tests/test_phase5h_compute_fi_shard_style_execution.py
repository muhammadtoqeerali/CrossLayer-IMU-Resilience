from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "experiments" / "phase_05"

if str(PHASE5) not in sys.path:
    sys.path.insert(
        0,
        str(PHASE5),
    )

from compute_fi_shard_runtime_v1 import (  # noqa: E402
    run_activation_buffer_sequence,
)


CONFIG = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5h_compute_fi_shard_style_execution_qualification_v1.json"
)

RUNTIME = (
    PHASE5
    / "compute_fi_shard_runtime_v1.py"
)

PHASE5G = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5g_compute_fi_outer_shard_executor_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5H_COMPUTE_FI_SHARD_STYLE_EXECUTION_QUALIFICATION_V1.md"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_contract_scope_is_training_calibration_only():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x["status"]
        == "FROZEN_PRE_QUALIFICATION_SHARD_STYLE_EXECUTION"
    )

    assert x[
        "partition"
    ] == "training_calibration"

    assert x[
        "sequence_length"
    ] == 5

    assert x[
        "persistent_onset_position"
    ] == 2


def test_fixture_matrix_is_complete():
    x = json.loads(
        CONFIG.read_text()
    )[
        "fixture_matrix"
    ]

    assert x[
        "fp32"
    ][
        "shard_fixture_count"
    ] == 4

    assert x[
        "ptq_v7"
    ][
        "shard_fixture_count"
    ] == 10

    assert x[
        "total_shard_fixture_count"
    ] == 14

    assert x[
        "clean_cache_fixture_count"
    ] == 2


def test_expected_masks_are_frozen():
    x = json.loads(
        CONFIG.read_text()
    )[
        "required_semantics"
    ]

    assert x[
        "transient_expected_active_mask"
    ] == [
        True,
        True,
        True,
        True,
        True,
    ]

    assert x[
        "persistent_expected_active_mask"
    ] == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_cross_trial_leakage_forbidden():
    x = json.loads(
        CONFIG.read_text()
    )[
        "required_semantics"
    ]

    assert x[
        "weight_session_reset_required"
    ] is True

    assert x[
        "cross_trial_weight_leakage_allowed"
    ] is False


def test_outer_gate_remains_false():
    x = json.loads(
        PHASE5G.read_text()
    )

    assert x[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ] is False


def test_runtime_hash_binding():
    x = json.loads(
        CONFIG.read_text()
    )

    assert (
        x[
            "frozen_dependencies"
        ][
            "shard_runtime_sha256"
        ]
        == sha(
            RUNTIME
        )
    )


def test_runtime_rejects_empty_sequence():
    with pytest.raises(
        ValueError
    ):
        run_activation_buffer_sequence(
            model=torch.nn.Identity(),
            inputs=[],
            inference_indices=[],
            identities=[],
        )


def test_documentation_records_narrow_syntax_repair():
    text = DOC.read_text()

    assert (
        "omitted one concatenation operator"
        in text
    )

    assert (
        "No fault shard had executed."
        in text
    )


def test_documentation_exists():
    assert DOC.is_file()
