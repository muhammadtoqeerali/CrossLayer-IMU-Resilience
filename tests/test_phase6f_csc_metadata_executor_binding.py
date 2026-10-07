from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import csc_outer_executor_v1 as csc


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/phase6f_csc_metadata_executor_binding_v1.json"
)

DOC = (
    ROOT
    / "docs/PHASE_6F_CSC_METADATA_EXECUTOR_BINDING_V1.md"
)

TEST = (
    ROOT
    / "tests/test_phase6f_csc_metadata_executor_binding.py"
)

MANIFEST = (
    ROOT
    / "manifests/phase_6f_csc_metadata_executor_binding_freeze_v1.json"
)


def load(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha(path: Path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_phase6f_status_and_role():
    cfg = load(CONFIG)

    assert (
        cfg["schema_version"]
        == "phase6f_csc_metadata_executor_binding_v1"
    )

    assert cfg["phase"] == "6F"

    assert (
        cfg["status"]
        == "FROZEN_METADATA_EXECUTOR_BINDING_PRE_EXECUTION"
    )

    assert cfg["scientific_change"] is False


def test_phase6e_parent_artifacts_are_byte_frozen():
    cfg = load(CONFIG)

    for record in cfg[
        "frozen_phase6e_plan"
    ].values():
        path = ROOT / record["path"]

        assert path.is_file()
        assert sha(path) == record["sha256"]


def test_bound_executor_and_test_are_byte_frozen():
    cfg = load(CONFIG)

    implementation = cfg[
        "qualified_metadata_executor"
    ][
        "implementation"
    ]

    regression = cfg[
        "qualified_metadata_executor"
    ][
        "regression_test"
    ]

    assert (
        sha(
            ROOT
            / implementation["path"]
        )
        == implementation["sha256"]
        == "99320b5811c5e72940bcfc779fa530df88fb8b7a2f8662711b8f56e4cf5d61fd"
    )

    assert (
        sha(
            ROOT
            / regression["path"]
        )
        == regression["sha256"]
        == "5da200d643c1c9333efba7e5fa38d32aa02ba72f8c8f8aa5eeed4451c882f0c2"
    )


def test_bound_scientific_surface_is_unchanged():
    cfg = load(CONFIG)

    surface = cfg[
        "bound_scientific_surface"
    ]

    assert (
        surface[
            "model_independent_csc_pair_count"
        ]
        == 4237835
    )

    assert (
        surface[
            "seed_variant_expanded_pair_member_count"
        ]
        == 21793038
    )

    assert (
        surface[
            "pair_surface_binding_sha256"
        ]
        == "3b39a909ea140db75da52315abcf5efe2748fcdb2d31f0fc98ddf766b000af53"
    )

    assert (
        surface[
            "combined_compute_coordinate_binding_sha256"
        ]
        == "2117c7b07084566607ccac46db150faae81d28bef51d0dcab7c700627d8eb435"
    )

    assert (
        surface[
            "temporal_workload_binding_sha256"
        ]
        == "5b447bca2f45d4889bcf0dc86d6170c40a1028171a9f1b0ea7a28e2884f12ba8"
    )


def test_execution_contract_is_hard_disabled():
    cfg = load(CONFIG)

    contract = cfg[
        "execution_contract"
    ]

    assert contract[
        "execution_enabled"
    ] is False

    assert contract[
        "imports_torch"
    ] is False

    assert contract[
        "imports_sensor_fault_operator_module"
    ] is False

    assert contract[
        "loads_model"
    ] is False

    assert contract[
        "reads_outer_arrays_during_validate_config"
    ] is False

    assert contract[
        "materializes_pair_files"
    ] is False

    assert contract[
        "calls_sensor_apply_fault"
    ] is False

    assert contract[
        "executes_compute_fault"
    ] is False

    assert contract[
        "executes_model_forward"
    ] is False

    assert contract[
        "computes_csc_metrics"
    ] is False

    assert (
        contract[
            "execution_capability_may_be_added_without_new_freeze"
        ]
        is False
    )


def test_runtime_hard_stop_is_still_present():
    assert csc.EXECUTION_ENABLED is False

    with pytest.raises(
        RuntimeError,
        match="PHASE6E_EXECUTION_DISABLED_METADATA_ONLY",
    ):
        csc.execute_shard(
            shard_id="phase6f-binding-test"
        )


def test_runtime_source_has_no_execution_imports():
    source = inspect.getsource(
        csc
    )

    assert "import torch" not in source
    assert "from torch" not in source
    assert "import sensor_fi_operators" not in source

    assert "execute_fault_sequence(" not in source
    assert "load_model_bundle(" not in source


def test_governance_is_outcome_independent():
    cfg = load(CONFIG)

    governance = cfg[
        "governance"
    ]

    for field in (
        "outer_performance_outcomes_used",
        "cs_performance_outcomes_used",
        "cc_performance_outcomes_used",
        "csc_performance_outcomes_used",
        "validation_performance_used",
        "onfield_used",
        "threshold_retuning",
        "checkpoint_selection",
        "resampling",
        "rebalancing",
        "pair_materialization",
        "model_forward",
        "commit_created",
        "push_performed",
    ):
        assert governance[field] is False


def test_manifest_integrity():
    manifest = load(MANIFEST)

    assert (
        manifest["schema_version"]
        == "phase6f_csc_metadata_executor_binding_freeze_v1"
    )

    assert (
        manifest["status"]
        == "FROZEN_METADATA_EXECUTOR_BINDING_PRE_EXECUTION"
    )

    artifacts = manifest[
        "artifacts"
    ]

    assert artifacts[
        "config"
    ][
        "sha256"
    ] == sha(CONFIG)

    assert artifacts[
        "documentation"
    ][
        "sha256"
    ] == sha(DOC)

    assert artifacts[
        "regression_test"
    ][
        "sha256"
    ] == sha(TEST)

    assert (
        manifest[
            "bound_implementation"
        ][
            "executor_sha256"
        ]
        == "99320b5811c5e72940bcfc779fa530df88fb8b7a2f8662711b8f56e4cf5d61fd"
    )

    assert (
        manifest[
            "bound_implementation"
        ][
            "executor_test_sha256"
        ]
        == "5da200d643c1c9333efba7e5fa38d32aa02ba72f8c8f8aa5eeed4451c882f0c2"
    )
