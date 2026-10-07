from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import csc_execution_adapter_v1 as adapter


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/phase6g_csc_pre_forward_adapter_binding_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_6g_csc_pre_forward_adapter_binding_freeze_v1.json"
)


def sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def test_phase6g_freeze_status_and_execution_boundary():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg[
            "status"
        ]
        == "FROZEN_PRE_FORWARD_ADAPTER_BINDING_EXECUTION_DISABLED"
    )

    assert cfg[
        "scientific_change"
    ] is False

    assert cfg[
        "execution_authorized"
    ] is False

    boundary = cfg[
        "adapter_boundary"
    ]

    for key in (
        "imports_torch",
        "imports_numpy",
        "imports_sensor_operator",
        "imports_phase5_executor",
        "reads_outer_arrays",
        "loads_model",
        "executes_sensor_fault",
        "executes_compute_fault",
        "executes_model_forward",
        "computes_metrics",
        "execution_enabled",
    ):
        assert boundary[
            key
        ] is False


def test_frozen_adapter_and_metadata_executor_hashes():
    cfg = load_json(
        CONFIG
    )

    for name in (
        "phase6g_pre_forward_adapter",
        "phase6g_pre_forward_adapter_test",
        "frozen_metadata_executor",
        "frozen_metadata_executor_test",
        "phase6e_execution_plan",
        "phase6f_binding_config",
        "phase6f_binding_manifest",
    ):
        record = cfg[
            "bindings"
        ][
            name
        ]

        path = (
            ROOT
            / record[
                "path"
            ]
        )

        assert sha256_file(
            path
        ) == record[
            "sha256"
        ]


def test_all_pinned_runtime_module_hashes():
    cfg = load_json(
        CONFIG
    )

    for record in cfg[
        "pinned_execution_api_modules"
    ].values():
        path = (
            ROOT
            / record[
                "path"
            ]
        )

        assert sha256_file(
            path
        ) == record[
            "sha256"
        ]


def test_adapter_runtime_remains_hard_disabled():
    assert adapter.EXECUTION_ENABLED is False

    with pytest.raises(
        RuntimeError,
        match="PHASE6G_EXECUTION_DISABLED_PRE_FORWARD_ADAPTER",
    ):
        adapter.execute_request(
            request={}
        )


def test_adapter_source_has_no_execution_runtime_imports():
    path = (
        ROOT
        / "experiments/phase_06/csc_execution_adapter_v1.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    imported = set()

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            imported.update(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ) and node.module:
            imported.add(
                node.module
            )

    assert "torch" not in imported
    assert "numpy" not in imported
    assert "sensor_fi_operators" not in imported
    assert "compute_fi_outer_fleet_executor_v1" not in imported
    assert "compute_fi_execution_harness" not in imported
    assert "compute_fi_fault_only_execution_v1" not in imported


def test_v1_source_helper_export_binding_is_frozen():
    validated = adapter.validate_pinned_bindings()

    helper = validated[
        "modules"
    ][
        "phase4h_runner"
    ][
        "exported_source_helpers"
    ]

    assert (
        helper[
            "binding_style"
        ]
        == "LOCAL_REWINDOW_PLUS_V1_EXPORTED_SOURCE_HELPERS"
    )

    assert (
        helper[
            "v1_sha256"
        ]
        == "bee7b43423723d5ccdbf237d6dbcf70633d9db489d2aca35d1aca26c5100d398"
    )

    assert helper[
        "exports"
    ] == {
        "load_source_trial":
            "load_source_trial",

        "source_trial_path":
            "source_trial_path",
    }


def test_frozen_execution_semantics():
    cfg = load_json(
        CONFIG
    )

    semantics = cfg[
        "frozen_execution_semantics"
    ]

    assert semantics[
        "sensor_corruption_precedes_phase5_window_tensor_conversion"
    ] is True

    assert (
        semantics[
            "transient_compute_execution"
        ]
        == "exactly_one_frozen_R1_selected_window_per_pair"
    )

    assert (
        semantics[
            "persistent_compute_execution"
        ]
        == "frozen_Phase5_onset_through_trial_end_suffix"
    )

    assert semantics[
        "ptq_clean_state_semantics_preserved"
    ] is True

    assert semantics[
        "ptq_weight_route_clean_state_restored_in_finally"
    ] is True

    assert semantics[
        "clean_reference_forward_inside_fault_loop"
    ] is False

    assert semantics[
        "zero_overlap_persistent_pairs_filtered"
    ] is False

    assert semantics[
        "zero_overlap_persistent_pairs_resampled"
    ] is False


def test_freeze_manifest_binds_exact_files():
    manifest = load_json(
        MANIFEST
    )

    assert (
        manifest[
            "status"
        ]
        == "FROZEN_PRE_FORWARD_ADAPTER_BINDING_EXECUTION_DISABLED"
    )

    for record in manifest[
        "frozen_files"
    ]:
        path = (
            ROOT
            / record[
                "path"
            ]
        )

        assert sha256_file(
            path
        ) == record[
            "sha256"
        ]


def test_governance_prohibits_outcome_driven_changes():
    cfg = load_json(
        CONFIG
    )

    governance = cfg[
        "governance"
    ]

    for key in (
        "outer_performance_outcomes_used",
        "validation_used",
        "onfield_used",
        "threshold_retuning",
        "checkpoint_selection",
        "resampling",
        "rebalancing",
        "pair_files_materialized",
        "commit_created",
        "push_performed",
    ):
        assert governance[
            key
        ] is False
