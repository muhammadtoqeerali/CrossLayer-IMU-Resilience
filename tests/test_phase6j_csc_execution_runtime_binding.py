from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import csc_execution_runtime_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/phase6j_csc_execution_runtime_binding_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_6j_csc_execution_runtime_binding_freeze_v1.json"
)


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def runtime_source() -> str:
    return (
        ROOT
        / "experiments/phase_06/csc_execution_runtime_v1.py"
    ).read_text(
        encoding="utf-8"
    )


def test_freeze_status_and_execution_boundary():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg["status"]
        == "FROZEN_PRE_AUTHORIZATION_CSC_EXECUTION_RUNTIME_BINDING"
    )

    assert cfg[
        "scientific_change"
    ] is False

    assert cfg[
        "execution_authorized"
    ] is False

    assert cfg[
        "csc_runtime_implemented"
    ] is True

    assert cfg[
        "production_outer_execution_body_released"
    ] is False


def test_runtime_and_qualification_test_are_byte_bound():
    cfg = load_json(
        CONFIG
    )

    frozen = cfg[
        "frozen_runtime"
    ]

    assert sha(
        ROOT
        / frozen[
            "path"
        ]
    ) == frozen[
        "sha256"
    ]

    assert sha(
        ROOT
        / frozen[
            "qualification_test_path"
        ]
    ) == frozen[
        "qualification_test_sha256"
    ]


def test_all_upstream_bindings_are_byte_exact():
    cfg = load_json(
        CONFIG
    )

    for record in cfg[
        "upstream_bindings"
    ].values():
        assert sha(
            ROOT
            / record[
                "path"
            ]
        ) == record[
            "sha256"
        ]


def test_public_execution_remains_hard_blocked_before_heavy_imports():
    source = runtime_source()
    tree = ast.parse(
        source
    )

    execute = next(
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "execute_shard"
        )
    )

    body = (
        ast.get_source_segment(
            source,
            execute,
        )
        or ""
    )

    assert runtime.EXECUTION_AUTHORIZED is False

    assert (
        runtime.OUTER_EXECUTION_BLOCK
        == "PHASE6J_OUTER_EXECUTION_NOT_AUTHORIZED"
    )

    assert (
        "if EXECUTION_AUTHORIZED is not True"
        in body
    )

    assert (
        body.index(
            "if EXECUTION_AUTHORIZED is not True"
        )
        < body.index(
            "load_bound_runtime_modules"
        )
    )

    assert (
        "PHASE6J_AUTHORIZED_EXECUTION_BODY_NOT_YET_RELEASED"
        in body
    )


def test_runtime_contract_covers_phase6h_required_orchestration():
    cfg = load_json(
        CONFIG
    )

    contract = cfg[
        "implemented_runtime_contracts"
    ]

    required_true = (
        "execution_shard_identity",
        "canonical_pair_member_order",
        "phase6i_source_trial_exposure_helper_bound",
        "stored_window_sensor_conditioning",
        "source_trial_sensor_conditioning",
        "sensor_corruption_precedes_compute_fault",
        "source_trial_fault_precedes_rewindow",
        "phase5_window_tensor_bound",
        "phase5_execute_fault_sequence_bound",
        "phase5_execution_active_mask_bound",
        "sensor_reference_generated_before_compute_fault_sequence",
        "zero_temporal_overlap_pairs_retained",
        "model_bundle_load_once_per_stream",
        "phase6_local_atomic_commit",
        "whole_shard_recompute_on_explicit_recompute_partial",
    )

    for key in required_true:
        assert contract[
            key
        ] is True

    assert contract[
        "phase6_c0_recompute"
    ] is False

    assert contract[
        "phase5_commit_atomic_artifact_reused"
    ] is False

    assert contract[
        "row_level_resume"
    ] is False

    assert (
        contract[
            "phase5_clean_cache_missing_or_invalid_action"
        ]
        == "ABORT_SHARD_NO_PHASE6_RECOMPUTE"
    )


def test_runtime_source_binds_clean_cache_exposure_and_phase5_compute():
    source = runtime_source()

    required_tokens = (
        "ABORT_SHARD_NO_PHASE6_RECOMPUTE",
        "validate_pinned_exposure_contract",
        "source_trial_exposed_window_indices",
        'parent_kind == "stored_window"',
        'parent_kind == "source_trial"',
        "operators.apply_fault",
        "runner.rewindow_sequence",
        "phase5_executor._window_tensor",
        "phase5_executor.execute_fault_sequence",
        "phase5_executor.execution_active_mask",
    )

    for token in required_tokens:
        assert token in source


def test_phase6_local_atomic_contract_remains_distinct_from_phase5_commit():
    source = runtime_source()
    tree = ast.parse(
        source
    )

    commit = next(
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "commit_phase6h_artifact"
        )
    )

    body = (
        ast.get_source_segment(
            source,
            commit,
        )
        or ""
    )

    assert "write_json" in body
    assert "os.replace" in body

    assert (
        body.index(
            "write_json"
        )
        < body.index(
            "os.replace"
        )
    )

    assert (
        "commit_atomic_artifact"
        not in body
    )


def test_ptq_weight_reset_is_delegated_to_pinned_phase5_finally():
    cfg = load_json(
        CONFIG
    )

    phase5 = (
        ROOT
        / cfg[
            "upstream_bindings"
        ][
            "phase5_executor"
        ][
            "path"
        ]
    )

    source = phase5.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    execute = next(
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "execute_fault_sequence"
        )
    )

    observed = False

    for node in ast.walk(
        execute
    ):
        if not isinstance(
            node,
            ast.Try,
        ):
            continue

        final_source = "\n".join(
            ast.get_source_segment(
                source,
                item,
            )
            or ""
            for item in node.finalbody
        )

        if "session.reset_clean()" in final_source:
            observed = True
            break

    assert observed is True


def test_qualification_and_governance_are_frozen():
    cfg = load_json(
        CONFIG
    )

    q = cfg[
        "qualification_entering_freeze"
    ]

    assert q[
        "dedicated_phase6j_tests_passed"
    ] == 22

    assert q[
        "phase6_tests_passed"
    ] == 148

    assert q[
        "full_repository_tests_passed"
    ] == 998

    assert q[
        "historical_phase5r_lifecycle_test_deselected"
    ] == 1

    governance = cfg[
        "governance"
    ]

    for key in (
        "validation_used",
        "outer_performance_outcomes_used",
        "onfield_used",
        "threshold_retuning",
        "checkpoint_selection",
        "resampling",
        "replacement",
        "relocation",
        "rebalancing",
        "commit_created",
        "push_performed",
    ):
        assert governance[
            key
        ] is False


def test_manifest_binds_all_phase6j_frozen_files():
    manifest = load_json(
        MANIFEST
    )

    assert (
        manifest[
            "status"
        ]
        == "FROZEN_PRE_AUTHORIZATION_CSC_EXECUTION_RUNTIME_BINDING"
    )

    assert manifest[
        "execution_authorized"
    ] is False

    assert manifest[
        "production_outer_execution_body_released"
    ] is False

    for record in manifest[
        "frozen_files"
    ]:
        assert sha(
            ROOT
            / record[
                "path"
            ]
        ) == record[
            "sha256"
        ]
