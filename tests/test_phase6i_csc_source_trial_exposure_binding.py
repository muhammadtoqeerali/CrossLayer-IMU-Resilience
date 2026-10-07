from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import csc_source_trial_exposure_v1 as exposure


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/phase6i_csc_source_trial_exposure_binding_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests/phase_6i_csc_source_trial_exposure_binding_freeze_v1.json"
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


def test_freeze_status_and_execution_boundary():
    cfg = load_json(
        CONFIG
    )

    assert (
        cfg["status"]
        == "FROZEN_METADATA_ONLY_SOURCE_TRIAL_EXPOSURE_HELPER_BINDING"
    )

    assert cfg[
        "scientific_change"
    ] is False

    assert cfg[
        "execution_authorized"
    ] is False

    assert cfg[
        "csc_runtime_implemented"
    ] is False


def test_helper_and_original_test_are_byte_bound():
    cfg = load_json(
        CONFIG
    )

    helper = cfg[
        "frozen_helper"
    ]

    assert sha(
        ROOT
        / helper["path"]
    ) == helper[
        "sha256"
    ]

    assert sha(
        ROOT
        / helper["qualification_test_path"]
    ) == helper[
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
            / record["path"]
        ) == record[
            "sha256"
        ]


def test_finite_and_until_end_support_contract():
    cfg = load_json(
        CONFIG
    )

    support = cfg[
        "frozen_active_support"
    ]

    assert set(
        support[
            "finite_episode_families"
        ]
    ) == {
        "drift",
        "stuck_channel",
        "dropout",
        "frame_loss",
    }

    assert (
        support[
            "finite_support"
        ]
        == "[onset_sample, onset_sample + duration_samples)"
    )

    assert set(
        support[
            "until_end_families"
        ]
    ) == {
        "jitter",
        "delay",
        "orientation",
    }

    assert (
        support[
            "until_end_support"
        ]
        == "[0, source_length)"
    )


def test_exact_overlap_predicate_is_frozen():
    cfg = load_json(
        CONFIG
    )

    geometry = cfg[
        "frozen_retained_window_geometry"
    ]

    assert geometry[
        "window_length_samples"
    ] == 30

    assert geometry[
        "window_stride_samples"
    ] == 15

    assert (
        geometry[
            "exposure_predicate"
        ]
        == (
            "window_start < fault_stop_exclusive "
            "and fault_start < window_stop_exclusive"
        )
    )


def test_r2_census_contract_has_no_fallback():
    cfg = load_json(
        CONFIG
    )

    contract = cfg[
        "r2_census_contract"
    ]

    assert (
        contract[
            "eligible_predicate"
        ]
        == "sensor_exposed_window_count >= 1"
    )

    assert (
        contract[
            "structural_omission_status"
        ]
        == "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR"
    )

    for key in (
        "fallback",
        "resampling",
        "replacement",
        "relocation",
        "rebalancing",
    ):
        assert contract[
            key
        ] is False


def test_helper_runtime_surface_remains_metadata_only():
    cfg = load_json(
        CONFIG
    )

    boundary = cfg[
        "helper_boundary"
    ]

    assert boundary[
        "metadata_only"
    ] is True

    for key in (
        "imports_torch",
        "imports_numpy",
        "imports_sensor_operator",
        "reads_outer_arrays",
        "applies_sensor_fault",
        "loads_model",
        "executes_compute_fault",
        "executes_model_forward",
        "materializes_pair_files",
        "computes_performance_metrics",
    ):
        assert boundary[
            key
        ] is False


def test_source_has_no_execution_imports():
    path = (
        ROOT
        / "experiments/phase_06/csc_source_trial_exposure_v1.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    imported = set()

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imported.update(
                alias.name
                for alias in node.names
            )

        elif (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and node.module
        ):
            imported.add(
                node.module
            )

    assert "torch" not in imported
    assert "numpy" not in imported
    assert "sensor_fi_operators" not in imported
    assert "compute_fi_outer_fleet_executor_v1" not in imported


def test_helper_still_validates_its_pinned_contract():
    observed = (
        exposure.validate_pinned_exposure_contract()
    )

    assert (
        observed[
            "status"
        ]
        == "PINNED_EXPOSURE_CONTRACT_VALID"
    )


def test_manifest_binds_all_frozen_files():
    manifest = load_json(
        MANIFEST
    )

    assert (
        manifest["status"]
        == "FROZEN_METADATA_ONLY_SOURCE_TRIAL_EXPOSURE_HELPER_BINDING"
    )

    assert manifest[
        "execution_authorized"
    ] is False

    for record in manifest[
        "frozen_files"
    ]:
        assert sha(
            ROOT
            / record["path"]
        ) == record[
            "sha256"
        ]
