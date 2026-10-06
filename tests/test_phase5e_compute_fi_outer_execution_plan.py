from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PLAN = (
    ROOT
    / "manifests"
    / "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

PLANNER = (
    ROOT
    / "experiments"
    / "phase_05"
    / "compute_fi_outer_execution_plan_v1.py"
)

PHASE5D = (
    ROOT
    / "configs"
    / "evaluation"
    / "phase5d_compute_fi_outer_protocol_v1.json"
)

DOC = (
    ROOT
    / "docs"
    / "PHASE_5E_COMPUTE_FI_OUTER_EXECUTION_PLAN_V1.md"
)


def load_plan():
    return json.loads(
        PLAN.read_text()
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_status_and_inventory():
    x = load_plan()

    assert (
        x["status"]
        == "QUALIFIED_FROZEN_DERIVED_EXECUTION_PLAN"
    )

    assert x[
        "inventory"
    ][
        "subject_count"
    ] == 61

    assert x[
        "inventory"
    ][
        "trial_count"
    ] == 6309

    assert x[
        "inventory"
    ][
        "window_count"
    ] == 273830


def test_exact_inventory_bindings():
    x = load_plan()[
        "inventory"
    ]

    assert (
        x[
            "trial_inventory_sha256"
        ]
        == "ffdd772db741c53ba452054e00ed8d9cba76412ed11ac4801edf4f5694f994d3"
    )

    assert (
        x[
            "subject_inventory_sha256"
        ]
        == "139aa8efcf613770300f6cafbe04eb690d0ac20384bf914b72d5cc8dcd110208"
    )

    assert (
        x[
            "persistent_onset_binding_sha256"
        ]
        == "4d17d427c2f0af5e30405c641b3a5f50d5eaf3c8aeae872a90be6a11876705d2"
    )


def test_header_only_boundary():
    b = load_plan()[
        "derivation_boundary"
    ]

    assert b[
        "segments_npy_header_bytes_read"
    ] is True

    for key in (
        "segments_npy_payload_bytes_read",
        "labels_npy_opened",
        "np_load_used",
        "memmap_used",
        "signal_array_materialized",
        "model_loaded",
        "model_forward_executed",
        "fault_injection_executed",
        "outer_prediction_read",
        "onfield_used",
    ):
        assert b[
            key
        ] is False, key


def test_planner_has_no_array_loader_calls():
    source = PLANNER.read_text()
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
            current = func

            while isinstance(
                current,
                ast.Attribute,
            ):
                parts.append(
                    current.attr
                )
                current = current.value

            if isinstance(
                current,
                ast.Name,
            ):
                parts.append(
                    current.id
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
            "np.memmap",
            "numpy.memmap",
            "np.lib.format.open_memmap",
            "numpy.lib.format.open_memmap",
            "torch.load",
        }:
            prohibited.append(
                name
            )

    assert prohibited == []


def test_732_fault_shards():
    x = load_plan()

    shards = x[
        "shards"
    ]

    assert len(
        shards
    ) == 732

    assert len({
        row[
            "shard_id"
        ]
        for row in shards
    }) == 732

    assert x[
        "sharding"
    ][
        "shard_count"
    ] == 732


def test_366_clean_caches():
    x = load_plan()

    clean = x[
        "clean_caches"
    ]

    assert len(
        clean
    ) == 366

    assert len({
        row[
            "clean_cache_id"
        ]
        for row in clean
    }) == 366

    assert x[
        "clean_cache"
    ][
        "cache_count"
    ] == 366


def test_frozen_outer_instance_cardinality():
    x = load_plan()[
        "expected_totals"
    ]

    assert x[
        "unique_sampling_ids"
    ] == 3921946

    assert x[
        "transient_outer_instance_ids"
    ] == 19715760

    assert x[
        "persistent_outer_instance_ids"
    ] == 454248

    assert x[
        "total_outer_instance_ids"
    ] == 20170008


def test_clean_forward_cardinality():
    assert (
        load_plan()[
            "expected_totals"
        ][
            "clean_model_window_evaluations"
        ]
        == 1642980
    )


def test_transient_forward_count_equals_transient_instances():
    x = load_plan()[
        "expected_totals"
    ]

    assert (
        x[
            "transient_faulted_model_window_evaluations"
        ]
        == x[
            "transient_outer_instance_ids"
        ]
    )


def test_persistent_forward_exposure_is_derived_and_larger():
    x = load_plan()[
        "expected_totals"
    ]

    assert (
        x[
            "persistent_faulted_model_window_evaluations"
        ]
        > x[
            "persistent_outer_instance_ids"
        ]
    )


def test_total_forward_arithmetic():
    x = load_plan()[
        "expected_totals"
    ]

    assert (
        x[
            "total_faulted_model_window_evaluations"
        ]
        == x[
            "transient_faulted_model_window_evaluations"
        ]
        + x[
            "persistent_faulted_model_window_evaluations"
        ]
    )

    assert (
        x[
            "total_model_window_evaluations_including_clean"
        ]
        == x[
            "clean_model_window_evaluations"
        ]
        + x[
            "total_faulted_model_window_evaluations"
        ]
    )


def test_each_subject_has_12_fault_shards():
    counts = {}

    for row in load_plan()[
        "shards"
    ]:
        subject = int(
            row[
                "subject"
            ]
        )

        counts[
            subject
        ] = (
            counts.get(
                subject,
                0,
            )
            + 1
        )

    assert len(
        counts
    ) == 61

    assert set(
        counts.values()
    ) == {
        12
    }


def test_variant_target_counts():
    for row in load_plan()[
        "shards"
    ]:
        if row[
            "model_variant"
        ] == "fp32":
            assert row[
                "target_count"
            ] == 10

        else:
            assert row[
                "model_variant"
            ] == "ptq_v7"

            assert row[
                "target_count"
            ] == 14


def test_every_fault_shard_references_a_clean_cache():
    x = load_plan()

    clean_ids = {
        row[
            "clean_cache_id"
        ]
        for row in x[
            "clean_caches"
        ]
    }

    for row in x[
        "shards"
    ]:
        assert row[
            "clean_cache_id"
        ] in clean_ids


def test_phase5d_hash_binding():
    x = load_plan()

    assert (
        x[
            "frozen_dependencies"
        ][
            "phase5d_protocol"
        ][
            "sha256"
        ]
        == sha(
            PHASE5D
        )
    )


def test_execution_contract_prohibits_adaptation():
    x = load_plan()[
        "execution_contract"
    ]

    for key in (
        "sampling_change_allowed",
        "target_change_allowed",
        "bit_resampling_allowed",
        "persistent_onset_resampling_allowed",
        "threshold_retuning_allowed",
    ):
        assert x[
            key
        ] is False, key


def test_documentation_exists():
    assert DOC.is_file()
