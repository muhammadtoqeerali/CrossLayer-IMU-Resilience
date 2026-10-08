from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import csc_execution_runtime_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]

RUNTIME_PATH = (
    ROOT
    / "experiments/phase_06/csc_execution_runtime_v1.py"
)


def synthetic_request(
    *,
    persistence: str = "persistent_from_onset_until_trial_end",
    compute_indices=(2, 3, 4),
    exposed=(1, 3),
):
    compute_indices = list(
        compute_indices
    )

    exposed = list(
        exposed
    )

    onset = compute_indices[
        0
    ]

    sensor_mask = [
        index in set(
            exposed
        )
        for index in compute_indices
    ]

    overlap = [
        index
        for index, active
        in zip(
            compute_indices,
            sensor_mask,
        )
        if active
    ]

    payload = {
        "schema_version":
            "phase6g_csc_pre_forward_execution_request_v1",

        "execution_enabled":
            False,

        "sensor_fault_id":
            "sensor-fault",

        "sensor_replay_id":
            "sensor-replay",

        "sensor_parent_kind":
            "source_trial",

        "target_name":
            "front_end_output_fp32",

        "representation_class":
            "fp32_activation",

        "target_role":
            "activation",

        "persistence":
            persistence,

        "model_variant":
            "fp32",

        "checkpoint_seed":
            42,

        "compute_sampling_instance_id":
            "compute-sampling",

        "element_index":
            3,

        "bit_position":
            17,

        "onset_or_inference_index":
            onset,

        "trial_window_count":
            5,

        "sensor_exposed_window_indices":
            exposed,

        "compute_execution_window_indices":
            compute_indices,

        "sensor_active_mask_on_compute_sequence":
            sensor_mask,

        "simultaneous_overlap_window_indices":
            overlap,

        "phase4h_sensor_path": {
            "executor_entrypoint":
                "condition_windows",

            "operator_entrypoint":
                "apply_fault",

            "sensor_corruption_precedes_window_tensor_conversion":
                True,
        },

        "phase5_input_conversion": {
                                       "source_window_dtype":
                                           "numpy.float32",
                                       "conversion":
                                           "torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0)",
                                       "batch_dimension":
                                           0,
                                       "transpose":
                                           False,
                                       "permute":
                                           False,
                                       "copy_required_by_adapter":
                                           False,
                                   },

        "phase5_model_bundle": {
            "loader":
                "load_model_bundle",
        },

        "phase5_compute_route": {
            "primitive":
                "run_fp32_fault_only",

            "requires_ptq_clean_state":
                False,

            "ptq_clean_state_restored_after_sequence":
                False,
        },

        "phase5_fault_sequence_contract": {
            "function":
                "execute_fault_sequence",

            "identity_count":
                1,

            "transient_one_identity_per_input":
                persistence
                == "transient_one_inference",

            "persistent_exactly_one_identity":
                persistence
                == "persistent_from_onset_until_trial_end",

            "persistent_all_post_onset_mutations_must_be_active":
                persistence
                == "persistent_from_onset_until_trial_end",
        },

        "clean_reference_contract": {
            "inside_fault_loop":
                False,

            "reuse_or_generate_before_fault_execution":
                True,
        },
    }

    payload[
        "execution_request_id"
    ] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return payload


def reference_summary(
    index,
    _window,
):
    return {
        "input_sha256":
            f"ref-input-{index}",

        "output_sha256":
            f"ref-output-{index}",

        "output_nonfinite":
            False,

        "output_values":
            [
                float(
                    index
                )
            ],

        "output_float32_hex":
            [
                f"{index:08x}"
            ],

        "softmax_values":
            [
                1.0
            ],
    }


def fault_sequence_summary(
    indices,
    _windows,
    _request,
):
    return [
        {
            "input_sha256":
                f"fault-input-{index}",

            "faulted_output_sha256":
                f"fault-output-{index}",

            "faulted_output_nonfinite":
                False,

            "faulted_output_values":
                [
                    float(
                        index
                    )
                ],

            "faulted_output_float32_hex":
                [
                    f"{index:08x}"
                ],

            "faulted_softmax_values":
                [
                    1.0
                ],

            "mutation": {
                "active":
                    True,
            },
        }
        for index in indices
    ]


def test_pinned_runtime_bindings_validate_without_heavy_imports():
    observed = (
        runtime.validate_pinned_runtime_bindings()
    )

    assert (
        observed[
            "status"
        ]
        == "PHASE6J_RUNTIME_BINDINGS_VALID"
    )

    assert len(
        observed[
            "hashes"
        ]
    ) == 13


def test_public_execution_is_hard_blocked_before_runtime_import(
    tmp_path,
    monkeypatch,
):
    called = {
        "heavy_import":
            False,
    }

    def forbidden():
        called[
            "heavy_import"
        ] = True

        raise AssertionError(
            "heavy runtime import must not occur"
        )

    monkeypatch.setattr(
        runtime,
        "load_bound_runtime_modules",
        forbidden,
    )

    with pytest.raises(
        RuntimeError,
        match="PHASE6J_OUTER_EXECUTION_NOT_AUTHORIZED",
    ):
        runtime.execute_shard(
            gate_path=tmp_path
            / "missing-gate.json",
            shard_id_value="not-authorized",
            dataset_root=tmp_path
            / "dataset",
            phase5_output_root=tmp_path
            / "phase5",
            output_root=tmp_path
            / "phase6",
        )

    assert called[
        "heavy_import"
    ] is False


def test_even_plausible_gate_cannot_bypass_execution_constant(
    tmp_path,
    monkeypatch,
):
    runtime_sha = runtime.sha256_file(
        RUNTIME_PATH
    )

    shard = "phase6h_csc_f1_s1_drift_deadbeefdeadbeef"

    gate = tmp_path / "gate.json"

    runtime.write_json(
        gate,
        {
            "execution_authorized":
                True,

            "runtime_sha256":
                runtime_sha,

            "authorized_shard_ids":
                [
                    shard
                ],
        },
    )

    called = {
        "heavy_import":
            False,
    }

    def forbidden():
        called[
            "heavy_import"
        ] = True

        raise AssertionError(
            "heavy import still prohibited"
        )

    monkeypatch.setattr(
        runtime,
        "load_bound_runtime_modules",
        forbidden,
    )

    with pytest.raises(
        RuntimeError,
        match="PHASE6J_OUTER_EXECUTION_NOT_AUTHORIZED",
    ):
        runtime.execute_shard(
            gate_path=gate,
            shard_id_value=shard,
            dataset_root=tmp_path,
            phase5_output_root=tmp_path,
            output_root=tmp_path,
        )

    assert called[
        "heavy_import"
    ] is False


def test_shard_identity_is_deterministic_and_family_bound():
    a = runtime.shard_id(
        fold=3,
        subject=12,
        sensor_family="dropout",
    )

    b = runtime.shard_id(
        fold=3,
        subject=12,
        sensor_family="dropout",
    )

    c = runtime.shard_id(
        fold=3,
        subject=12,
        sensor_family="drift",
    )

    assert a == b
    assert a != c
    assert a.startswith(
        "phase6h_csc_f3_s12_dropout_"
    )


def test_sensor_reference_cache_identity_is_model_member_bound():
    common = dict(
        fold=3,
        subject=12,
        task=1,
        trial=7,
        sensor_parent_kind="source_trial",
        parent_local_index=-1,
        sensor_fault_id="fault",
        sensor_replay_id="replay",
        model_variant="fp32",
        checkpoint_seed=42,
    )

    a = runtime.sensor_reference_cache_id(
        **common
    )

    b = runtime.sensor_reference_cache_id(
        **{
            **common,
            "checkpoint_seed":
                123,
        }
    )

    assert a != b


def test_canonical_pair_member_order_is_variant_seed_then_pair():
    rows = [
        {
            "model_variant":
                "ptq_v7",

            "checkpoint_seed":
                42,

            "sensor_parent_kind":
                "stored_window",

            "parent_local_index":
                3,

            "task":
                1,

            "trial":
                1,

            "severity":
                "L1",

            "sensor_replay_id":
                "b",
        },
        {
            "model_variant":
                "fp32",

            "checkpoint_seed":
                2025,

            "sensor_parent_kind":
                "stored_window",

            "parent_local_index":
                3,

            "task":
                1,

            "trial":
                1,

            "severity":
                "L1",

            "sensor_replay_id":
                "b",
        },
        {
            "model_variant":
                "fp32",

            "checkpoint_seed":
                42,

            "sensor_parent_kind":
                "stored_window",

            "parent_local_index":
                3,

            "task":
                1,

            "trial":
                1,

            "severity":
                "L1",

            "sensor_replay_id":
                "a",
        },
    ]

    ordered = sorted(
        rows,
        key=runtime.canonical_pair_member_sort_key,
    )

    assert [
        (
            row[
                "model_variant"
            ],
            row[
                "checkpoint_seed"
            ],
            row[
                "sensor_replay_id"
            ],
        )
        for row in ordered
    ] == [
        (
            "fp32",
            42,
            "a",
        ),
        (
            "fp32",
            2025,
            "b",
        ),
        (
            "ptq_v7",
            42,
            "b",
        ),
    ]


def test_synthetic_orchestration_generates_references_before_faults():
    events = []

    def ref(
        index,
        window,
    ):
        events.append(
            (
                "reference",
                index,
                window,
            )
        )

        return reference_summary(
            index,
            window,
        )

    def fault(
        indices,
        windows,
        request,
    ):
        events.append(
            (
                "fault_sequence",
                list(
                    indices
                ),
                list(
                    windows
                ),
            )
        )

        return fault_sequence_summary(
            indices,
            windows,
            request,
        )

    request = synthetic_request()

    observed = runtime.synthetic_execute_pair_member(
        request=request,
        shard_id_value="shard",
        fold=1,
        subject=9,
        task=1,
        trial=2,
        parent_local_index=-1,
        phase5_clean_cache_id="clean-cache",
        sensor_reference_cache_id_value="sensor-cache",
        windows_by_index={
            1:
                "w1",
            2:
                "w2",
            3:
                "w3",
            4:
                "w4",
        },
        hooks=runtime.SyntheticHooks(
            reference_forward=ref,
            fault_sequence=fault,
        ),
    )

    assert events[
        :2
    ] == [
        (
            "reference",
            1,
            "w1",
        ),
        (
            "reference",
            3,
            "w3",
        ),
    ]

    assert events[
        2
    ][
        0
    ] == "fault_sequence"

    assert len(
        observed[
            "sensor_reference_rows"
        ]
    ) == 2

    assert len(
        observed[
            "csc_fault_rows"
        ]
    ) == 3

    assert observed[
        "pair_member"
    ][
        "sensor_reference_record_count"
    ] == 2

    assert observed[
        "pair_member"
    ][
        "csc_fault_record_count"
    ] == 3


def test_reference_kind_switches_per_compute_window():
    observed = runtime.synthetic_execute_pair_member(
        request=synthetic_request(),
        shard_id_value="shard",
        fold=1,
        subject=9,
        task=1,
        trial=2,
        parent_local_index=-1,
        phase5_clean_cache_id="clean-cache",
        sensor_reference_cache_id_value="sensor-cache",
        windows_by_index={
            1:
                "w1",
            2:
                "w2",
            3:
                "w3",
            4:
                "w4",
        },
        hooks=runtime.SyntheticHooks(
            reference_forward=reference_summary,
            fault_sequence=fault_sequence_summary,
        ),
    )

    rows = observed[
        "csc_fault_rows"
    ]

    assert [
        row[
            "execution_window_index"
        ]
        for row in rows
    ] == [
        2,
        3,
        4,
    ]

    assert [
        row[
            "reference_kind"
        ]
        for row in rows
    ] == [
        "phase5_clean_cache",
        "phase6_sensor_reference",
        "phase5_clean_cache",
    ]


def test_zero_overlap_persistent_pair_is_retained_and_uses_clean_reference():
    request = synthetic_request(
        exposed=(
            0,
            1,
        ),
        compute_indices=(
            2,
            3,
            4,
        ),
    )

    observed = runtime.synthetic_execute_pair_member(
        request=request,
        shard_id_value="shard",
        fold=1,
        subject=9,
        task=1,
        trial=2,
        parent_local_index=-1,
        phase5_clean_cache_id="clean-cache",
        sensor_reference_cache_id_value="sensor-cache",
        windows_by_index={
            0:
                "w0",
            1:
                "w1",
            2:
                "w2",
            3:
                "w3",
            4:
                "w4",
        },
        hooks=runtime.SyntheticHooks(
            reference_forward=reference_summary,
            fault_sequence=fault_sequence_summary,
        ),
    )

    assert observed[
        "pair_member"
    ][
        "zero_temporal_overlap"
    ] is True

    assert observed[
        "pair_member"
    ][
        "simultaneous_overlap_window_indices"
    ] == []

    assert all(
        row[
            "reference_kind"
        ]
        == "phase5_clean_cache"
        for row in observed[
            "csc_fault_rows"
        ]
    )

    assert len(
        observed[
            "sensor_reference_rows"
        ]
    ) == 2


def test_transient_request_has_exact_one_fault_row():
    request = synthetic_request(
        persistence="transient_one_inference",
        compute_indices=(
            3,
        ),
        exposed=(
            3,
        ),
    )

    observed = runtime.synthetic_execute_pair_member(
        request=request,
        shard_id_value="shard",
        fold=1,
        subject=9,
        task=1,
        trial=2,
        parent_local_index=-1,
        phase5_clean_cache_id="clean-cache",
        sensor_reference_cache_id_value="sensor-cache",
        windows_by_index={
            3:
                "w3",
        },
        hooks=runtime.SyntheticHooks(
            reference_forward=reference_summary,
            fault_sequence=fault_sequence_summary,
        ),
    )

    assert len(
        observed[
            "csc_fault_rows"
        ]
    ) == 1

    assert observed[
        "csc_fault_rows"
    ][0][
        "reference_kind"
    ] == "phase6_sensor_reference"


def test_phase6h_atomic_commit_writes_marker_before_os_replace(
    tmp_path,
    monkeypatch,
):
    output_root = (
        tmp_path
        / "out"
    )

    shard = runtime.shard_id(
        fold=1,
        subject=1,
        sensor_family="drift",
    )

    action, partial, success = (
        runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=shard,
            gate_sha256="gate",
            runtime_sha256="runtime",
            recompute_partial=False,
        )
    )

    assert action == "compute"
    assert success is None
    assert partial is not None

    for name in runtime.OUTPUT_FILES:
        (
            partial
            / name
        ).write_text(
            name
            + "\n",
            encoding="utf-8",
        )

    hashes = {
        name:
            runtime.sha256_file(
                partial
                / name
            )
        for name in runtime.OUTPUT_FILES
    }

    final = (
        output_root
        / "shards"
        / shard
    )

    real_replace = runtime.os.replace
    observed = {
        "marker_before_replace":
            False,
    }

    def checked_replace(
        source,
        target,
    ):
        source = Path(
            source
        )

        marker = (
            source
            / runtime.SUCCESS_MARKER
        )

        assert marker.is_file()

        observed[
            "marker_before_replace"
        ] = True

        return real_replace(
            source,
            target,
        )

    monkeypatch.setattr(
        runtime.os,
        "replace",
        checked_replace,
    )

    runtime.commit_phase6h_artifact(
        partial_dir=partial,
        final_dir=final,
        shard_id_value=shard,
        gate_sha256="gate",
        runtime_sha256="runtime",
        output_hashes=hashes,
        coverage={
            "pair_member_records":
                1,
        },
    )

    assert observed[
        "marker_before_replace"
    ] is True

    assert (
        final
        / runtime.SUCCESS_MARKER
    ).is_file()


def test_valid_final_artifact_is_reusable(
    tmp_path,
):
    output_root = tmp_path / "out"

    shard = runtime.shard_id(
        fold=1,
        subject=1,
        sensor_family="drift",
    )

    action, partial, _ = (
        runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=shard,
            gate_sha256="gate",
            runtime_sha256="runtime",
            recompute_partial=False,
        )
    )

    assert action == "compute"
    assert partial is not None

    for name in runtime.OUTPUT_FILES:
        (
            partial
            / name
        ).write_text(
            name
            + "\n",
            encoding="utf-8",
        )

    hashes = {
        name:
            runtime.sha256_file(
                partial
                / name
            )
        for name in runtime.OUTPUT_FILES
    }

    final = (
        output_root
        / "shards"
        / shard
    )

    runtime.commit_phase6h_artifact(
        partial_dir=partial,
        final_dir=final,
        shard_id_value=shard,
        gate_sha256="gate",
        runtime_sha256="runtime",
        output_hashes=hashes,
        coverage={},
    )

    action, partial2, success = (
        runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=shard,
            gate_sha256="gate",
            runtime_sha256="runtime",
            recompute_partial=False,
        )
    )

    assert action == "reuse"
    assert partial2 is None
    assert success[
        "status"
    ] == "PASS"


def test_invalid_partial_requires_explicit_recompute(
    tmp_path,
):
    output_root = (
        tmp_path
        / "out"
    )

    shard = runtime.shard_id(
        fold=1,
        subject=1,
        sensor_family="drift",
    )

    partial = (
        output_root
        / "shards"
        / (
            shard
            + ".partial"
        )
    )

    partial.mkdir(
        parents=True
    )

    with pytest.raises(
        ValueError,
        match="recompute_partial required",
    ):
        runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=shard,
            gate_sha256="gate",
            runtime_sha256="runtime",
            recompute_partial=False,
        )

    action, new_partial, _ = (
        runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=shard,
            gate_sha256="gate",
            runtime_sha256="runtime",
            recompute_partial=True,
        )
    )

    assert action == "compute"
    assert new_partial == partial


def test_runtime_never_calls_phase5_commit_atomic_artifact():
    source = RUNTIME_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    called_attributes = set()

    called_names = set()

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            called_names.add(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            called_attributes.add(
                node.func.attr
            )

    assert (
        "commit_atomic_artifact"
        not in called_names
    )

    assert (
        "commit_atomic_artifact"
        not in called_attributes
    )


def test_module_import_surface_has_no_torch_or_numpy():
    tree = ast.parse(
        RUNTIME_PATH.read_text(
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


def test_frozen_phase6i_helper_remains_unmodified():
    path = (
        ROOT
        / "experiments/phase_06/csc_source_trial_exposure_v1.py"
    )

    assert runtime.sha256_file(
        path
    ) == (
        "f4db0c1dfecd5992deed7947665c5a8a"
        "fc8d60db28f97457ac7ad11bc24b0aeb"
    )


# PHASE6J_PREAUTH_ORCHESTRATION_EXTENSION_V1

def test_phase5_clean_cache_resolution_validates_exact_scope(tmp_path):
    calls = []

    class FakeOuterCore:
        @staticmethod
        def validate_success_marker(**kwargs):
            calls.append(
                kwargs
            )
            return {
                "status":
                    "PASS",
            }

    plan = {
        "clean_caches": [
            {
                "clean_cache_id":
                    "clean-exact",

                "fold":
                    2,

                "subject":
                    17,

                "model_variant":
                    "fp32",

                "checkpoint_seed":
                    123,
            },
        ],
    }

    observed = runtime.resolve_phase5_clean_cache(
        phase5_plan=plan,
        fold=2,
        subject=17,
        model_variant="fp32",
        checkpoint_seed=123,
        phase5_output_root=tmp_path,
        expected_plan_sha256="plan-sha",
        expected_executor_sha256="executor-sha",
        phase5_outer_core=FakeOuterCore,
    )

    assert observed[
        "clean_cache_id"
    ] == "clean-exact"

    assert calls == [
        {
            "final_dir":
                tmp_path
                / "clean-exact",

            "expected_artifact_id":
                "clean-exact",

            "expected_artifact_kind":
                "clean_cache",

            "expected_plan_sha256":
                "plan-sha",

            "expected_executor_sha256":
                "executor-sha",
        },
    ]


def test_phase5_clean_cache_missing_or_invalid_aborts_without_recompute(
    tmp_path,
):
    class MissingOuterCore:
        @staticmethod
        def validate_success_marker(**_kwargs):
            return None

    plan = {
        "clean_caches": [
            {
                "clean_cache_id":
                    "clean-exact",

                "fold":
                    2,

                "subject":
                    17,

                "model_variant":
                    "fp32",

                "checkpoint_seed":
                    123,
            },
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="ABORT_SHARD_NO_PHASE6_RECOMPUTE",
    ):
        runtime.resolve_phase5_clean_cache(
            phase5_plan=plan,
            fold=2,
            subject=17,
            model_variant="fp32",
            checkpoint_seed=123,
            phase5_output_root=tmp_path,
            expected_plan_sha256="plan-sha",
            expected_executor_sha256="executor-sha",
            phase5_outer_core=MissingOuterCore,
        )

    with pytest.raises(
        RuntimeError,
        match="ABORT_SHARD_NO_PHASE6_RECOMPUTE",
    ):
        runtime.resolve_phase5_clean_cache(
            phase5_plan=plan,
            fold=2,
            subject=99,
            model_variant="fp32",
            checkpoint_seed=123,
            phase5_output_root=tmp_path,
            expected_plan_sha256="plan-sha",
            expected_executor_sha256="executor-sha",
            phase5_outer_core=MissingOuterCore,
        )


def test_phase6i_exposure_binding_handles_stored_and_source_parents():
    calls = []

    class FakeExposure:
        @staticmethod
        def validate_pinned_exposure_contract():
            calls.append(
                "validate"
            )
            return {
                "status":
                    "ok",
            }

        @staticmethod
        def source_trial_exposed_window_indices(
            *,
            instance,
            historical_window_ends,
            source_length,
        ):
            calls.append(
                (
                    "source",
                    instance[
                        "fault_id"
                    ],
                    list(
                        historical_window_ends
                    ),
                    source_length,
                )
            )
            return [
                1,
                2,
            ]

    stored = runtime.derive_sensor_exposed_window_indices(
        parent_kind="stored_window",
        instance={
            "fault_id":
                "stored-fault",
        },
        parent_local_index=3,
        trial_window_count=5,
        historical_window_ends=None,
        source_length=None,
        exposure_helper=FakeExposure,
    )

    source = runtime.derive_sensor_exposed_window_indices(
        parent_kind="source_trial",
        instance={
            "fault_id":
                "source-fault",
        },
        parent_local_index=-1,
        trial_window_count=4,
        historical_window_ends=[
            30,
            45,
            60,
            75,
        ],
        source_length=75,
        exposure_helper=FakeExposure,
    )

    assert stored == [
        3,
    ]

    assert source == [
        1,
        2,
    ]

    assert calls == [
        "validate",
        "validate",
        (
            "source",
            "source-fault",
            [
                30,
                45,
                60,
                75,
            ],
            75,
        ),
    ]


def test_sensor_conditioning_preserves_stored_vs_source_ordering():
    events = []

    class FakeExposure:
        @staticmethod
        def validate_pinned_exposure_contract():
            events.append(
                "exposure-validate"
            )

        @staticmethod
        def source_trial_exposed_window_indices(**_kwargs):
            events.append(
                "exposure-source"
            )
            return [
                1,
            ]

    class FakeOperators:
        @staticmethod
        def apply_fault(
            data,
            instance,
            *,
            reference_scales=None,
        ):
            events.append(
                (
                    "fault",
                    data,
                    reference_scales,
                )
            )
            return (
                f"faulted:{data}",
                {
                    "family":
                        instance[
                            "family"
                        ],
                },
            )

    class FakeRunner:
        @staticmethod
        def rewindow_sequence(
            sequence,
            labels,
            *,
            fall_start_frame=None,
        ):
            events.append(
                (
                    "rewindow",
                    sequence,
                    labels,
                    fall_start_frame,
                )
            )
            return [
                "rw0",
                "rw1",
                "rw2",
            ]

    stored = runtime.condition_sensor_parent(
        parent_kind="stored_window",
        instance={
            "family":
                "bias",
        },
        parent_local_index=1,
        trial_window_count=3,
        stored_windows=[
            "w0",
            "w1",
            "w2",
        ],
        source_trial=None,
        stored_labels=None,
        fall_start_frame=None,
        historical_window_ends=None,
        source_length=None,
        reference_scales={
            "scale":
                1,
        },
        operators=FakeOperators,
        runner=FakeRunner,
        exposure_helper=FakeExposure,
    )

    assert stored[
        "conditioned_windows_by_index"
    ] == {
        1:
            "faulted:w1",
    }

    source = runtime.condition_sensor_parent(
        parent_kind="source_trial",
        instance={
            "family":
                "drift",
            "fault_id":
                "src",
        },
        parent_local_index=-1,
        trial_window_count=3,
        stored_windows=None,
        source_trial="source",
        stored_labels="labels",
        fall_start_frame=50,
        historical_window_ends=[
            30,
            45,
            80,
        ],
        source_length=80,
        reference_scales=None,
        operators=FakeOperators,
        runner=FakeRunner,
        exposure_helper=FakeExposure,
    )

    assert source[
        "sensor_exposed_window_indices"
    ] == [
        1,
    ]

    assert source[
        "conditioned_windows_by_index"
    ] == {
        1:
            "rw1",
    }

    assert (
        events.index(
            (
                "fault",
                "source",
                None,
            )
        )
        < events.index(
            (
                "rewindow",
                "faulted:source",
                "labels",
                50,
            )
        )
    )


def test_compute_fault_sequence_uses_phase5_executor_and_active_mask():
    request = synthetic_request()

    events = []

    class Mutation:
        def __init__(self, active):
            self.active = active

    class Execution:
        def __init__(self, active):
            self.mutation = Mutation(
                active
            )

    class FakePhase5Executor:
        @staticmethod
        def _window_tensor(
            windows,
            index,
        ):
            events.append(
                (
                    "tensor",
                    index,
                )
            )
            return windows[
                index
            ]

        @staticmethod
        def execute_fault_sequence(**kwargs):
            events.append(
                (
                    "execute",
                    list(
                        kwargs[
                            "inference_indices"
                        ]
                    ),
                    list(
                        kwargs[
                            "inputs"
                        ]
                    ),
                    kwargs[
                        "ptq_clean_state"
                    ],
                )
            )
            return [
                Execution(
                    True
                )
                for _ in kwargs[
                    "inference_indices"
                ]
            ]

        @staticmethod
        def execution_active_mask(executions):
            events.append(
                "active-mask"
            )
            return [
                execution.mutation.active
                for execution in executions
            ]

    observed = runtime.execute_bound_compute_fault_sequence(
        request=request,
        trial_windows=[
            "w0",
            "w1",
            "w2",
            "w3",
            "w4",
        ],
        identities=[
            "identity",
        ],
        model_bundle={
            "model":
                "model",

            "ptq_clean_state":
                None,
        },
        phase5_executor=FakePhase5Executor,
    )

    assert observed[
        "execution_window_indices"
    ] == [
        2,
        3,
        4,
    ]

    assert observed[
        "execution_active_mask"
    ] == [
        True,
        True,
        True,
    ]

    assert events[-2:] == [
        (
            "execute",
            [
                2,
                3,
                4,
            ],
            [
                "w2",
                "w3",
                "w4",
            ],
            None,
        ),
        "active-mask",
    ]


def test_model_member_stream_loads_once_and_releases_in_finally():
    events = []

    request_a = synthetic_request()
    request_b = dict(
        request_a
    )

    jobs = [
        {
            "request":
                request_a,
            "name":
                "a",
        },
        {
            "request":
                request_b,
            "name":
                "b",
        },
    ]

    def release():
        events.append(
            "release"
        )

    def load(identity):
        events.append(
            (
                "load",
                dict(
                    identity
                ),
            )
        )
        return {
            "model":
                "model",
            "ptq_clean_state":
                None,
            "release":
                release,
        }

    def execute(bundle, job):
        events.append(
            (
                "pair",
                bundle[
                    "model"
                ],
                job[
                    "name"
                ],
            )
        )
        return job[
            "name"
        ]

    observed = runtime.run_model_member_stream(
        fold=1,
        subject=9,
        model_variant="fp32",
        checkpoint_seed=42,
        pair_jobs=jobs,
        load_model_bundle_hook=load,
        pair_executor_hook=execute,
    )

    assert observed == [
        "a",
        "b",
    ]

    assert [
        event
        for event in events
        if isinstance(
            event,
            tuple,
        )
        and event[
            0
        ]
        == "load"
    ] == [
        (
            "load",
            {
                "fold":
                    1,
                "subject":
                    9,
                "model_variant":
                    "fp32",
                "checkpoint_seed":
                    42,
            },
        ),
    ]

    assert events[-1] == "release"
