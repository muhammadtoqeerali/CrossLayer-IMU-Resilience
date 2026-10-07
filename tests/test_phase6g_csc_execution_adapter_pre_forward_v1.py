from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

import csc_execution_adapter_v1 as adapter


ROOT = Path(__file__).resolve().parents[1]


def synthetic_pair(
    *,
    target_name: str,
    persistence: str,
    inference_index: int,
    overlap_count: int,
    eligible_model_variants: list[str],
):
    return {
        "schema_version":
            "phase6e_csc_pair_metadata_v1",

        "execution_performed":
            False,

        "sensor_parent_kind":
            "source_trial",

        "sensor_fault_id":
            "synthetic-sensor-fault",

        "sensor_replay_id":
            "synthetic-sensor-replay",

        "compute_stratum_index":
            0,

        "target_name":
            target_name,

        "persistence":
            persistence,

        "eligible_model_variants":
            eligible_model_variants,

        "checkpoint_seeds":
            [
                42,
                123,
                2025,
            ],

        "pair_member_multiplier":
            (
                len(
                    eligible_model_variants
                )
                * 3
            ),

        "compute_coordinate": {
            "sampling_instance_id":
                "synthetic-sampling-id",

            "parent_kind":
                (
                    "window"
                    if persistence
                    == "transient_one_inference"
                    else "trial"
                ),

            "element_index":
                0,

            "bit_position":
                0,

            "inference_index":
                inference_index,
        },

        "temporal_accounting": {
            "sensor_exposed_window_count":
                1,

            "compute_active_window_count":
                1,

            "temporal_overlap":
                overlap_count > 0,

            "temporal_overlap_window_count":
                overlap_count,

            "sensor_compute_union_window_count":
                1,

            "overlap_window_indices":
                [],
        },
    }


def test_execution_is_hard_disabled():
    assert adapter.EXECUTION_ENABLED is False

    with pytest.raises(
        RuntimeError,
        match="PHASE6G_EXECUTION_DISABLED_PRE_FORWARD_ADAPTER",
    ):
        adapter.execute_request(
            request={}
        )


def test_adapter_has_no_execution_runtime_imports():
    path = (
        ROOT
        / "experiments/phase_06/csc_execution_adapter_v1.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    imported_modules = set()

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            imported_modules.update(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imported_modules.add(
                    node.module
                )

    assert "torch" not in imported_modules
    assert "numpy" not in imported_modules
    assert "sensor_fi_operators" not in imported_modules
    assert "sensor_fi_outer_executor_v1" not in imported_modules
    assert "compute_fi_outer_fleet_executor_v1" not in imported_modules
    assert "compute_fi_execution_harness" not in imported_modules
    assert "compute_fi_fault_only_execution_v1" not in imported_modules


def test_all_pinned_hashes_and_symbols_validate():
    validated = adapter.validate_pinned_bindings()

    assert len(
        validated[
            "modules"
        ]
    ) == 7

    assert (
        validated[
            "phase6e_validated"
        ][
            "plan"
        ][
            "pair_surface"
        ][
            "model_independent_csc_pair_count"
        ]
        == 4237835
    )


def test_input_conversion_contract_is_exact_phase5_semantics():
    assert adapter.INPUT_CONVERSION_CONTRACT == {
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
    }


def test_fp32_route():
    validated = adapter.validate_pinned_bindings()

    target = adapter._target_record(
        validated=validated,
        target_name="front_end_output_fp32",
    )

    route = adapter.compute_execution_route(
        model_variant="fp32",
        target=target,
    )

    assert (
        route[
            "primitive"
        ]
        == "run_fp32_fault_only"
    )

    assert (
        route[
            "requires_ptq_clean_state"
        ]
        is False
    )


def test_ptq_activation_route():
    validated = adapter.validate_pinned_bindings()

    target = adapter._target_record(
        validated=validated,
        target_name="conv2_quantized_output",
    )

    route = adapter.compute_execution_route(
        model_variant="ptq_v7",
        target=target,
    )

    assert (
        route[
            "primitive"
        ]
        == "run_ptq_activation_buffer_fault_only"
    )

    assert (
        route[
            "requires_ptq_clean_state"
        ]
        is False
    )


def test_ptq_weight_route_requires_and_restores_clean_state():
    validated = adapter.validate_pinned_bindings()

    target = adapter._target_record(
        validated=validated,
        target_name="conv_2.0.weight",
    )

    route = adapter.compute_execution_route(
        model_variant="ptq_v7",
        target=target,
    )

    assert (
        route[
            "primitive"
        ]
        == "PTQWeightFaultOnlySession.run_fault_only"
    )

    assert (
        route[
            "requires_ptq_clean_state"
        ]
        is True
    )

    assert (
        route[
            "ptq_clean_state_restored_after_sequence"
        ]
        is True
    )

    assert "finally" in route[
        "restoration"
    ]


def test_transient_execution_request_is_exactly_one_window_and_overlap():
    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="transient_one_inference",
        inference_index=7,
        overlap_count=1,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    request = adapter.build_execution_request(
        pair_metadata=pair,
        sensor_exposed_window_indices=[
            7,
        ],
        trial_window_count=20,
        model_variant="fp32",
        checkpoint_seed=42,
        validated_bindings=validated,
    )

    adapter.validate_execution_request(
        request
    )

    assert (
        request[
            "compute_execution_window_indices"
        ]
        == [
            7
        ]
    )

    assert (
        request[
            "sensor_active_mask_on_compute_sequence"
        ]
        == [
            True
        ]
    )

    assert (
        request[
            "simultaneous_overlap_window_indices"
        ]
        == [
            7
        ]
    )

    assert (
        request[
            "phase5_fault_sequence_contract"
        ][
            "identity_count"
        ]
        == 1
    )


def test_persistent_request_uses_exact_onset_to_end_suffix():
    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="persistent_from_onset_until_trial_end",
        inference_index=5,
        overlap_count=2,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    request = adapter.build_execution_request(
        pair_metadata=pair,
        sensor_exposed_window_indices=[
            6,
            8,
        ],
        trial_window_count=10,
        model_variant="fp32",
        checkpoint_seed=123,
        validated_bindings=validated,
    )

    adapter.validate_execution_request(
        request
    )

    assert (
        request[
            "compute_execution_window_indices"
        ]
        == [
            5,
            6,
            7,
            8,
            9,
        ]
    )

    assert (
        request[
            "sensor_active_mask_on_compute_sequence"
        ]
        == [
            False,
            True,
            False,
            True,
            False,
        ]
    )

    assert (
        request[
            "simultaneous_overlap_window_indices"
        ]
        == [
            6,
            8,
        ]
    )


def test_zero_overlap_persistent_request_is_retained():
    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="persistent_from_onset_until_trial_end",
        inference_index=5,
        overlap_count=0,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    request = adapter.build_execution_request(
        pair_metadata=pair,
        sensor_exposed_window_indices=[
            0,
            1,
        ],
        trial_window_count=10,
        model_variant="ptq_v7",
        checkpoint_seed=2025,
        validated_bindings=validated,
    )

    adapter.validate_execution_request(
        request
    )

    assert (
        request[
            "simultaneous_overlap_window_indices"
        ]
        == []
    )

    assert not any(
        request[
            "sensor_active_mask_on_compute_sequence"
        ]
    )

    assert (
        request[
            "compute_execution_window_indices"
        ]
        == [
            5,
            6,
            7,
            8,
            9,
        ]
    )


def test_request_id_is_deterministic_and_tamper_evident():
    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="transient_one_inference",
        inference_index=3,
        overlap_count=1,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    kwargs = {
        "pair_metadata":
            pair,

        "sensor_exposed_window_indices":
            [
                3
            ],

        "trial_window_count":
            12,

        "model_variant":
            "fp32",

        "checkpoint_seed":
            42,

        "validated_bindings":
            validated,
    }

    first = adapter.build_execution_request(
        **kwargs
    )

    second = adapter.build_execution_request(
        **kwargs
    )

    assert (
        first[
            "execution_request_id"
        ]
        == second[
            "execution_request_id"
        ]
    )

    tampered = dict(
        first
    )

    tampered[
        "checkpoint_seed"
    ] = 123

    with pytest.raises(
        ValueError,
        match="execution request ID mismatch",
    ):
        adapter.validate_execution_request(
            tampered
        )


def test_clean_reference_is_explicitly_outside_fault_loop():
    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="transient_one_inference",
        inference_index=1,
        overlap_count=1,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    request = adapter.build_execution_request(
        pair_metadata=pair,
        sensor_exposed_window_indices=[
            1
        ],
        trial_window_count=4,
        model_variant="fp32",
        checkpoint_seed=42,
        validated_bindings=validated,
    )

    assert (
        request[
            "clean_reference_contract"
        ][
            "inside_fault_loop"
        ]
        is False
    )

    assert (
        request[
            "clean_reference_contract"
        ][
            "reuse_or_generate_before_fault_execution"
        ]
        is True
    )


def test_building_requests_does_not_change_frozen_executor():
    path = (
        ROOT
        / "experiments/phase_06/csc_outer_executor_v1.py"
    )

    before = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert (
        before
        == adapter.FROZEN_METADATA_EXECUTOR_SHA256
    )

    validated = adapter.validate_pinned_bindings()

    pair = synthetic_pair(
        target_name="front_end_output_fp32",
        persistence="transient_one_inference",
        inference_index=2,
        overlap_count=1,
        eligible_model_variants=[
            "fp32",
            "ptq_v7",
        ],
    )

    adapter.build_execution_request(
        pair_metadata=pair,
        sensor_exposed_window_indices=[
            2
        ],
        trial_window_count=5,
        model_variant="fp32",
        checkpoint_seed=42,
        validated_bindings=validated,
    )

    after = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert after == before


def test_source_contains_no_execution_call():
    source = inspect.getsource(
        adapter
    )

    assert "torch.no_grad" not in source
    assert "model(" not in source
    assert ".apply_fault(" not in source
    assert ".execute_fault_sequence(" not in source
    assert ".run_fault_only(" not in source
