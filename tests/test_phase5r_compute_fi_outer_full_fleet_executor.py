from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from collections import Counter
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
    / "compute_fi_outer_fleet_executor_v1.py"
)

QUALIFIER = (
    PHASE5
    / "qualify_compute_fi_outer_fleet_executor_v1.py"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5r_compute_fi_outer_fleet_executor_qualification_v1.json"
)

GATE = (
    ROOT
    / "configs/evaluation/"
    "phase5p_compute_fi_outer_full_fleet_completion_gate_v1.json"
)

PLAN = (
    ROOT
    / "manifests/"
    "phase_5e_compute_fi_outer_execution_plan_v1.json"
)

PHASE5D = (
    ROOT
    / "configs/evaluation/"
    "phase5d_compute_fi_outer_protocol_v1.json"
)

RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/"
    "results/phase5r_compute_fi_outer_fleet_executor_qualification_v1/"
    "qualification.json"
)

OUTER_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/"
    "results/phase5_outer_compute_fi_prospective_v1"
)

CANARY_SHARD = (
    "p5e-o1-f5-s009-fp32-seed42-transient-6d97ac3095dc8dc9"
)

CANARY_CLEAN = (
    "p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b"
)

DOC = (
    ROOT
    / "docs/"
    "PHASE_5R_COMPUTE_FI_OUTER_FULL_FLEET_EXECUTOR_QUALIFICATION_V1.md"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def module():
    spec = importlib.util.spec_from_file_location(
        "phase5r_executor",
        EXECUTOR,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        mod
    )

    return mod


def test_qualification_result_hash_frozen_inputs():
    cfg = load(CONFIG)

    assert (
        cfg["status"]
        == "FROZEN_PRE_OUTER_FULL_FLEET_EXECUTOR_QUALIFICATION_CONTRACT"
    )

    assert (
        cfg["qualification_partition"]
        == "training_calibration"
    )

    assert (
        cfg["outer_execution_allowed_during_qualification"]
        is False
    )


def test_phase5p_gate_hash_is_hard_bound():
    mod = module()

    assert (
        mod.EXPECTED_PHASE5P_GATE_SHA256
        == sha(GATE)
    )


def test_gate_validation_exact_estate():
    mod = module()

    validated = mod.validate_gate()

    assert len(
        validated["plan"]["shards"]
    ) == 732

    assert len(
        validated["remaining_shards"]
    ) == 731

    assert len(
        validated["remaining_clean_caches"]
    ) == 365


def test_accepted_canary_is_reuse_only():
    mod = module()

    validated = mod.validate_gate()

    auth = mod.shard_authorization(
        validated,
        mod.ACCEPTED_CANARY_SHARD_ID,
    )

    assert (
        auth["mode"]
        == "accepted_canary_reuse_only"
    )


def test_unknown_shard_rejected():
    mod = module()

    validated = mod.validate_gate()

    with pytest.raises(
        ValueError,
        match="not in frozen Phase-5E plan",
    ):
        mod.shard_authorization(
            validated,
            "not-a-real-shard",
        )


def test_complete_plan_has_four_equal_execution_classes():
    plan = load(PLAN)

    counts = Counter(
        (
            row["model_variant"],
            row["persistence"],
        )
        for row in plan["shards"]
    )

    assert counts == {
        (
            "fp32",
            "transient_one_inference",
        ):
            183,

        (
            "fp32",
            "persistent_from_onset_until_trial_end",
        ):
            183,

        (
            "ptq_v7",
            "transient_one_inference",
        ):
            183,

        (
            "ptq_v7",
            "persistent_from_onset_until_trial_end",
        ):
            183,
    }


def test_target_counts_unchanged():
    phase5d = load(PHASE5D)

    assert len([
        row
        for row in phase5d["target_inventory"]
        if "fp32" in row["model_variants"]
    ]) == 10

    assert len([
        row
        for row in phase5d["target_inventory"]
        if "ptq_v7" in row["model_variants"]
    ]) == 14


def test_persistent_binding_rederives_from_frozen_plan():
    mod = module()

    validated = mod.validate_gate()

    shard = next(
        row
        for row in validated["plan"]["shards"]
        if row["model_variant"] == "ptq_v7"
        and row["persistence"]
        == "persistent_from_onset_until_trial_end"
    )

    derived = mod.validate_shard_structure(
        phase5d=validated["phase5d"],
        plan=validated["plan"],
        shard=shard,
    )

    assert (
        derived["expected_outer_instance_ids"]
        == shard["expected_outer_instance_ids"]
    )

    assert (
        derived[
            "expected_faulted_model_window_evaluations"
        ]
        == shard[
            "expected_faulted_model_window_evaluations"
        ]
    )


def test_validation_functions_cannot_cross_outer_execution_boundary():
    tree = ast.parse(
        EXECUTOR.read_text()
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
        "build_frozen_ptq_eager",
        "run_fp32_fault_only",
        "run_ptq_activation_buffer_fault_only",
        "_load_outer_trial_signal",
        "execute_shard",
        "execute_fleet",
    }

    def calls(node):
        result = set()

        for child in ast.walk(node):
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

            elif (
                isinstance(
                    func,
                    ast.Attribute,
                )
                and isinstance(
                    func.value,
                    ast.Name,
                )
            ):
                result.add(
                    f"{func.value.id}.{func.attr}"
                )

        return result

    for name in (
        "validate_gate",
        "shard_authorization",
        "validate_shard_structure",
        "persistent_target_binding",
        "build_outer_identity",
    ):
        assert not (
            calls(
                functions[name]
            )
            & prohibited
        )


def test_only_outer_signal_loader_uses_numpy_load():
    tree = ast.parse(
        EXECUTOR.read_text()
    )

    owners = []

    for node in tree.body:
        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        for child in ast.walk(node):
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
        "_load_outer_trial_signal"
    ]


def test_no_paired_runner_or_label_file_access():
    source = EXECUTOR.read_text()

    assert "run_fp32_paired" not in source
    assert "run_ptq_activation_buffer_paired" not in source
    assert ".run_paired(" not in source
    assert "labels.npy" not in source.lower()


def test_qualifier_uses_actual_phase5k_onfield_key():
    source = QUALIFIER.read_text()

    assert '''assert boundary[
        "onfield_used"
    ] is False''' in source

    assert '''assert boundary[
        "onfield_payload_used"
    ] is False''' not in source

    # Phase-5R's own output schema intentionally keeps this name.
    assert source.count(
        '"onfield_payload_used":'
    ) == 1


def test_training_calibration_qualification_result():
    result = load(RESULT)

    assert (
        result["status"]
        == "QUALIFIED_FULL_FLEET_EXECUTOR_PRE_OUTER"
    )

    assert (
        result["qualification_partition"]
        == "training_calibration"
    )

    assert result[
        "qualification_case_count"
    ] == 6

    assert result[
        "fault_only_sequence_executions"
    ] == 30

    assert result[
        "ptq_eager_torchscript_clean_equal"
    ] is True

    assert result[
        "ptq_weight_reset_after_transient"
    ] is True

    assert result[
        "ptq_weight_reset_after_persistent"
    ] is True

    assert result[
        "source_artifacts_unchanged"
    ] is True

    assert result[
        "accepted_canary_artifacts_unchanged"
    ] is True


def test_qualification_case_active_masks():
    result = load(RESULT)

    for row in result["qualification_cases"]:
        if "transient" in row["name"]:
            assert row["active_mask"] == [
                True,
                True,
                True,
                True,
                True,
            ]
        else:
            assert row["active_mask"] == [
                False,
                False,
                True,
                True,
                True,
            ]

        assert (
            row["embedded_clean_reference_forward"]
            is False
        )


def test_training_calibration_scientific_boundary():
    boundary = load(RESULT)[
        "scientific_boundary"
    ]

    assert boundary[
        "training_calibration_payload_used"
    ] is True

    for key in (
        "validation_payload_used",
        "outer_test_payload_used",
        "onfield_payload_used",
        "outer_full_fleet_execution_started",
        "additional_outer_fault_shard_executed",
        "additional_outer_clean_cache_executed",
        "prediction_outcomes_used_for_selection",
        "threshold_applied",
        "threshold_changed",
        "metric_computed",
        "sampling_changed",
        "identity_changed",
        "phase5e_plan_changed",
        "phase5p_gate_changed",
        "aggregate_CC_result_generated",
        "CSC_result_generated",
    ):
        assert boundary[key] is False


def test_outer_estate_remains_single_accepted_canary():
    dirs = {
        path.name
        for path in OUTER_ROOT.iterdir()
        if path.is_dir()
    }

    assert dirs == {
        CANARY_SHARD,
        CANARY_CLEAN,
    }


def test_documentation_exists():
    assert DOC.is_file()
