import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase4h_sensor_fi_outer_executor_v1.json"
)

EXECUTOR = (
    ROOT
    / "experiments/phase_04/"
    "sensor_fi_outer_executor_v1.py"
)

FREEZE = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_executor_v1_freeze.json"
)

V2_PLAN = (
    ROOT
    / "manifests/"
    "phase_4h_sensor_fi_outer_execution_shards_v2_historical_truth.json"
)


def load_executor():
    spec = importlib.util.spec_from_file_location(
        "outer_executor_test",
        EXECUTOR,
    )

    m = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        m
    )

    return m


def test_status_and_partition():
    d = json.loads(
        CONFIG.read_text()
    )

    assert d[
        "status"
    ] == (
        "FROZEN_PRE_OUTER_EXECUTION_EXECUTOR"
    )

    assert d[
        "partition"
    ] == "outer_test"


def test_dry_run_prohibitions():
    d = json.loads(
        CONFIG.read_text()
    )[
        "dry_run_contract"
    ]

    assert d[
        "may_load_model_weights"
    ] is False

    assert d[
        "may_call_model_forward"
    ] is False

    assert d[
        "may_call_fault_operator"
    ] is False

    assert d[
        "may_compute_outer_performance"
    ] is False


def test_raw_input_not_externally_normalized():
    d = json.loads(
        CONFIG.read_text()
    )[
        "model_input_contract"
    ]

    assert d[
        "executor_input_shape"
    ] == "N x 30 x 9"

    assert d[
        "external_normalization_before_model"
    ] is False


def test_expected_global_matrix():
    d = json.loads(
        CONFIG.read_text()
    )[
        "expected_global_matrix"
    ]

    assert d[
        "shards"
    ] == 793

    assert d[
        "subjects"
    ] == 61

    assert d[
        "trials"
    ] == 6309

    assert d[
        "historical_activity_trials"
    ] == 3390

    assert d[
        "historical_falling_trials"
    ] == 2919

    assert d[
        "unique_fault_instances"
    ] == 42766632

    assert d[
        "model_window_evaluations"
    ] == 479750160

    assert d[
        "subject_condition_rows"
    ] == 320616


def test_trigger_episode_semantics():
    m = load_executor()

    p = np.asarray([
        0.1,
        0.9,
        0.95,
        0.96,
        0.2,
        0.91,
        0.92,
    ])

    assert m.trigger_episodes(
        p,
        0.8,
        2,
    ) == [
        1,
        5,
    ]


def test_first_valid_trigger_requires_full_falling_run():
    m = load_executor()

    p = np.asarray([
        0.1,
        0.95,
        0.96,
        0.2,
        0.97,
        0.98,
    ])

    labels = np.asarray([
        0,
        0,
        1,
        1,
        1,
        1,
    ])

    assert m.first_valid_trigger(
        p,
        0.9,
        2,
        labels,
    ) == 4


def test_exception_has_no_valid_falling_trigger():
    m = load_executor()

    probabilities = np.ones(
        20,
        dtype=float,
    )

    labels = np.zeros(
        20,
        dtype=int,
    )

    assert m.first_valid_trigger(
        probabilities,
        0.1,
        1,
        labels,
    ) == -1


def test_metric_confusion_and_lead_synthetic():
    m = load_executor()

    trial_records = [
        {
            "probabilities":
                np.asarray([
                    0.1,
                    0.2,
                ]),

            "labels":
                np.asarray([
                    0,
                    0,
                ]),

            "true_fall":
                False,

            "window_ends":
                np.asarray([
                    30,
                    45,
                ]),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                -1,

            "impact_position":
                -1,
        },
        {
            "probabilities":
                np.asarray([
                    0.1,
                    0.9,
                    0.95,
                ]),

            "labels":
                np.asarray([
                    0,
                    1,
                    1,
                ]),

            "true_fall":
                True,

            "window_ends":
                np.asarray([
                    30,
                    130,
                    145,
                ]),

            "sampling_rate_hz":
                100.0,

            "fall_start_position":
                100,

            "impact_position":
                180,
        },
    ]

    result = m.compute_condition_metrics(
        trial_records,
        threshold=0.8,
        required_consecutive=1,
        quantile_method="linear",
    )

    assert result[
        "TP"
    ] == 1

    assert result[
        "TN"
    ] == 1

    assert result[
        "FP"
    ] == 0

    assert result[
        "FN"
    ] == 0

    assert result[
        "eligible_event_count"
    ] == 1

    assert result[
        "detected_event_count"
    ] == 1

    assert result[
        "median_sensor_lead_ms"
    ] == 500.0


def test_dry_run_function_contains_no_execution_calls():
    m = load_executor()

    source = inspect.getsource(
        m.dry_run
    )

    prohibited = [
        "load_fp32_model(",
        "load_ptq_model(",
        "logits_and_positive_probability(",
        "apply_fault(",
        "execute_shard(",
    ]

    for token in prohibited:
        assert token not in source


def test_execute_path_exists_but_is_separate():
    m = load_executor()

    source = inspect.getsource(
        m.execute_shard
    )

    assert "load_models_for_shard(" in source

    assert "condition_windows(" in source


def test_no_external_model_normalizer_call():
    source = EXECUTOR.read_text()

    assert "IMUNormalizer(" not in source

    assert "normalizer(" not in source


def test_output_files_and_success_last_contract():
    d = json.loads(
        CONFIG.read_text()
    )

    r = d[
        "resume_contract"
    ]

    assert r[
        "success_written_last"
    ] is True

    assert r[
        "final_shard_immutable"
    ] is True

    assert r[
        "partial_shard_never_aggregated"
    ] is True


def test_v2_plan_bound():
    config = json.loads(
        CONFIG.read_text()
    )

    plan = json.loads(
        V2_PLAN.read_text()
    )

    assert plan[
        "shard_count"
    ] == 793

    assert config[
        "frozen_dependencies"
    ][
        "outer_v2_plan"
    ][
        "sha256"
    ]


def test_python_syntax():
    ast.parse(
        EXECUTOR.read_text()
    )
