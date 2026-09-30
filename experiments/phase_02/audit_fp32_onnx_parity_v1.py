from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from crosslayer_resilience.baseline import (  # noqa: E402
    argmax_from_logits,
    decision_from_logits,
)
from crosslayer_resilience.baseline.state import (  # noqa: E402
    load_date2025_from_state_artifact,
)


STATE = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_state_dict_v1.pt"
)

ONNX_PATH = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_fp32_v1.onnx"
)

TMP1 = (
    ROOT
    / "artifacts"
    / "baseline"
    / ".repeat_export_1.onnx"
)

TMP2 = (
    ROOT
    / "artifacts"
    / "baseline"
    / ".repeat_export_2.onnx"
)

SOURCE_MANIFEST = (
    ROOT
    / "manifests"
    / "phase_2b_fp32_onnx_validation_v1.json"
)

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_2b_fp32_onnx_hardening_v1.json"
)

EXPECTED_STATE_SHA = (
    "c98987476536320191f8875316cf8cae"
    "eb0b3f2edc51d65be7ac7d2eaea03124"
)

EXPECTED_ONNX_SHA = (
    "f3a55933bab5ed63225a1647a8b78aa"
    "909c4bb5a81405b06f235ba1b3bf6de1f"
)

ATOL = 1.0e-5
RTOL = 1.0e-5

STRUCTURED_COUNT = 68
RANDOM_COUNT = 256
TOTAL_COUNT = 324
RANDOM_SEED = 260930

INPUT_NAME = "imu_window"
OUTPUT_NAME = "logits"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def vectors() -> np.ndarray:
    items = [
        np.zeros(
            (1, 40, 9),
            dtype=np.float32,
        ),
        np.ones(
            (1, 40, 9),
            dtype=np.float32,
        ),
        -np.ones(
            (1, 40, 9),
            dtype=np.float32,
        ),
        np.linspace(
            -1.0,
            1.0,
            num=360,
            dtype=np.float32,
        ).reshape(
            1,
            40,
            9,
        ),
    ]

    for index in range(64):
        window = np.empty(
            (1, 40, 9),
            dtype=np.float32,
        )

        for t in range(40):
            for c in range(9):
                numerator = (
                    (
                        index * 37
                        + t * 17
                        + c * 13
                    )
                    % 257
                ) - 128

                window[0, t, c] = np.float32(
                    numerator / 64.0
                )

        items.append(window)

    structured = np.concatenate(
        items,
        axis=0,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    random = rng.normal(
        loc=0.0,
        scale=2500.0,
        size=(
            RANDOM_COUNT,
            40,
            9,
        ),
    ).astype(
        np.float32
    )

    combined = np.concatenate(
        [
            structured,
            random,
        ],
        axis=0,
    )

    if combined.shape != (
        TOTAL_COUNT,
        40,
        9,
    ):
        raise RuntimeError(
            f"Unexpected vector shape: {combined.shape}"
        )

    return combined


def onnx_session(
    path: Path,
) -> ort.InferenceSession:
    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel
        .ORT_ENABLE_BASIC
    )

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def run_serial(
    session: ort.InferenceSession,
    x: np.ndarray,
) -> np.ndarray:
    outputs = []

    for index in range(
        x.shape[0]
    ):
        sample = np.ascontiguousarray(
            x[
                index:index + 1
            ],
            dtype=np.float32,
        )

        result = session.run(
            [OUTPUT_NAME],
            {
                INPUT_NAME: sample,
            },
        )[0]

        result = np.asarray(
            result,
            dtype=np.float32,
        )

        if result.shape != (
            1,
            2,
        ):
            raise RuntimeError(
                f"Unexpected ONNX result shape: {result.shape}"
            )

        outputs.append(result)

    return np.concatenate(
        outputs,
        axis=0,
    )


def export(
    model: torch.nn.Module,
    path: Path,
) -> None:
    if path.exists():
        path.unlink()

    dummy = torch.zeros(
        1,
        40,
        9,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(path),
            export_params=True,
            verbose=False,
            input_names=[
                INPUT_NAME,
            ],
            output_names=[
                OUTPUT_NAME,
            ],
            opset_version=13,
            dynamic_axes=None,
            keep_initializers_as_inputs=False,
            training=(
                torch.onnx
                .TrainingMode
                .EVAL
            ),
            do_constant_folding=True,
            dynamo=False,
        )


def graph_summary(
    path: Path,
) -> dict:
    model = onnx.load(
        str(path),
        load_external_data=False,
    )

    graph = model.graph

    initializer_hashes = {}

    for tensor in graph.initializer:
        payload = (
            tensor.raw_data
            if tensor.raw_data
            else tensor.SerializeToString()
        )

        initializer_hashes[
            tensor.name
        ] = hashlib.sha256(
            payload
        ).hexdigest()

    nodes = [
        {
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for node in graph.node
    ]

    return {
        "node_count":
            len(graph.node),

        "initializer_count":
            len(graph.initializer),

        "nodes":
            nodes,

        "initializer_sha256":
            dict(
                sorted(
                    initializer_hashes.items()
                )
            ),
    }


def main() -> int:
    source = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if source["status"] != "PASS":
        raise RuntimeError(
            "Source Phase-2B validation did not pass"
        )

    if sha256(
        ONNX_PATH
    ) != EXPECTED_ONNX_SHA:
        raise RuntimeError(
            "Existing ONNX artifact SHA changed"
        )

    model = (
        load_date2025_from_state_artifact(
            STATE,
            expected_state_sha256=(
                EXPECTED_STATE_SHA
            ),
        )
    )

    model.cpu()
    model.eval()

    x = vectors()

    with torch.inference_mode():
        torch_logits = (
            model(
                torch.from_numpy(x)
            )
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    ort_logits = run_serial(
        onnx_session(
            ONNX_PATH
        ),
        x,
    )

    absolute = np.abs(
        torch_logits
        - ort_logits
    )

    tolerance = (
        ATOL
        + RTOL
        * np.abs(
            torch_logits
        )
    )

    violations = (
        absolute
        > tolerance
    )

    violation_count = int(
        violations.sum()
    )

    if violation_count != 0:
        raise RuntimeError(
            f"Elementwise allclose violations: {violation_count}"
        )

    relative = (
        absolute
        / np.maximum(
            np.abs(
                torch_logits
            ),
            np.float32(
                1.0e-12
            ),
        )
    )

    flat_index = int(
        np.argmax(
            absolute
        )
    )

    sample_index, class_index = (
        np.unravel_index(
            flat_index,
            absolute.shape,
        )
    )

    worst_partition = (
        "structured"
        if sample_index < STRUCTURED_COUNT
        else "seeded_random_stress"
    )

    worst_relative = float(
        relative[
            sample_index,
            class_index,
        ]
    )

    max_relative = float(
        relative.max()
    )

    max_abs = float(
        absolute.max()
    )

    structured_max_abs = float(
        absolute[
            :STRUCTURED_COUNT
        ].max()
    )

    random_max_abs = float(
        absolute[
            STRUCTURED_COUNT:
        ].max()
    )

    torch_tensor = torch.from_numpy(
        torch_logits
    )

    ort_tensor = torch.from_numpy(
        ort_logits
    )

    torch_argmax = argmax_from_logits(
        torch_tensor
    )

    ort_argmax = argmax_from_logits(
        ort_tensor
    )

    argmax_diff_count = int(
        (
            torch_argmax
            != ort_argmax
        )
        .sum()
        .item()
    )

    torch_historical = decision_from_logits(
        torch_tensor
    )

    ort_historical = decision_from_logits(
        ort_tensor
    )

    historical_diff_count = int(
        (
            torch_historical
            != ort_historical
        )
        .sum()
        .item()
    )

    # Classification margin between two logits.
    torch_margin = np.abs(
        torch_logits[:, 0]
        - torch_logits[:, 1]
    )

    ort_margin = np.abs(
        ort_logits[:, 0]
        - ort_logits[:, 1]
    )

    min_torch_margin = float(
        torch_margin.min()
    )

    min_ort_margin = float(
        ort_margin.min()
    )

    minimum_margin_index = int(
        np.argmin(
            torch_margin
        )
    )

    # Historical threshold distance in probability space.
    torch_probability = torch.softmax(
        torch_tensor,
        dim=1,
    )[:, 1].numpy()

    ort_probability = torch.softmax(
        ort_tensor,
        dim=1,
    )[:, 1].numpy()

    torch_threshold_distance = np.abs(
        torch_probability
        - 0.9
    )

    ort_threshold_distance = np.abs(
        ort_probability
        - 0.9
    )

    min_torch_threshold_distance = float(
        torch_threshold_distance.min()
    )

    min_ort_threshold_distance = float(
        ort_threshold_distance.min()
    )

    # Repeat-export determinism.
    export(
        model,
        TMP1,
    )

    export(
        model,
        TMP2,
    )

    sha_original = sha256(
        ONNX_PATH
    )

    sha_repeat_1 = sha256(
        TMP1
    )

    sha_repeat_2 = sha256(
        TMP2
    )

    byte_deterministic = bool(
        sha_original
        == sha_repeat_1
        == sha_repeat_2
    )

    graph_original = graph_summary(
        ONNX_PATH
    )

    graph_repeat_1 = graph_summary(
        TMP1
    )

    graph_repeat_2 = graph_summary(
        TMP2
    )

    structural_deterministic = bool(
        graph_original
        == graph_repeat_1
        == graph_repeat_2
    )

    if not structural_deterministic:
        raise RuntimeError(
            "Repeated ONNX exports changed graph structure or initializers"
        )

    # Also require repeated artifacts to execute identically.
    repeat_1_logits = run_serial(
        onnx_session(
            TMP1
        ),
        x,
    )

    repeat_2_logits = run_serial(
        onnx_session(
            TMP2
        ),
        x,
    )

    repeat_output_equal = bool(
        np.array_equal(
            ort_logits,
            repeat_1_logits,
        )
        and np.array_equal(
            ort_logits,
            repeat_2_logits,
        )
    )

    if not repeat_output_equal:
        raise RuntimeError(
            "Repeated export changed ONNX Runtime outputs"
        )

    result = {
        "schema":
            "crosslayer_phase2b_fp32_onnx_hardening_v1",

        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),

        "status":
            "PASS",

        "baseline_id":
            "DATE2025_CNN_400MS_RECONSTRUCTED",

        "baseline_freeze_status":
            "NOT_FROZEN",

        "onnx_sha256":
            sha_original,

        "deployment_contract": {
            "input_shape":
                [1, 40, 9],

            "output_shape":
                [1, 2],

            "batch":
                1,

            "parity_execution":
                "fixed_batch_1_serial",
        },

        "parity": {
            "structured_count":
                STRUCTURED_COUNT,

            "seeded_random_stress_count":
                RANDOM_COUNT,

            "total_count":
                TOTAL_COUNT,

            "atol":
                ATOL,

            "rtol":
                RTOL,

            "elementwise_tolerance_violation_count":
                violation_count,

            "max_abs_difference":
                max_abs,

            "structured_max_abs_difference":
                structured_max_abs,

            "seeded_random_max_abs_difference":
                random_max_abs,

            "max_relative_difference":
                max_relative,

            "worst_case": {
                "sample_index":
                    int(
                        sample_index
                    ),

                "class_index":
                    int(
                        class_index
                    ),

                "partition":
                    worst_partition,

                "torch_logit":
                    float(
                        torch_logits[
                            sample_index,
                            class_index,
                        ]
                    ),

                "onnx_logit":
                    float(
                        ort_logits[
                            sample_index,
                            class_index,
                        ]
                    ),

                "absolute_difference":
                    float(
                        absolute[
                            sample_index,
                            class_index,
                        ]
                    ),

                "relative_difference":
                    worst_relative,

                "allowed_tolerance":
                    float(
                        tolerance[
                            sample_index,
                            class_index,
                        ]
                    ),
            },

            "argmax_decision_difference_count":
                argmax_diff_count,

            "historical_decision_difference_count":
                historical_diff_count,

            "minimum_torch_argmax_margin":
                min_torch_margin,

            "minimum_onnx_argmax_margin":
                min_ort_margin,

            "minimum_argmax_margin_sample":
                minimum_margin_index,

            "minimum_torch_distance_to_historical_threshold":
                min_torch_threshold_distance,

            "minimum_onnx_distance_to_historical_threshold":
                min_ort_threshold_distance,
        },

        "repeat_export": {
            "original_sha256":
                sha_original,

            "repeat_1_sha256":
                sha_repeat_1,

            "repeat_2_sha256":
                sha_repeat_2,

            "byte_identical":
                byte_deterministic,

            "graph_and_initializer_identical":
                structural_deterministic,

            "runtime_outputs_bit_identical":
                repeat_output_equal,
        },

        "acceptance_candidate": {
            "structured_vectors": (
                "max_abs_difference <= 1e-5"
            ),

            "numerical_stress_vectors": (
                "elementwise abs(error) <= "
                "1e-5 + 1e-5 * abs(reference)"
            ),

            "all_vectors": [
                "zero elementwise tolerance violations",
                "zero argmax decision differences",
                "zero historical decision differences",
            ],

            "purpose": (
                "FP32 representation parity only; "
                "not dataset-performance evidence"
            ),
        },

        "scientific_boundary": {
            "dataset_samples_used":
                False,

            "quantization_calibration_used":
                False,

            "task_metrics_frozen":
                False,

            "decision_semantics_frozen":
                False,
        },
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Tolerance violations:",
        violation_count,
    )

    print(
        "Max absolute difference:",
        max_abs,
    )

    print(
        "Max relative difference:",
        max_relative,
    )

    print(
        "Structured max absolute difference:",
        structured_max_abs,
    )

    print(
        "Random-stress max absolute difference:",
        random_max_abs,
    )

    print(
        "Worst-case partition:",
        worst_partition,
    )

    print(
        "Worst PyTorch logit:",
        result[
            "parity"
        ][
            "worst_case"
        ][
            "torch_logit"
        ],
    )

    print(
        "Worst ONNX logit:",
        result[
            "parity"
        ][
            "worst_case"
        ][
            "onnx_logit"
        ],
    )

    print(
        "Worst allowed tolerance:",
        result[
            "parity"
        ][
            "worst_case"
        ][
            "allowed_tolerance"
        ],
    )

    print(
        "Argmax decision differences:",
        argmax_diff_count,
    )

    print(
        "Historical decision differences:",
        historical_diff_count,
    )

    print(
        "Repeat export byte-identical:",
        byte_deterministic,
    )

    print(
        "Repeat export graph-identical:",
        structural_deterministic,
    )

    print(
        "Repeat runtime outputs bit-identical:",
        repeat_output_equal,
    )

    TMP1.unlink(
        missing_ok=True
    )

    TMP2.unlink(
        missing_ok=True
    )

    print()
    print(
        "PHASE_2B_HARDENING=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
