from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


from crosslayer_resilience.baseline import (  # noqa: E402
    argmax_from_logits,
    decision_from_logits,
)
from crosslayer_resilience.baseline.state import (  # noqa: E402
    canonical_state_sha256,
    load_date2025_from_state_artifact,
)


STATE_ARTIFACT = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_state_dict_v1.pt"
)

EXPECTED_STATE_SHA256 = (
    "c98987476536320191f8875316cf8cae"
    "eb0b3f2edc51d65be7ac7d2eaea03124"
)

ARTIFACT = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_fp32_v1.onnx"
)

TEMP_ARTIFACT = (
    ROOT
    / "artifacts"
    / "baseline"
    / ".date2025_cnn400_fp32_v1.onnx.tmp"
)

RESULT = (
    ROOT
    / "manifests"
    / "phase_2b_fp32_onnx_validation_v1.json"
)

OPSET_VERSION = 13

INPUT_NAME = "imu_window"
OUTPUT_NAME = "logits"

INPUT_SHAPE = (1, 40, 9)
OUTPUT_SHAPE = (1, 2)

ATOL = 1.0e-5
RTOL = 1.0e-5

STRUCTURED_COUNT = 68
SEEDED_RANDOM_COUNT = 256
TOTAL_VECTOR_COUNT = (
    STRUCTURED_COUNT
    + SEEDED_RANDOM_COUNT
)

RANDOM_SEED = 260930


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def structured_vectors() -> np.ndarray:
    vectors = [
        np.zeros(
            INPUT_SHAPE,
            dtype=np.float32,
        ),
        np.ones(
            INPUT_SHAPE,
            dtype=np.float32,
        ),
        -np.ones(
            INPUT_SHAPE,
            dtype=np.float32,
        ),
        np.linspace(
            -1.0,
            1.0,
            num=360,
            dtype=np.float32,
        ).reshape(
            INPUT_SHAPE
        ),
    ]

    for index in range(64):
        window = np.empty(
            INPUT_SHAPE,
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

                window[
                    0,
                    t,
                    c,
                ] = np.float32(
                    numerator / 64.0
                )

        vectors.append(window)

    result = np.concatenate(
        vectors,
        axis=0,
    )

    if result.shape != (
        STRUCTURED_COUNT,
        40,
        9,
    ):
        raise RuntimeError(
            "Structured-vector contract changed"
        )

    return result


def seeded_random_vectors() -> np.ndarray:
    """
    Numerical stress vectors only.

    They are not dataset samples and are not used for scientific performance
    evaluation or quantization calibration.
    """
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    result = rng.normal(
        loc=0.0,
        scale=2500.0,
        size=(
            SEEDED_RANDOM_COUNT,
            40,
            9,
        ),
    ).astype(
        np.float32
    )

    return result


def parity_vectors() -> np.ndarray:
    result = np.concatenate(
        [
            structured_vectors(),
            seeded_random_vectors(),
        ],
        axis=0,
    )

    if result.shape != (
        TOTAL_VECTOR_COUNT,
        40,
        9,
    ):
        raise RuntimeError(
            "Parity vector count/shape changed"
        )

    return result


def tensor_shape(
    value_info,
) -> list[int]:
    dimensions = []

    for dim in (
        value_info
        .type
        .tensor_type
        .shape
        .dim
    ):
        if dim.dim_param:
            raise RuntimeError(
                "Dynamic ONNX dimension found"
            )

        if not dim.HasField(
            "dim_value"
        ):
            raise RuntimeError(
                "ONNX dimension missing fixed value"
            )

        dimensions.append(
            int(dim.dim_value)
        )

    return dimensions


def export_model(
    model: torch.nn.Module,
) -> None:
    ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEMP_ARTIFACT.exists():
        TEMP_ARTIFACT.unlink()

    dummy = torch.zeros(
        INPUT_SHAPE,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(TEMP_ARTIFACT),
            export_params=True,
            verbose=False,
            input_names=[
                INPUT_NAME,
            ],
            output_names=[
                OUTPUT_NAME,
            ],
            opset_version=OPSET_VERSION,
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

    if not TEMP_ARTIFACT.is_file():
        raise RuntimeError(
            "ONNX export failed to create artifact"
        )

    TEMP_ARTIFACT.replace(
        ARTIFACT
    )


def validate_graph() -> dict:
    graph_model = onnx.load(
        str(ARTIFACT),
        load_external_data=False,
    )

    onnx.checker.check_model(
        graph_model
    )

    graph = graph_model.graph

    if len(graph.input) != 1:
        raise RuntimeError(
            "Expected exactly one graph input"
        )

    if len(graph.output) != 1:
        raise RuntimeError(
            "Expected exactly one graph output"
        )

    graph_input = graph.input[0]
    graph_output = graph.output[0]

    if graph_input.name != INPUT_NAME:
        raise RuntimeError(
            "ONNX input name changed"
        )

    if graph_output.name != OUTPUT_NAME:
        raise RuntimeError(
            "ONNX output name changed"
        )

    input_shape = tensor_shape(
        graph_input
    )

    output_shape = tensor_shape(
        graph_output
    )

    if input_shape != list(
        INPUT_SHAPE
    ):
        raise RuntimeError(
            f"Input shape mismatch: {input_shape}"
        )

    if output_shape != list(
        OUTPUT_SHAPE
    ):
        raise RuntimeError(
            f"Output shape mismatch: {output_shape}"
        )

    opset = {
        entry.domain or "ai.onnx":
            int(entry.version)
        for entry
        in graph_model.opset_import
    }

    operator_counts = Counter(
        node.op_type
        for node in graph.node
    )

    initializer_bytes = sum(
        len(
            tensor.raw_data
        )
        for tensor
        in graph.initializer
    )

    return {
        "input_name":
            graph_input.name,

        "output_name":
            graph_output.name,

        "input_shape":
            input_shape,

        "output_shape":
            output_shape,

        "opset_imports":
            dict(
                sorted(
                    opset.items()
                )
            ),

        "node_count":
            len(graph.node),

        "initializer_count":
            len(graph.initializer),

        "initializer_raw_bytes":
            initializer_bytes,

        "operator_counts":
            dict(
                sorted(
                    operator_counts.items()
                )
            ),
    }


def run_onnx(
    inputs: np.ndarray,
) -> tuple[
    np.ndarray,
    list[str],
]:
    """
    Execute parity vectors against the fixed batch-1 deployment graph.

    The exported deployment tensor contract is intentionally:

        [1, 40, 9] -> [1, 2]

    Numerical parity may contain many vectors, but they are therefore run
    serially through the same batch-1 graph rather than changing the exported
    deployment contract to a dynamic batch dimension.
    """
    providers = (
        ort.get_available_providers()
    )

    if (
        "CPUExecutionProvider"
        not in providers
    ):
        raise RuntimeError(
            "CPUExecutionProvider unavailable"
        )

    array = np.asarray(
        inputs,
        dtype=np.float32,
    )

    if array.ndim != 3:
        raise RuntimeError(
            "Expected parity tensor [N,40,9], "
            f"got {array.shape}"
        )

    if tuple(
        array.shape[1:]
    ) != (
        INPUT_SHAPE[1],
        INPUT_SHAPE[2],
    ):
        raise RuntimeError(
            "Parity-vector sample shape changed: "
            f"{array.shape}"
        )

    if array.shape[0] <= 0:
        raise RuntimeError(
            "Parity-vector set is empty"
        )

    options = (
        ort.SessionOptions()
    )

    # Avoid backend-specific aggressive transformations during
    # reference parity validation.
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel
        .ORT_ENABLE_BASIC
    )

    session = ort.InferenceSession(
        str(ARTIFACT),
        sess_options=options,
        providers=[
            "CPUExecutionProvider",
        ],
    )

    outputs = []

    for index in range(
        array.shape[0]
    ):
        sample = np.ascontiguousarray(
            array[
                index:index + 1
            ],
            dtype=np.float32,
        )

        if tuple(
            sample.shape
        ) != INPUT_SHAPE:
            raise RuntimeError(
                "Serialized ONNX input shape changed: "
                f"{sample.shape}"
            )

        output = session.run(
            [
                OUTPUT_NAME,
            ],
            {
                INPUT_NAME:
                    sample,
            },
        )[0]

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        if tuple(
            output.shape
        ) != OUTPUT_SHAPE:
            raise RuntimeError(
                "Serialized ONNX output shape changed: "
                f"{output.shape}"
            )

        outputs.append(
            output
        )

    combined = np.concatenate(
        outputs,
        axis=0,
    )

    expected_combined_shape = (
        array.shape[0],
        OUTPUT_SHAPE[1],
    )

    if tuple(
        combined.shape
    ) != expected_combined_shape:
        raise RuntimeError(
            "Combined ONNX parity output shape changed: "
            f"{combined.shape}"
        )

    return (
        combined,
        providers,
    )


def main() -> int:
    if not STATE_ARTIFACT.is_file():
        raise RuntimeError(
            "Validated state artifact missing"
        )

    model = (
        load_date2025_from_state_artifact(
            STATE_ARTIFACT,
            expected_state_sha256=(
                EXPECTED_STATE_SHA256
            ),
        )
    )

    observed_state_sha = (
        canonical_state_sha256(
            model.state_dict()
        )
    )

    if (
        observed_state_sha
        != EXPECTED_STATE_SHA256
    ):
        raise RuntimeError(
            "Canonical task state changed"
        )

    model.cpu()
    model.eval()

    export_model(
        model
    )

    graph = validate_graph()

    vectors = parity_vectors()

    torch_input = torch.from_numpy(
        vectors
    )

    with torch.inference_mode():
        torch_logits = (
            model(
                torch_input
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    (
        onnx_logits,
        providers,
    ) = run_onnx(
        vectors
    )

    if (
        torch_logits.shape
        != onnx_logits.shape
    ):
        raise RuntimeError(
            "PyTorch/ONNX output shape mismatch"
        )

    absolute = np.abs(
        torch_logits
        - onnx_logits
    )

    max_abs = float(
        absolute.max()
    )

    mean_abs = float(
        absolute.mean()
    )

    reference_scale = np.maximum(
        np.abs(
            torch_logits
        ),
        np.float32(1.0e-12),
    )

    max_relative = float(
        (
            absolute
            / reference_scale
        ).max()
    )

    close = bool(
        np.allclose(
            torch_logits,
            onnx_logits,
            atol=ATOL,
            rtol=RTOL,
        )
    )

    if not close:
        raise RuntimeError(
            "PyTorch/ONNX numerical parity failed. "
            f"max_abs={max_abs}, "
            f"max_relative={max_relative}"
        )

    torch_tensor = torch.from_numpy(
        torch_logits
    )

    onnx_tensor = torch.from_numpy(
        onnx_logits
    )

    torch_argmax = (
        argmax_from_logits(
            torch_tensor
        )
    )

    onnx_argmax = (
        argmax_from_logits(
            onnx_tensor
        )
    )

    argmax_equal = bool(
        torch.equal(
            torch_argmax,
            onnx_argmax,
        )
    )

    if not argmax_equal:
        raise RuntimeError(
            "Argmax decisions differ after ONNX export"
        )

    torch_historical = (
        decision_from_logits(
            torch_tensor
        )
    )

    onnx_historical = (
        decision_from_logits(
            onnx_tensor
        )
    )

    historical_equal = bool(
        torch.equal(
            torch_historical,
            onnx_historical,
        )
    )

    if not historical_equal:
        raise RuntimeError(
            "Historical decisions differ after ONNX export"
        )

    structured_slice = slice(
        0,
        STRUCTURED_COUNT,
    )

    random_slice = slice(
        STRUCTURED_COUNT,
        TOTAL_VECTOR_COUNT,
    )

    structured_max_abs = float(
        np.abs(
            torch_logits[
                structured_slice
            ]
            - onnx_logits[
                structured_slice
            ]
        ).max()
    )

    random_max_abs = float(
        np.abs(
            torch_logits[
                random_slice
            ]
            - onnx_logits[
                random_slice
            ]
        ).max()
    )

    result = {
        "schema":
            "crosslayer_phase2b_fp32_onnx_validation_v1",

        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),

        "status":
            "PASS",

        "baseline": {
            "id":
                "DATE2025_CNN_400MS_RECONSTRUCTED",

            "freeze_status":
                "NOT_FROZEN",

            "canonical_state_sha256":
                observed_state_sha,

            "parameter_count":
                sum(
                    int(
                        parameter.numel()
                    )
                    for parameter
                    in model.parameters()
                ),
        },

        "export": {
            "format":
                "ONNX",

            "precision":
                "FP32",

            "opset":
                OPSET_VERSION,

            "artifact_path":
                str(
                    ARTIFACT.relative_to(
                        ROOT
                    )
                ),

            "artifact_sha256":
                sha256_file(
                    ARTIFACT
                ),

            "artifact_bytes":
                ARTIFACT.stat().st_size,

            "git_tracked":
                False,

            "graph":
                graph,
        },

        "parity": {
            "structured_vector_count":
                STRUCTURED_COUNT,

            "seeded_random_vector_count":
                SEEDED_RANDOM_COUNT,

            "total_vector_count":
                TOTAL_VECTOR_COUNT,

            "random_seed":
                RANDOM_SEED,

            "atol":
                ATOL,

            "rtol":
                RTOL,

            "max_abs_difference":
                max_abs,

            "mean_abs_difference":
                mean_abs,

            "max_relative_difference":
                max_relative,

            "structured_max_abs_difference":
                structured_max_abs,

            "seeded_random_max_abs_difference":
                random_max_abs,

            "allclose":
                close,

            "argmax_decisions_equal":
                argmax_equal,

            "historical_decisions_equal":
                historical_equal,
        },

        "runtime": {
            "onnx_version":
                onnx.__version__,

            "onnxruntime_version":
                ort.__version__,

            "available_providers":
                providers,

            "validation_provider":
                "CPUExecutionProvider",

            "graph_optimization":
                "ORT_ENABLE_BASIC",

            "python":
                platform.python_version(),

            "torch":
                torch.__version__,

            "numpy":
                np.__version__,
        },

        "scientific_boundary": {
            "dataset_samples_used":
                False,

            "quantization_calibration_used":
                False,

            "dataset_metrics_frozen":
                False,

            "decision_semantics_frozen":
                False,

            "quantized_model_frozen":
                False,
        },
    }

    RESULT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "ONNX artifact:",
        ARTIFACT,
    )

    print(
        "ONNX SHA256:",
        result["export"][
            "artifact_sha256"
        ],
    )

    print(
        "ONNX bytes:",
        result["export"][
            "artifact_bytes"
        ],
    )

    print(
        "ONNX nodes:",
        graph[
            "node_count"
        ],
    )

    print(
        "ONNX operators:",
        graph[
            "operator_counts"
        ],
    )

    print(
        "Parity vectors:",
        TOTAL_VECTOR_COUNT,
    )

    print(
        "Max absolute difference:",
        max_abs,
    )

    print(
        "Structured max difference:",
        structured_max_abs,
    )

    print(
        "Random max difference:",
        random_max_abs,
    )

    print(
        "Argmax parity:",
        argmax_equal,
    )

    print(
        "Historical decision parity:",
        historical_equal,
    )

    print()
    print(
        "PHASE_2B_FP32_ONNX_PARITY=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
