from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]

FP32 = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_fp32_v1.onnx"
)

DYNAMIC = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_dynamic_int8_smoke.onnx"
)

OUTPUT = (
    ROOT
    / "manifests"
    / "phase_2c_quantization_support_audit_v1.json"
)

EXPECTED_FP32_SHA = (
    "f3a55933bab5ed63225a1647a8b78aa"
    "909c4bb5a81405b06f235ba1b3bf6de1f"
)

INPUT_NAME = "imu_window"
OUTPUT_NAME = "logits"

STRUCTURED_COUNT = 68


def sha256(
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

                window[
                    0,
                    t,
                    c,
                ] = np.float32(
                    numerator / 64.0
                )

        vectors.append(
            window
        )

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


def session(
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
    runtime: ort.InferenceSession,
    data: np.ndarray,
) -> np.ndarray:
    outputs = []

    for index in range(
        data.shape[0]
    ):
        sample = np.ascontiguousarray(
            data[
                index:index + 1
            ],
            dtype=np.float32,
        )

        result = runtime.run(
            [OUTPUT_NAME],
            {
                INPUT_NAME: sample,
            },
        )[0]

        outputs.append(
            np.asarray(
                result,
                dtype=np.float32,
            )
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def graph_inventory(
    path: Path,
) -> dict:
    model = onnx.load(
        str(path),
        load_external_data=False,
    )

    onnx.checker.check_model(
        model
    )

    operators = Counter(
        node.op_type
        for node
        in model.graph.node
    )

    nodes = []

    for index, node in enumerate(
        model.graph.node
    ):
        nodes.append(
            {
                "index":
                    index,

                "name":
                    node.name,

                "op_type":
                    node.op_type,

                "inputs":
                    list(
                        node.input
                    ),

                "outputs":
                    list(
                        node.output
                    ),
            }
        )

    return {
        "node_count":
            len(
                model.graph.node
            ),

        "operator_counts":
            dict(
                sorted(
                    operators.items()
                )
            ),

        "nodes":
            nodes,
    }


def registry_keys(
    registry_module,
    attribute: str,
) -> list[str]:
    value = getattr(
        registry_module,
        attribute,
        None,
    )

    if isinstance(
        value,
        dict,
    ):
        return sorted(
            str(key)
            for key
            in value.keys()
        )

    return []


def decision_arrays(
    logits: np.ndarray,
) -> dict:
    argmax = np.argmax(
        logits,
        axis=1,
    ).astype(
        np.int64
    )

    shifted = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(
        shifted
    )

    probs = (
        exp
        / np.sum(
            exp,
            axis=1,
            keepdims=True,
        )
    )

    historical = np.zeros(
        logits.shape[0],
        dtype=np.int64,
    )

    max_prob = np.max(
        probs,
        axis=1,
    )

    argmax_prob = np.argmax(
        probs,
        axis=1,
    )

    accepted = (
        max_prob
        > 0.9
    )

    historical[
        accepted
    ] = argmax_prob[
        accepted
    ]

    return {
        "argmax":
            argmax,

        "historical":
            historical,
    }


def main() -> int:
    if not FP32.is_file():
        raise RuntimeError(
            "FP32 ONNX artifact missing"
        )

    if sha256(
        FP32
    ) != EXPECTED_FP32_SHA:
        raise RuntimeError(
            "FP32 ONNX checksum changed"
        )

    graph = graph_inventory(
        FP32
    )

    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )

        quantization_import = True

    except Exception as exc:
        quantization_import = False

        result = {
            "schema":
                "crosslayer_phase2c_quantization_support_audit_v1",

            "generated_utc": (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            ),

            "status":
                "BLOCKED",

            "reason":
                repr(exc),

            "fp32_graph":
                graph,

            "scientific_boundary": {
                "project_dataset_samples_used":
                    False,

                "final_static_calibration_performed":
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

        raise RuntimeError(
            "ONNX Runtime quantization API unavailable"
        )

    from onnxruntime.quantization import (
        registry as ort_registry,
    )

    qlinear_registry = registry_keys(
        ort_registry,
        "QLinearOpsRegistry",
    )

    integer_registry = registry_keys(
        ort_registry,
        "IntegerOpsRegistry",
    )

    qdq_registry = registry_keys(
        ort_registry,
        "QDQRegistry",
    )

    graph_ops = sorted(
        graph[
            "operator_counts"
        ].keys()
    )

    support = {}

    for op in graph_ops:
        support[op] = {
            "node_count":
                int(
                    graph[
                        "operator_counts"
                    ][op]
                ),

            "qlinear_registry":
                op
                in qlinear_registry,

            "integer_registry":
                op
                in integer_registry,

            "qdq_registry":
                op
                in qdq_registry,
        }

    calibration_methods = []

    try:
        calibration_methods = sorted(
            str(name)
            for name
            in CalibrationMethod.__members__.keys()
        )
    except Exception:
        pass

    quant_formats = []

    try:
        quant_formats = sorted(
            str(name)
            for name
            in QuantFormat.__members__.keys()
        )
    except Exception:
        pass

    quant_types = []

    try:
        quant_types = sorted(
            str(name)
            for name
            in QuantType.__members__.keys()
        )
    except Exception:
        pass

    api_signatures = {
        "quantize_static":
            str(
                inspect.signature(
                    quantize_static
                )
            ),

        "quantize_dynamic":
            str(
                inspect.signature(
                    quantize_dynamic
                )
            ),
    }

    # ----------------------------------------------------------
    # Dynamic quantization smoke test.
    #
    # This uses no calibration data and is NOT the deployment
    # quantization candidate. It only validates the installed
    # toolchain and reveals which graph pieces the current ORT
    # dynamic quantizer transforms.
    # ----------------------------------------------------------

    if DYNAMIC.exists():
        DYNAMIC.unlink()

    dynamic = {
        "attempted":
            True,

        "scientific_role":
            "toolchain_smoke_only",

        "deployment_candidate":
            False,

        "project_dataset_samples_used":
            False,
    }

    try:
        quantize_dynamic(
            model_input=str(
                FP32
            ),
            model_output=str(
                DYNAMIC
            ),
            per_channel=True,
            reduce_range=False,
            weight_type=(
                QuantType.QInt8
            ),
        )

        if not DYNAMIC.is_file():
            raise RuntimeError(
                "Dynamic quantizer produced no artifact"
            )

        dynamic_graph = graph_inventory(
            DYNAMIC
        )

        x = structured_vectors()

        fp32_logits = run_serial(
            session(
                FP32
            ),
            x,
        )

        dynamic_logits = run_serial(
            session(
                DYNAMIC
            ),
            x,
        )

        absolute = np.abs(
            fp32_logits
            - dynamic_logits
        )

        fp32_decisions = decision_arrays(
            fp32_logits
        )

        dynamic_decisions = decision_arrays(
            dynamic_logits
        )

        dynamic.update(
            {
                "status":
                    "PASS",

                "artifact_sha256":
                    sha256(
                        DYNAMIC
                    ),

                "artifact_bytes":
                    DYNAMIC.stat().st_size,

                "graph":
                    dynamic_graph,

                "structured_vector_count":
                    STRUCTURED_COUNT,

                "max_abs_logit_difference":
                    float(
                        absolute.max()
                    ),

                "mean_abs_logit_difference":
                    float(
                        absolute.mean()
                    ),

                "argmax_decision_difference_count":
                    int(
                        np.sum(
                            fp32_decisions[
                                "argmax"
                            ]
                            != dynamic_decisions[
                                "argmax"
                            ]
                        )
                    ),

                "historical_decision_difference_count":
                    int(
                        np.sum(
                            fp32_decisions[
                                "historical"
                            ]
                            != dynamic_decisions[
                                "historical"
                            ]
                        )
                    ),
            }
        )

    except Exception as exc:
        dynamic.update(
            {
                "status":
                    "FAILED_NONBLOCKING",

                "error":
                    repr(
                        exc
                    ),
            }
        )

    result = {
        "schema":
            "crosslayer_phase2c_quantization_support_audit_v1",

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

        "fp32_artifact": {
            "sha256":
                sha256(
                    FP32
                ),

            "bytes":
                FP32.stat().st_size,

            "graph":
                graph,
        },

        "ort_quantization": {
            "import_available":
                quantization_import,

            "onnxruntime_version":
                ort.__version__,

            "calibration_methods":
                calibration_methods,

            "quant_formats":
                quant_formats,

            "quant_types":
                quant_types,

            "qlinear_registry":
                qlinear_registry,

            "integer_registry":
                integer_registry,

            "qdq_registry":
                qdq_registry,

            "api_signatures":
                api_signatures,
        },

        "graph_operator_support": {
            "operators":
                support,

            "all_graph_operator_types":
                graph_ops,
        },

        "dynamic_quantization_smoke":
            dynamic,

        "protocol_direction": {
            "primary_candidate_method":
                "static_post_training_quantization",

            "final_calibration_status":
                "DEFERRED",

            "reason":
                (
                    "representative calibration data must come "
                    "from a frozen non-test calibration partition"
                ),

            "weight_identity_policy":
                (
                    "historical FP32 task weights remain source "
                    "of the quantized deployment candidate"
                ),

            "qat_primary_candidate":
                False,

            "qat_reason":
                (
                    "QAT would modify the recovered historical "
                    "task weights and therefore changes baseline identity"
                ),

            "dynamic_quantization_primary_candidate":
                False,

            "dynamic_quantization_role":
                "toolchain_smoke_only",
        },

        "scientific_boundary": {
            "project_dataset_samples_used":
                False,

            "final_static_calibration_performed":
                False,

            "final_int8_model_created":
                False,

            "dataset_performance_metrics_used":
                False,

            "fault_robustness_used_for_quantizer_selection":
                False,

            "final_test_used":
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
        "Graph operators:",
        graph[
            "operator_counts"
        ],
    )

    print()
    print(
        "QLinear registry intersection:"
    )

    for op in graph_ops:
        if (
            op
            in qlinear_registry
        ):
            print(
                " ",
                op,
            )

    print()
    print(
        "Integer registry intersection:"
    )

    for op in graph_ops:
        if (
            op
            in integer_registry
        ):
            print(
                " ",
                op,
            )

    print()
    print(
        "QDQ registry intersection:"
    )

    for op in graph_ops:
        if (
            op
            in qdq_registry
        ):
            print(
                " ",
                op,
            )

    print()
    print(
        "Calibration methods:",
        calibration_methods,
    )

    print(
        "Quant formats:",
        quant_formats,
    )

    print(
        "Quant types:",
        quant_types,
    )

    print()
    print(
        "Dynamic smoke status:",
        dynamic[
            "status"
        ],
    )

    if (
        dynamic[
            "status"
        ]
        == "PASS"
    ):
        print(
            "Dynamic artifact bytes:",
            dynamic[
                "artifact_bytes"
            ],
        )

        print(
            "Dynamic operators:",
            dynamic[
                "graph"
            ][
                "operator_counts"
            ],
        )

        print(
            "Dynamic max abs logit difference:",
            dynamic[
                "max_abs_logit_difference"
            ],
        )

        print(
            "Dynamic argmax differences:",
            dynamic[
                "argmax_decision_difference_count"
            ],
        )

        print(
            "Dynamic historical differences:",
            dynamic[
                "historical_decision_difference_count"
            ],
        )

    print()
    print(
        "FINAL_STATIC_CALIBRATION=DEFERRED"
    )

    print(
        "PHASE_2C_QUANTIZATION_SUPPORT_AUDIT=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
