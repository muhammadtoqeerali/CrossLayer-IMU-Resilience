from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

EXPECTED_STATE_SHA = (
    "c98987476536320191f8875316cf8cae"
    "eb0b3f2edc51d65be7ac7d2eaea03124"
)

OUT = (
    ROOT
    / "manifests"
    / "phase_2a_execution_environment_audit_v1.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def command_output(
    args: list[str],
) -> str | None:
    try:
        proc = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=20,
        )

        if proc.returncode != 0:
            return None

        return proc.stdout.strip()
    except Exception:
        return None


def package_status(
    name: str,
) -> dict:
    spec = importlib.util.find_spec(name)

    if spec is None:
        return {
            "available": False,
            "version": None,
        }

    try:
        module = __import__(name)

        version = getattr(
            module,
            "__version__",
            None,
        )
    except Exception:
        version = None

    return {
        "available": True,
        "version": (
            str(version)
            if version is not None
            else None
        ),
    }


def deterministic_vectors() -> torch.Tensor:
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

                window[0, t, c] = np.float32(
                    numerator / 64.0
                )

        vectors.append(window)

    return torch.from_numpy(
        np.concatenate(
            vectors,
            axis=0,
        )
    )


def tensor_storage_summary(
    model: torch.nn.Module,
) -> dict:
    parameter_elements = 0
    parameter_bytes = 0

    buffer_elements = 0
    buffer_bytes = 0

    dtype_parameters = {}
    dtype_buffers = {}

    for parameter in model.parameters():
        n = int(parameter.numel())
        b = n * int(parameter.element_size())

        parameter_elements += n
        parameter_bytes += b

        key = str(parameter.dtype)

        dtype_parameters[key] = (
            dtype_parameters.get(key, 0)
            + n
        )

    for buffer in model.buffers():
        n = int(buffer.numel())
        b = n * int(buffer.element_size())

        buffer_elements += n
        buffer_bytes += b

        key = str(buffer.dtype)

        dtype_buffers[key] = (
            dtype_buffers.get(key, 0)
            + n
        )

    return {
        "parameter_elements": parameter_elements,
        "parameter_bytes": parameter_bytes,
        "parameter_mib": (
            parameter_bytes / (1024 ** 2)
        ),
        "buffer_elements": buffer_elements,
        "buffer_bytes": buffer_bytes,
        "parameter_dtype_elements": dtype_parameters,
        "buffer_dtype_elements": dtype_buffers,
    }


def module_inventory(
    model: torch.nn.Module,
) -> dict:
    counts = {}

    for module in model.modules():
        name = module.__class__.__name__

        counts[name] = (
            counts.get(name, 0)
            + 1
        )

    return dict(sorted(counts.items()))


def estimate_conv_linear_macs(
    model: torch.nn.Module,
) -> dict:
    """
    Deterministic structural MAC estimate for one [1,40,9] input.

    Counts multiply-accumulate pairs for Conv1d and Linear only.
    This excludes normalization, activation, pooling, softmax and preprocessing.
    """
    records = []
    hooks = []

    def conv_hook(
        module,
        inputs,
        output,
    ):
        batch = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_length = int(output.shape[2])

        kernel = int(module.kernel_size[0])

        macs_per_output = (
            int(module.in_channels)
            // int(module.groups)
            * kernel
        )

        macs = (
            batch
            * out_channels
            * out_length
            * macs_per_output
        )

        records.append(
            {
                "type": "Conv1d",
                "macs": int(macs),
                "output_shape": [
                    int(v)
                    for v in output.shape
                ],
            }
        )

    def linear_hook(
        module,
        inputs,
        output,
    ):
        leading = int(
            output.numel()
            // module.out_features
        )

        macs = (
            leading
            * int(module.in_features)
            * int(module.out_features)
        )

        records.append(
            {
                "type": "Linear",
                "macs": int(macs),
                "output_shape": [
                    int(v)
                    for v in output.shape
                ],
            }
        )

    for module in model.modules():
        if isinstance(
            module,
            torch.nn.Conv1d,
        ):
            hooks.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

        if isinstance(
            module,
            torch.nn.Linear,
        ):
            hooks.append(
                module.register_forward_hook(
                    linear_hook
                )
            )

    x = torch.zeros(
        1,
        40,
        9,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        model(x)

    for hook in hooks:
        hook.remove()

    total = sum(
        record["macs"]
        for record in records
    )

    return {
        "definition": (
            "Conv1d and Linear multiply-accumulate pairs only"
        ),
        "records": records,
        "total_macs": int(total),
    }


def benchmark_host(
    model: torch.nn.Module,
) -> dict:
    """
    Development-workstation reference only.

    These are NOT MCU measurements and must never be reported as embedded
    latency evidence.
    """
    torch.set_num_threads(1)

    x = torch.zeros(
        1,
        40,
        9,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for _ in range(100):
            model(x)

        timings = []

        for _ in range(1000):
            start = time.perf_counter_ns()

            model(x)

            stop = time.perf_counter_ns()

            timings.append(
                (stop - start)
                / 1_000_000.0
            )

    ordered = sorted(timings)

    def percentile(p: float) -> float:
        index = int(
            round(
                (len(ordered) - 1)
                * p
            )
        )

        return float(
            ordered[index]
        )

    return {
        "scope": (
            "development_workstation_cpu_reference_only"
        ),
        "torch_threads": 1,
        "warmup_iterations": 100,
        "timed_iterations": 1000,
        "mean_ms": float(
            statistics.mean(timings)
        ),
        "median_ms": float(
            statistics.median(timings)
        ),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
    }


def main() -> int:
    if not STATE_ARTIFACT.is_file():
        raise RuntimeError(
            f"State artifact missing: {STATE_ARTIFACT}"
        )

    model = load_date2025_from_state_artifact(
        STATE_ARTIFACT,
        expected_state_sha256=EXPECTED_STATE_SHA,
    )

    state_sha = canonical_state_sha256(
        model.state_dict()
    )

    if state_sha != EXPECTED_STATE_SHA:
        raise RuntimeError(
            "Canonical model state changed"
        )

    model.eval()

    vectors = deterministic_vectors()

    with torch.inference_mode():
        logits, features = (
            model.forward_with_features(
                vectors
            )
        )

    argmax_decision = argmax_from_logits(
        logits
    )

    historical_decision = decision_from_logits(
        logits
    )

    decision_difference_count = int(
        (
            argmax_decision
            != historical_decision
        )
        .sum()
        .item()
    )

    package_names = (
        "onnx",
        "onnxruntime",
        "onnxruntime.quantization",
        "torchvision",
    )

    packages = {
        name: package_status(name)
        for name in package_names
    }

    try:
        quantized_engines = list(
            torch.backends.quantized.supported_engines
        )
    except Exception:
        quantized_engines = []

    try:
        current_quantized_engine = (
            torch.backends.quantized.engine
        )
    except Exception:
        current_quantized_engine = None

    try:
        importlib.import_module(
            "torch.ao.quantization"
        )

        torch_ao_quantization = True
    except Exception:
        torch_ao_quantization = False

    try:
        importlib.import_module(
            "torch.ao.quantization.quantize_fx"
        )

        torch_fx_quantization = True
    except Exception:
        torch_fx_quantization = False

    storage = tensor_storage_summary(
        model
    )

    modules = module_inventory(
        model
    )

    macs = estimate_conv_linear_macs(
        model
    )

    host_benchmark = benchmark_host(
        model
    )

    result = {
        "schema": (
            "crosslayer_phase2a_"
            "execution_environment_audit_v1"
        ),
        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "status": "PASS",
        "baseline": {
            "id": (
                "DATE2025_CNN_400MS_RECONSTRUCTED"
            ),
            "freeze_status": "NOT_FROZEN",
            "canonical_state_sha256": state_sha,
            "state_artifact_file_sha256": (
                sha256(STATE_ARTIFACT)
            ),
            "parameter_count": (
                sum(
                    int(p.numel())
                    for p in model.parameters()
                )
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "cuda_version": torch.version.cuda,
            "cpu_count": os.cpu_count(),
            "uname": command_output(
                ["uname", "-a"]
            ),
        },
        "packages": packages,
        "quantization_capabilities": {
            "torch_ao_quantization": (
                torch_ao_quantization
            ),
            "torch_fx_quantization": (
                torch_fx_quantization
            ),
            "supported_engines": (
                quantized_engines
            ),
            "current_engine": (
                current_quantized_engine
            ),
        },
        "storage": storage,
        "module_inventory": modules,
        "structural_compute": macs,
        "deterministic_reference": {
            "vector_count": int(
                vectors.shape[0]
            ),
            "logits_sha256": (
                hashlib.sha256(
                    logits
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            ),
            "features_sha256": (
                hashlib.sha256(
                    features
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            ),
            "argmax_prediction_sha256": (
                hashlib.sha256(
                    argmax_decision
                    .detach()
                    .cpu()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            ),
            "historical_prediction_sha256": (
                hashlib.sha256(
                    historical_decision
                    .detach()
                    .cpu()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            ),
            "argmax_vs_historical_difference_count": (
                decision_difference_count
            ),
        },
        "host_reference_latency": host_benchmark,
        "scientific_limits": {
            "host_latency_is_embedded_evidence": False,
            "dataset_accuracy_frozen": False,
            "quantized_model_frozen": False,
            "decision_semantics_frozen": False,
        },
    }

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Baseline:", result["baseline"]["id"])
    print(
        "Canonical state:",
        result["baseline"][
            "canonical_state_sha256"
        ],
    )
    print(
        "Parameters:",
        result["baseline"][
            "parameter_count"
        ],
    )
    print(
        "FP32 parameter bytes:",
        storage["parameter_bytes"],
    )
    print(
        "Structural Conv/Linear MACs:",
        macs["total_macs"],
    )
    print(
        "Argmax/historical differences on 68 vectors:",
        decision_difference_count,
    )
    print(
        "ONNX available:",
        packages["onnx"]["available"],
    )
    print(
        "ONNX Runtime available:",
        packages["onnxruntime"]["available"],
    )
    print(
        "Torch AO quantization:",
        torch_ao_quantization,
    )
    print(
        "FX quantization:",
        torch_fx_quantization,
    )
    print(
        "Supported quantized engines:",
        quantized_engines,
    )
    print(
        "Current quantized engine:",
        current_quantized_engine,
    )
    print(
        "Host mean latency ms:",
        host_benchmark["mean_ms"],
    )
    print(
        "Host p95 latency ms:",
        host_benchmark["p95_ms"],
    )
    print()
    print("PHASE_2A_ENVIRONMENT_AUDIT=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
