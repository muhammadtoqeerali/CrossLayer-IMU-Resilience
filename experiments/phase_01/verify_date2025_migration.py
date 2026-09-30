from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
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
    Date2025CNN400,
    decision_from_logits,
)
from crosslayer_resilience.baseline.state import (  # noqa: E402
    canonical_state_sha256,
    load_date2025_from_state_artifact,
)


REFERENCE_REPO = (
    Path.home()
    / "toqeer"
    / "RC-RGD-IMU_publish"
)

REFERENCE_LOADER = (
    REFERENCE_REPO
    / "experiments"
    / "03_ood"
    / "evaluate_ood_calibration_v1b.py"
)

CHECKPOINT = Path(
    "/mnt/hdd16T/protechto/checkpoints/CNN/400ms/"
    "2025-02-25_12_24_47/best-checkpoint.ckpt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "ee7c0079bfb8555bff45c3077cc24eaa"
    "4373c57729045d92a831a1d7a3ea9bb1"
)

EXPECTED_PARAMETER_COUNT = 63173

ARTIFACT = (
    ROOT
    / "artifacts"
    / "baseline"
    / "date2025_cnn400_state_dict_v1.pt"
)

RESULT = (
    ROOT
    / "manifests"
    / "phase_1d_baseline_migration_validation_v1.json"
)


def file_sha256(
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


def load_reference_model():
    if not REFERENCE_LOADER.is_file():
        raise RuntimeError(
            f"Reference loader missing: {REFERENCE_LOADER}"
        )

    spec = importlib.util.spec_from_file_location(
        "crosslayer_reference_rc_rgd_loader",
        REFERENCE_LOADER,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not create reference loader module"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    model = module.load_protected_model()

    model.cpu()
    model.eval()

    return model


def parity_vectors() -> torch.Tensor:
    vectors: list[np.ndarray] = []

    vectors.extend(
        [
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
    )

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

        vectors.append(window)

    if len(vectors) != 68:
        raise RuntimeError(
            "Parity-vector construction changed"
        )

    array = np.concatenate(
        vectors,
        axis=0,
    )

    return torch.from_numpy(array)


def main() -> int:
    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Checkpoint missing: {CHECKPOINT}"
        )

    checkpoint_sha = file_sha256(
        CHECKPOINT
    )

    if (
        checkpoint_sha
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Historical checkpoint SHA-256 mismatch"
        )

    print(
        "[PASS] Historical checkpoint SHA-256"
    )

    reference = load_reference_model()

    reference_parameter_count = sum(
        int(parameter.numel())
        for parameter in reference.parameters()
    )

    if (
        reference_parameter_count
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Reference parameter count changed"
        )

    print(
        "[PASS] Trusted historical loader"
    )

    migrated = Date2025CNN400()

    missing, unexpected = (
        migrated.load_state_dict(
            reference.state_dict(),
            strict=True,
        )
    )

    if missing or unexpected:
        raise RuntimeError(
            "Unexpected state-dict mismatch"
        )

    migrated.eval()

    if (
        migrated.parameter_count()
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Migrated parameter count changed"
        )

    print(
        "[PASS] Migrated architecture strict state load"
    )

    state_digest = canonical_state_sha256(
        migrated.state_dict()
    )

    x = parity_vectors()

    with torch.inference_mode():
        reference_logits, reference_features = (
            reference.forward_with_features(x)
        )

        migrated_logits, migrated_features = (
            migrated.forward_with_features(x)
        )

    logit_abs = (
        reference_logits
        - migrated_logits
    ).abs()

    feature_abs = (
        reference_features
        - migrated_features
    ).abs()

    max_logit_difference = float(
        logit_abs.max().item()
    )

    max_feature_difference = float(
        feature_abs.max().item()
    )

    predictions_equal = bool(
        torch.equal(
            reference_logits.argmax(dim=1),
            migrated_logits.argmax(dim=1),
        )
    )

    if max_logit_difference != 0.0:
        raise RuntimeError(
            "Migrated logits are not bit-identical "
            f"to reference: {max_logit_difference}"
        )

    if max_feature_difference != 0.0:
        raise RuntimeError(
            "Migrated features are not bit-identical "
            f"to reference: {max_feature_difference}"
        )

    if not predictions_equal:
        raise RuntimeError(
            "Migrated predictions differ"
        )

    print(
        "[PASS] 68-vector task-logit parity"
    )

    print(
        "[PASS] 68-vector penultimate-feature parity"
    )

    # Compare the explicit historical decision implementation.
    from imu_reliability.baseline.historical_decision import (
        decision_from_logits as reference_decision,
    )

    reference_streaming = reference_decision(
        reference_logits
    )

    migrated_streaming = decision_from_logits(
        migrated_logits
    )

    historical_decision_equal = bool(
        torch.equal(
            reference_streaming,
            migrated_streaming,
        )
    )

    if not historical_decision_equal:
        raise RuntimeError(
            "Historical decision semantics differ"
        )

    print(
        "[PASS] Historical streaming-decision parity"
    )

    # Create clean state-dict-only artifact.
    ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        migrated.state_dict(),
        ARTIFACT,
    )

    artifact_sha256 = file_sha256(
        ARTIFACT
    )

    reloaded = load_date2025_from_state_artifact(
        ARTIFACT,
        expected_state_sha256=state_digest,
    )

    with torch.inference_mode():
        reloaded_logits, reloaded_features = (
            reloaded.forward_with_features(x)
        )

    reloaded_logit_difference = float(
        (
            migrated_logits
            - reloaded_logits
        )
        .abs()
        .max()
        .item()
    )

    reloaded_feature_difference = float(
        (
            migrated_features
            - reloaded_features
        )
        .abs()
        .max()
        .item()
    )

    if reloaded_logit_difference != 0.0:
        raise RuntimeError(
            "Normalized state artifact changed logits"
        )

    if reloaded_feature_difference != 0.0:
        raise RuntimeError(
            "Normalized state artifact changed features"
        )

    print(
        "[PASS] Normalized state artifact reload parity"
    )

    result = {
        "schema": (
            "crosslayer_phase1d_"
            "baseline_migration_validation_v1"
        ),
        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "status": "PASS",
        "baseline_id": (
            "DATE2025_CNN_400MS_RECONSTRUCTED"
        ),
        "baseline_freeze_status": "NOT_FROZEN",
        "source": {
            "repository": (
                "muhammadtoqeerali/RC-RGD-IMU"
            ),
            "clean_commit": (
                "e1db7880e17a5632bc4ce238125923f986e85519"
            ),
            "reference_loader": str(
                REFERENCE_LOADER
            ),
        },
        "historical_checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": checkpoint_sha,
            "bytes": CHECKPOINT.stat().st_size,
        },
        "model": {
            "parameter_count": (
                migrated.parameter_count()
            ),
            "input_shape": [1, 40, 9],
            "output_shape": [1, 2],
            "penultimate_feature_dim": 256,
            "canonical_state_sha256": (
                state_digest
            ),
        },
        "parity": {
            "vector_count": int(
                x.shape[0]
            ),
            "max_abs_logit_difference": (
                max_logit_difference
            ),
            "max_abs_feature_difference": (
                max_feature_difference
            ),
            "argmax_predictions_equal": (
                predictions_equal
            ),
            "historical_decision_equal": (
                historical_decision_equal
            ),
            "reloaded_max_abs_logit_difference": (
                reloaded_logit_difference
            ),
            "reloaded_max_abs_feature_difference": (
                reloaded_feature_difference
            ),
        },
        "normalized_state_artifact": {
            "path": str(
                ARTIFACT.relative_to(ROOT)
            ),
            "git_tracked": False,
            "file_sha256": artifact_sha256,
            "canonical_state_sha256": (
                state_digest
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "scientific_boundary": {
            "contains_sensor_integrity": False,
            "contains_compute_integrity": False,
            "contains_ood": False,
            "contains_runtime_supervisor": False,
            "contains_recovery": False,
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

    print()
    print(
        "CANONICAL_STATE_SHA256="
        + state_digest
    )
    print(
        "NORMALIZED_ARTIFACT_SHA256="
        + artifact_sha256
    )
    print(
        "MAX_LOGIT_DIFFERENCE="
        + str(max_logit_difference)
    )
    print(
        "MAX_FEATURE_DIFFERENCE="
        + str(max_feature_difference)
    )
    print(
        "HISTORICAL_DECISION_PARITY="
        + str(historical_decision_equal)
    )
    print()
    print(
        "PHASE_1D_MIGRATION_PARITY=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
