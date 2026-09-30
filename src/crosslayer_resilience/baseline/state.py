from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

from .date2025_cnn400 import Date2025CNN400


def canonical_state_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """
    Hash tensor state independently of torch.save ZIP/container metadata.

    The digest commits to:
    - key name
    - dtype
    - shape
    - raw tensor bytes

    Keys are sorted lexicographically.
    """
    digest = hashlib.sha256()

    for key in sorted(state_dict):
        tensor = state_dict[key]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"State value {key!r} is not a torch.Tensor"
            )

        tensor = (
            tensor
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(key.encode("utf-8"))
        digest.update(b"\0")

        digest.update(
            str(tensor.dtype).encode("ascii")
        )
        digest.update(b"\0")

        shape_text = ",".join(
            str(int(value))
            for value in tensor.shape
        )

        digest.update(
            shape_text.encode("ascii")
        )
        digest.update(b"\0")

        # Reshape first so scalar state entries such as BatchNorm's
        # num_batches_tracked (0-D int64) can also be viewed byte-wise.
        #
        # tensor is already detached, CPU-resident and contiguous here.
        byte_view = tensor.reshape(-1).view(torch.uint8)

        digest.update(
            byte_view.numpy().tobytes()
        )

        digest.update(b"\0")

    return digest.hexdigest()


def load_state_dict_artifact(
    path: str | Path,
) -> Mapping[str, torch.Tensor]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older torch versions.
        state = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(state, Mapping):
        raise RuntimeError(
            "Normalized artifact is not a state-dict mapping"
        )

    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError(
                "State-dict key is not a string"
            )

        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"State-dict value {key!r} is not a tensor"
            )

    return state


def load_date2025_from_state_artifact(
    path: str | Path,
    expected_state_sha256: str | None = None,
) -> Date2025CNN400:
    state = load_state_dict_artifact(path)

    observed_state_sha256 = canonical_state_sha256(
        state
    )

    if (
        expected_state_sha256 is not None
        and observed_state_sha256
        != expected_state_sha256
    ):
        raise RuntimeError(
            "Normalized state-dict canonical SHA-256 mismatch"
        )

    model = Date2025CNN400()

    model.load_state_dict(
        state,
        strict=True,
    )

    parameter_count = model.parameter_count()

    if (
        parameter_count
        != Date2025CNN400.EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Protected baseline parameter count changed: "
            f"{parameter_count}"
        )

    model.eval()

    return model
