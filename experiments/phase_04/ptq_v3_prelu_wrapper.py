"""Phase-4G v3 mixed-precision static-PTQ implementation.

Frozen design:
- PyTorch FX x86 default static PTQ for eligible operators.
- Conv1d and Linear remain quantization targets.
- Every learned PReLU is replaced by a mathematically equivalent
  opaque FloatPReLU module.
- FloatPReLU is excluded from quantization and marked non-traceable,
  causing explicit dequantize/requantize boundaries around it.

This module contains no data selection or evaluation logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig


PRELU_MODULE_NAMES = (
    "conv_1.2",
    "conv_2.2",
    "fc.2",
)


class FloatPReLU(nn.Module):
    """Exact scalar-PReLU implementation retained in FP32."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()

        frozen = weight.detach().clone()

        if frozen.numel() != 1:
            raise ValueError(
                "Phase-4G v3 expects scalar PReLU weight; "
                f"received {tuple(frozen.shape)}"
            )

        self.weight = nn.Parameter(frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x >= 0,
            x,
            self.weight * x,
        )


def replace_prelus_with_float_wrappers(model: nn.Module) -> nn.Module:
    """Replace the three frozen CNN PReLUs without changing weights."""

    for full_name in PRELU_MODULE_NAMES:
        parent_name, child_name = full_name.rsplit(".", 1)

        parent = model.get_submodule(parent_name)
        source = model.get_submodule(full_name)

        if not isinstance(source, nn.PReLU):
            raise TypeError(
                f"{full_name} is not nn.PReLU: "
                f"{source.__class__.__module__}."
                f"{source.__class__.__name__}"
            )

        replacement = FloatPReLU(source.weight)

        if not torch.equal(
            replacement.weight.detach(),
            source.weight.detach(),
        ):
            raise RuntimeError(
                f"PReLU weight changed during replacement: {full_name}"
            )

        setattr(
            parent,
            child_name,
            replacement,
        )

    return model


def build_v3_qconfig_mapping():
    """x86 default PTQ with only FloatPReLU excluded."""

    mapping = get_default_qconfig_mapping("x86")
    mapping.set_object_type(FloatPReLU, None)

    return mapping


def build_v3_prepare_custom_config() -> PrepareCustomConfig:
    """Keep FloatPReLU opaque so FX inserts explicit Q/DQ boundaries."""

    config = PrepareCustomConfig()
    config.set_non_traceable_module_classes(
        [FloatPReLU]
    )

    return config


def audit_float_prelu_weights(
    fp32_model: nn.Module,
    v3_model: nn.Module,
) -> dict:
    """Return exact learned-slope preservation evidence."""

    result = {}

    for name in PRELU_MODULE_NAMES:
        source = fp32_model.get_submodule(name)
        target = v3_model.get_submodule(name)

        source_weight = source.weight.detach().cpu()
        target_weight = target.weight.detach().cpu()

        result[name] = {
            "source_weight": source_weight.tolist(),
            "target_weight": target_weight.tolist(),
            "exact_equal": bool(
                torch.equal(
                    source_weight,
                    target_weight,
                )
            ),
        }

    return result
