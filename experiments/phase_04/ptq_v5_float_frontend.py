"""Phase-4G v5 mixed-precision static-PTQ implementation.

Single precision-boundary change from frozen v4:

    normalizer -> permute(0,2,1) -> conv_1

is grouped into one opaque FP32 front-end.

Quantization begins only after that front-end.

Unchanged downstream v4 precision placement:
- conv_2.0: quantized Conv1d
- conv_2.2: FP32 FloatPReLU
- fc.1: quantized Linear
- fc.2: FP32 FloatPReLU
- fc.4: FP32 Linear

The FloatPReLU implementation is inherited unchanged from v3/v4.
"""

from __future__ import annotations

import copy

import torch.nn as nn

from ptq_v4_float_classifier import (
    FloatPReLU,
    audit_float_prelu_weights,
    build_v4_prepare_custom_config,
    build_v4_qconfig_mapping,
    replace_prelus_with_float_wrappers,
)


FRONT_END_MODULE_NAME = "front_end"


class FloatFrontEnd(nn.Module):
    """Opaque FP32 normalizer + permutation + first CNN block."""

    def __init__(
        self,
        normalizer: nn.Module,
        conv_1: nn.Module,
    ):
        super().__init__()

        self.normalizer = copy.deepcopy(
            normalizer
        )

        self.conv_1 = copy.deepcopy(
            conv_1
        )

    def forward(self, x):
        x = self.normalizer(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        return x


class V5FloatFrontEndCNN(nn.Module):
    """Same CNN semantics with an explicit opaque FP32 front end."""

    def __init__(self, source: nn.Module):
        super().__init__()

        self.front_end = FloatFrontEnd(
            source.normalizer,
            source.conv_1,
        )

        self.conv_2 = copy.deepcopy(
            source.conv_2
        )

        self.fc = copy.deepcopy(
            source.fc
        )

    def forward(self, x):
        x = self.front_end(x)
        x = self.conv_2(x)
        x = self.fc(x)
        return x


def build_v5_candidate(
    fp32_model: nn.Module,
) -> V5FloatFrontEndCNN:
    """Create v5 without modifying the supplied FP32 checkpoint model."""

    source = copy.deepcopy(
        fp32_model
    ).eval()

    replace_prelus_with_float_wrappers(
        source
    )

    return V5FloatFrontEndCNN(
        source
    ).eval()


def build_v5_qconfig_mapping():
    """Frozen v4 qconfig plus opaque front-end exclusion."""

    mapping = build_v4_qconfig_mapping()

    mapping.set_module_name(
        FRONT_END_MODULE_NAME,
        None,
    )

    return mapping


def build_v5_prepare_custom_config():
    """Frozen v4 custom config plus opaque front-end tracing rule."""

    config = build_v4_prepare_custom_config()

    config.set_non_traceable_module_names(
        [FRONT_END_MODULE_NAME]
    )

    return config
