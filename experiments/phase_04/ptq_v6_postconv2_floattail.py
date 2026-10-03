"""Phase-4G v6 mixed-precision static-PTQ implementation.

Single precision-boundary change from frozen v5:

    conv_2.2 FloatPReLU
      -> conv_2.3 MaxPool
      -> conv_2.4 Dropout
      -> fc.0 Flatten

remain FP32, and quantization begins only after fc.0 immediately
before the quantized fc.1 layer.

Unchanged from v5:
- opaque FP32 front_end:
    normalizer -> permute(0,2,1) -> complete conv_1
- conv_2.0 quantized Conv1d
- conv_2.2 FP32 FloatPReLU
- fc.1 quantized Linear
- fc.2 FP32 FloatPReLU
- later v5 requantization after fc.2 remains
- fc.4 FP32 Linear
"""

from __future__ import annotations

from ptq_v5_float_frontend import (
    FloatFrontEnd,
    V5FloatFrontEndCNN,
    build_v5_candidate,
    build_v5_prepare_custom_config,
    build_v5_qconfig_mapping,
)


V6_FP32_POST_CONV2_MODULES = (
    "conv_2.3",
    "conv_2.4",
    "fc.0",
)


def build_v6_candidate(fp32_model):
    """v6 preserves the exact v5 model semantics before PTQ."""
    return build_v5_candidate(
        fp32_model
    )


def build_v6_qconfig_mapping():
    """Frozen v5 mapping with the second quantization boundary moved."""

    mapping = build_v5_qconfig_mapping()

    for module_name in V6_FP32_POST_CONV2_MODULES:
        mapping.set_module_name(
            module_name,
            None,
        )

    return mapping


def build_v6_prepare_custom_config():
    """v6 retains the exact v5 FX tracing contract."""
    return build_v5_prepare_custom_config()
