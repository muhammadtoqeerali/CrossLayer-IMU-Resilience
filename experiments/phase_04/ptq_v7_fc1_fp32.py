"""Phase-4G v7 mixed-precision static-PTQ implementation.

Single precision-boundary change from frozen v6:

    fc.0 Flatten
      -> fc.1 Linear
      -> fc.2 FloatPReLU

remain continuously FP32.

The quantization boundary immediately after fc.2 / before fc.3 is
preserved.

Unchanged from v6:
- opaque FP32 front_end:
    normalizer -> permute(0,2,1) -> complete conv_1
- conv_2.0 quantized Conv1d
- conv_2.2 FP32 FloatPReLU
- conv_2.3 FP32 MaxPool
- conv_2.4 FP32 Dropout
- fc.0 FP32 Flatten
- fc.2 FP32 FloatPReLU
- post-fc.2 quantization before fc.3
- fc.4 FP32 Linear
"""

from __future__ import annotations

from ptq_v6_postconv2_floattail import (
    build_v6_candidate,
    build_v6_prepare_custom_config,
    build_v6_qconfig_mapping,
)


def build_v7_candidate(fp32_model):
    """v7 preserves the exact v6 model semantics before PTQ."""
    return build_v6_candidate(
        fp32_model
    )


def build_v7_qconfig_mapping():
    """Frozen v6 mapping with fc.1 retained in FP32."""

    mapping = build_v6_qconfig_mapping()

    mapping.set_module_name(
        "fc.1",
        None,
    )

    return mapping


def build_v7_prepare_custom_config():
    """v7 retains the exact v6 FX tracing contract."""
    return build_v6_prepare_custom_config()
