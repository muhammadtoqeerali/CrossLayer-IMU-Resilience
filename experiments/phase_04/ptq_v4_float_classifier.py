"""Phase-4G v4 mixed-precision static-PTQ implementation.

The v4 implementation deliberately reuses the hash-frozen v3
FloatPReLU implementation and changes exactly one qconfig decision:

    fc.4 -> qconfig None

Therefore:
- conv_1.0 and conv_2.0 remain static-quantized Conv1d;
- fc.1 remains static-quantized Linear;
- all three FloatPReLU wrappers remain FP32 exactly as v3;
- final classifier fc.4 remains FP32;
- the final model output is FP32.

No calibration selection or task evaluation logic exists here.
"""

from __future__ import annotations

from ptq_v3_prelu_wrapper import (
    FloatPReLU,
    PRELU_MODULE_NAMES,
    audit_float_prelu_weights,
    build_v3_prepare_custom_config,
    build_v3_qconfig_mapping,
    replace_prelus_with_float_wrappers,
)


FINAL_CLASSIFIER_MODULE_NAME = "fc.4"


def build_v4_qconfig_mapping():
    """Frozen v3 mapping plus one module-name exclusion: fc.4."""

    mapping = build_v3_qconfig_mapping()

    mapping.set_module_name(
        FINAL_CLASSIFIER_MODULE_NAME,
        None,
    )

    return mapping


def build_v4_prepare_custom_config():
    """Identical FloatPReLU custom configuration to frozen v3."""

    return build_v3_prepare_custom_config()
