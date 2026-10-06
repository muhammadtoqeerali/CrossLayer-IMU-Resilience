"""Phase-5A P0 compute-fault representation contract.

This module contains only deterministic identity and representation
semantics.  It does not run model inference and it does not inject faults
into a model.

Evidence boundary:
- P0 software fault model only.
- No P1/P2/P3 claim.
- No MCU/physical-fault equivalence claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


INT8_BIT_POSITIONS = tuple(range(8))
FP32_BIT_POSITIONS = tuple(range(32))

FP32_MANTISSA_BITS = tuple(range(0, 23))
FP32_EXPONENT_BITS = tuple(range(23, 31))
FP32_SIGN_BITS = (31,)

FAULT_FAMILIES = (
    "int8_weight_single_bit_flip",
    "quantized_activation_single_bit_flip",
    "quantized_buffer_single_bit_flip",
    "fp32_activation_single_bit_flip",
    "fp32_buffer_single_bit_flip",
)

REPRESENTATION_CLASSES = (
    "int8_persistent_weight",
    "quantized_activation",
    "quantized_buffer",
    "fp32_activation",
    "fp32_buffer",
)

PERSISTENCE_MODES = (
    "transient_one_inference",
    "persistent_from_onset_until_trial_end",
)

INITIAL_MULTIPLICITY = 1

REQUIRED_FAULT_ID_FIELDS = (
    "protocol",
    "model_variant",
    "checkpoint_seed",
    "fold",
    "fault_family",
    "representation_class",
    "target_name",
    "target_role",
    "element_index",
    "bit_position",
    "inference_index",
    "persistence",
    "multiplicity",
    "replicate_index",
)


@dataclass(frozen=True)
class FaultIdentity:
    protocol: str
    model_variant: str
    checkpoint_seed: int
    fold: int
    fault_family: str
    representation_class: str
    target_name: str
    target_role: str
    element_index: int
    bit_position: int
    inference_index: int
    persistence: str
    multiplicity: int
    replicate_index: int

    def canonical_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in REQUIRED_FAULT_ID_FIELDS
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def fault_id(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()


def fp32_bit_class(bit_position: int) -> str:
    if bit_position in FP32_MANTISSA_BITS:
        return "mantissa"
    if bit_position in FP32_EXPONENT_BITS:
        return "exponent"
    if bit_position in FP32_SIGN_BITS:
        return "sign"
    raise ValueError(
        f"invalid FP32 bit position: {bit_position}"
    )


def int8_bit_class(bit_position: int) -> str:
    if bit_position not in INT8_BIT_POSITIONS:
        raise ValueError(
            f"invalid INT8 bit position: {bit_position}"
        )

    if bit_position == 7:
        return "msb_sign"

    return "payload_bit"


def deterministic_index(
    *,
    namespace: str,
    upper_bound: int,
    fields: Mapping[str, Any],
) -> int:
    """Map canonical identity fields deterministically into [0, upper_bound).

    This defines replayable target sampling without relying on process-global
    RNG state.
    """

    if upper_bound <= 0:
        raise ValueError(
            "upper_bound must be positive"
        )

    payload = {
        "namespace": namespace,
        "fields": dict(fields),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % upper_bound


def validate_fault_identity(
    identity: FaultIdentity,
) -> None:
    if identity.fault_family not in FAULT_FAMILIES:
        raise ValueError(
            f"unknown fault family: {identity.fault_family}"
        )

    if (
        identity.representation_class
        not in REPRESENTATION_CLASSES
    ):
        raise ValueError(
            "unknown representation class: "
            f"{identity.representation_class}"
        )

    if identity.persistence not in PERSISTENCE_MODES:
        raise ValueError(
            f"unknown persistence: {identity.persistence}"
        )

    if identity.multiplicity != INITIAL_MULTIPLICITY:
        raise ValueError(
            "Phase-5A v1 freezes multiplicity=1 only"
        )

    if identity.element_index < 0:
        raise ValueError(
            "element_index must be non-negative"
        )

    if identity.inference_index < 0:
        raise ValueError(
            "inference_index must be non-negative"
        )

    if identity.replicate_index < 0:
        raise ValueError(
            "replicate_index must be non-negative"
        )

    if identity.representation_class.startswith(
        "int8_"
    ) or identity.representation_class.startswith(
        "quantized_"
    ):
        if identity.bit_position not in INT8_BIT_POSITIONS:
            raise ValueError(
                "quantized/int8 representation requires bit 0..7"
            )

    if identity.representation_class.startswith(
        "fp32_"
    ):
        if identity.bit_position not in FP32_BIT_POSITIONS:
            raise ValueError(
                "FP32 representation requires bit 0..31"
            )


def protocol_capabilities() -> dict[str, Any]:
    return {
        "evidence_tier": "P0",
        "model_forward_pass_required": False,
        "dataset_required": False,
        "outer_test_required": False,
        "onfield_required": False,
        "physical_fault_equivalence": False,
        "mcu_fault_equivalence": False,
        "fault_families": list(
            FAULT_FAMILIES
        ),
        "representation_classes": list(
            REPRESENTATION_CLASSES
        ),
        "persistence_modes": list(
            PERSISTENCE_MODES
        ),
        "initial_multiplicity": INITIAL_MULTIPLICITY,
    }
