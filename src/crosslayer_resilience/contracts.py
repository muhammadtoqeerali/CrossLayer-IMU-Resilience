from __future__ import annotations

from enum import Enum


class FaultDomain(str, Enum):
    CLEAN = "clean"
    SENSOR = "sensor"
    COMPUTE = "compute"
    COMBINED = "combined"


class EvidenceTier(str, Enum):
    P0 = "offline_synthetic_or_software_injected"
    P1 = "firmware_injected"
    P2 = "hardware_in_loop_or_interface"
    P3 = "physical_or_trusted_status"


MANDATORY_FAULT_DOMAINS = tuple(FaultDomain)
