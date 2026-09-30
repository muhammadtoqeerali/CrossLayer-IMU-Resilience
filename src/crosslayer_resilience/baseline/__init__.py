"""
Protected task baselines for CrossLayer-IMU-Resilience.

No sensor-integrity, compute-integrity, OOD, recovery, or runtime-supervisor
logic belongs in this package's protected task baseline.
"""

from .date2025_cnn400 import (
    Date2025CNN400,
    HistoricalIMUNormalizer,
)
from .decision import (
    ACTIVITY_CLASS,
    FALLING_CLASS,
    HISTORICAL_PREDICTION_BIAS,
    argmax_from_logits,
    decision_from_logits,
    decision_from_probabilities,
)

__all__ = [
    "Date2025CNN400",
    "HistoricalIMUNormalizer",
    "ACTIVITY_CLASS",
    "FALLING_CLASS",
    "HISTORICAL_PREDICTION_BIAS",
    "argmax_from_logits",
    "decision_from_logits",
    "decision_from_probabilities",
]
