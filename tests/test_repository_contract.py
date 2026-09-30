from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crosslayer_resilience.contracts import (  # noqa: E402
    EvidenceTier,
    FaultDomain,
)


class TestFaultDomains(unittest.TestCase):

    def test_exact_domains(self) -> None:
        self.assertEqual(
            {item.value for item in FaultDomain},
            {
                "clean",
                "sensor",
                "compute",
                "combined",
            },
        )

    def test_combined_is_distinct(self) -> None:
        self.assertNotEqual(
            FaultDomain.COMBINED,
            FaultDomain.SENSOR,
        )
        self.assertNotEqual(
            FaultDomain.COMBINED,
            FaultDomain.COMPUTE,
        )


class TestEvidenceTiers(unittest.TestCase):

    def test_all_tiers_exist(self) -> None:
        self.assertEqual(
            {item.name for item in EvidenceTier},
            {"P0", "P1", "P2", "P3"},
        )


class TestScientificContracts(unittest.TestCase):

    def test_four_regimes_documented(self) -> None:
        text = (
            ROOT / "docs/EXPERIMENT_CONTRACT.md"
        ).read_text(encoding="utf-8")

        for regime in ("C0", "CS", "CC", "CSC"):
            self.assertIn(regime, text)

    def test_fp32_reference_baseline_frozen(self) -> None:
        text = (
            ROOT / "configs/project.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "status: fp32_reference_frozen",
            text,
        )

        self.assertIn(
            "primary_candidate: DATE2025_CNN_400MS_RECONSTRUCTED",
            text,
        )

        self.assertIn(
            "decision_primary: historical_streaming_0p9_strict",
            text,
        )

        self.assertIn(
            "final_int8_status: deferred_until_data_protocol_freeze",
            text,
        )

    def test_hardware_not_frozen(self) -> None:
        text = (
            ROOT / "configs/project.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "target_status: candidate_not_frozen",
            text,
        )

    def test_split_before_fault_injection(self) -> None:
        text = (
            ROOT / "configs/project.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "split_before_fault_injection: true",
            text,
        )

    def test_final_test_tuning_disabled(self) -> None:
        text = (
            ROOT / "configs/project.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "test_data_for_tuning: false",
            text,
        )


if __name__ == "__main__":
    unittest.main()
