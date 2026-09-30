from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REFERENCE = (
    ROOT
    / "configs"
    / "baseline"
    / "frozen_fp32_reference_v1.json"
)


class TestPhase2FrozenReference(unittest.TestCase):

    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.data = json.loads(
            REFERENCE.read_text(
                encoding="utf-8"
            )
        )

    def test_reference_is_frozen(self) -> None:
        self.assertEqual(
            self.data["status"],
            "FROZEN",
        )

        self.assertEqual(
            self.data["freeze_scope"],
            "FP32_REFERENCE_BASELINE",
        )

    def test_baseline_identity(self) -> None:
        self.assertEqual(
            self.data["baseline_id"],
            "DATE2025_CNN_400MS_RECONSTRUCTED",
        )

        self.assertEqual(
            self.data[
                "historical_weights"
            ][
                "canonical_tensor_state_sha256"
            ],
            (
                "c98987476536320191f8875316cf8cae"
                "eb0b3f2edc51d65be7ac7d2eaea03124"
            ),
        )

    def test_primary_decision_semantics(self) -> None:
        decision = self.data[
            "primary_decision_semantics"
        ]

        self.assertEqual(
            decision["id"],
            "historical_streaming_0p9_strict",
        )

        self.assertEqual(
            decision["threshold"],
            0.9,
        )

        self.assertEqual(
            decision["comparison"],
            "strict_greater_than",
        )

    def test_fp32_onnx_identity(self) -> None:
        artifact = self.data[
            "fp32_onnx"
        ]

        self.assertEqual(
            artifact["artifact_sha256"],
            (
                "f3a55933bab5ed63225a1647a8b78aa"
                "909c4bb5a81405b06f235ba1b3bf6de1f"
            ),
        )

        self.assertEqual(
            artifact["input_shape"],
            [1, 40, 9],
        )

        self.assertEqual(
            artifact["output_shape"],
            [1, 2],
        )

        self.assertFalse(
            artifact["tracked_in_git"]
        )

    def test_fp32_parity_has_no_decision_differences(
        self,
    ) -> None:
        parity = self.data[
            "fp32_parity"
        ]

        self.assertEqual(
            parity[
                "elementwise_tolerance_violations"
            ],
            0,
        )

        self.assertEqual(
            parity[
                "argmax_decision_differences"
            ],
            0,
        )

        self.assertEqual(
            parity[
                "historical_decision_differences"
            ],
            0,
        )

    def test_int8_calibration_is_deferred(self) -> None:
        quant = self.data[
            "quantization"
        ]

        self.assertEqual(
            quant["primary_method"],
            "static_post_training_quantization",
        )

        self.assertEqual(
            quant["final_int8_status"],
            "DEFERRED_UNTIL_DATA_PROTOCOL_FREEZE",
        )

        self.assertFalse(
            quant[
                "qat_allowed_as_silent_substitution"
            ]
        )

    def test_baseline_is_unprotected(self) -> None:
        self.assertFalse(
            any(
                self.data[
                    "protection_boundary"
                ].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
