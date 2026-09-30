from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


try:
    import torch
except Exception:
    torch = None


@unittest.skipIf(
    torch is None,
    "PyTorch unavailable in this Python interpreter",
)
class TestDate2025Candidate(unittest.TestCase):

    def setUp(self) -> None:
        from crosslayer_resilience.baseline import (
            Date2025CNN400,
        )

        self.model = Date2025CNN400()
        self.model.eval()

    def test_parameter_count(self) -> None:
        self.assertEqual(
            self.model.parameter_count(),
            63173,
        )

    def test_input_output_shape(self) -> None:
        x = torch.zeros(
            3,
            40,
            9,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            logits = self.model(x)

        self.assertEqual(
            tuple(logits.shape),
            (3, 2),
        )

    def test_feature_shape(self) -> None:
        x = torch.zeros(
            2,
            40,
            9,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            logits, features = (
                self.model.forward_with_features(x)
            )

        self.assertEqual(
            tuple(logits.shape),
            (2, 2),
        )

        self.assertEqual(
            tuple(features.shape),
            (2, 256),
        )

    def test_wrong_window_rejected(self) -> None:
        x = torch.zeros(
            1,
            39,
            9,
        )

        with self.assertRaises(ValueError):
            self.model(x)

    def test_wrong_channel_count_rejected(self) -> None:
        x = torch.zeros(
            1,
            40,
            6,
        )

        with self.assertRaises(ValueError):
            self.model(x)

    def test_normalizer_contract(self) -> None:
        from crosslayer_resilience.baseline import (
            HistoricalIMUNormalizer,
        )

        x = torch.zeros(
            1,
            40,
            9,
            dtype=torch.float32,
        )

        x[:, :, 0:3] = 4000.0
        x[:, :, 3:6] = 1800000.0
        x[:, :, 6:9] = 999.0

        y = HistoricalIMUNormalizer()(x)

        self.assertEqual(
            tuple(y.shape),
            (1, 40, 6),
        )

        self.assertTrue(
            torch.equal(
                y[:, :, 0:3],
                torch.ones_like(
                    y[:, :, 0:3]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                y[:, :, 3:6],
                torch.ones_like(
                    y[:, :, 3:6]
                ),
            )
        )

    def test_canonical_state_hash_handles_scalar_long(self) -> None:
        from crosslayer_resilience.baseline.state import (
            canonical_state_sha256,
        )

        state = {
            "weight": torch.tensor(
                [1.0, 2.0],
                dtype=torch.float32,
            ),
            "num_batches_tracked": torch.tensor(
                7,
                dtype=torch.long,
            ),
        }

        first = canonical_state_sha256(state)
        second = canonical_state_sha256(state)

        self.assertEqual(
            first,
            second,
        )

        self.assertEqual(
            len(first),
            64,
        )


    def test_historical_decision_is_strict(self) -> None:
        from crosslayer_resilience.baseline import (
            decision_from_probabilities,
        )

        probabilities = torch.tensor(
            [
                [0.10, 0.90],
                [0.09, 0.91],
                [0.95, 0.05],
            ],
            dtype=torch.float32,
        )

        result = decision_from_probabilities(
            probabilities
        )

        self.assertEqual(
            result.tolist(),
            [0, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
