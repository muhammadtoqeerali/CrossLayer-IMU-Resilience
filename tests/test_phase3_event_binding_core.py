import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "manifests/"
      "phase_3m_label_time_repair_v2.json"
)


class TestPhase3EventBindingCore(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(
            DATA.read_text(
                encoding="utf-8"
            )
        )

    def test_status(self):
        self.assertEqual(
            self.data["status"],
            "PASS",
        )

    def test_event_population(self):
        binding = self.data[
            "canonical_binding"
        ]

        self.assertEqual(
            binding[
                "annotated_event_count"
            ],
            2919,
        )

        self.assertEqual(
            binding[
                "dataset_counts"
            ],
            {
                "KFALL": 2346,
                "UNIVR": 573,
            },
        )

        self.assertEqual(
            binding[
                "processed_trial_missing_count"
            ],
            0,
        )

    def test_single_activity_only_annotated_event(self):
        self.assertEqual(
            self.data[
                "historical_label_semantics"
            ][
                "annotated_events_with_no_falling_window"
            ],
            [
                "KFALL_106_T27_R05"
            ],
        )

    def test_deadline_is_separate_from_historical_labels(self):
        self.assertTrue(
            self.data[
                "scientific_interpretation"
            ][
                "safety_deadline_is_separate_evaluation_layer"
            ]
        )

        self.assertTrue(
            self.data[
                "scientific_interpretation"
            ][
                "historical_training_labels_retained_unchanged"
            ]
        )

    def test_no_model_execution(self):
        boundary = self.data[
            "scientific_boundary"
        ]

        self.assertFalse(
            boundary[
                "model_trained"
            ]
        )

        self.assertFalse(
            boundary[
                "predictions_executed"
            ]
        )

        self.assertFalse(
            boundary[
                "faults_injected"
            ]
        )


if __name__ == "__main__":
    unittest.main()
