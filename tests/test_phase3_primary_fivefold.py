import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]


class TestPrimaryFiveFold(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(
            (
                ROOT
                / "configs"
                / "datasets"
                / "primary_300ms_fivefold_v1.json"
            ).read_text(
                encoding="utf-8"
            )
        )

    def test_protocol_identity(self):
        self.assertEqual(
            self.data["status"],
            "FROZEN",
        )

        self.assertEqual(
            self.data["window_ms"],
            300,
        )

        self.assertEqual(
            self.data["sampling_hz"],
            100,
        )

        self.assertEqual(
            self.data["overlap_percent"],
            50,
        )

        self.assertEqual(
            self.data["stride_ms"],
            150,
        )

    def test_subject_population(self):
        subjects = self.data[
            "subjects"
        ]

        self.assertEqual(
            len(subjects),
            61,
        )

        canonical = {
            item[
                "canonical_id"
            ]
            for item
            in subjects
        }

        self.assertEqual(
            len(canonical),
            61,
        )

        self.assertEqual(
            sum(
                item[
                    "dataset"
                ]
                == "UNIVR"
                for item
                in subjects
            ),
            29,
        )

        self.assertEqual(
            sum(
                item[
                    "dataset"
                ]
                == "KFALL"
                for item
                in subjects
            ),
            32,
        )

    def test_five_folds_disjoint(self):
        self.assertEqual(
            len(
                self.data[
                    "folds"
                ]
            ),
            5,
        )

        population = {
            item[
                "storage_id"
            ]
            for item
            in self.data[
                "subjects"
            ]
        }

        for fold in self.data[
            "folds"
        ]:
            train = set(
                fold[
                    "train"
                ][
                    "storage_ids"
                ]
            )

            validation = set(
                fold[
                    "validation"
                ][
                    "storage_ids"
                ]
            )

            test = set(
                fold[
                    "outer_test"
                ][
                    "storage_ids"
                ]
            )

            self.assertFalse(
                train
                & validation
            )

            self.assertFalse(
                train
                & test
            )

            self.assertFalse(
                validation
                & test
            )

            self.assertEqual(
                train
                | validation
                | test,
                population,
            )

    def test_outer_test_exact_once(self):
        counts = Counter()

        for fold in self.data[
            "folds"
        ]:
            counts.update(
                fold[
                    "outer_test"
                ][
                    "storage_ids"
                ]
            )

        self.assertEqual(
            len(counts),
            61,
        )

        self.assertTrue(
            all(
                value == 1
                for value
                in counts.values()
            )
        )

    def test_fold_sizes(self):
        sizes = [
            (
                fold[
                    "train"
                ][
                    "subject_count"
                ],
                fold[
                    "validation"
                ][
                    "subject_count"
                ],
                fold[
                    "outer_test"
                ][
                    "subject_count"
                ],
            )
            for fold
            in self.data[
                "folds"
            ]
        ]

        self.assertEqual(
            sizes,
            [
                (38, 10, 13),
                (39, 10, 12),
                (39, 10, 12),
                (39, 10, 12),
                (39, 10, 12),
            ],
        )

    def test_outer_test_never_for_calibration(self):
        policy = self.data[
            "partition_role_policy"
        ][
            "outer_test"
        ]

        self.assertFalse(
            policy[
                "int8_calibration_allowed"
            ]
        )

        self.assertFalse(
            policy[
                "fault_parameter_selection_allowed"
            ]
        )

        self.assertFalse(
            policy[
                "threshold_tuning_allowed"
            ]
        )

        self.assertTrue(
            policy[
                "evaluation_only"
            ]
        )

    def test_int8_source_is_training_only(self):
        policy = self.data[
            "partition_role_policy"
        ]

        self.assertTrue(
            policy[
                "training"
            ][
                "future_int8_calibration_source_allowed"
            ]
        )

        self.assertFalse(
            policy[
                "validation"
            ][
                "int8_calibration_source_allowed"
            ]
        )

        self.assertFalse(
            policy[
                "outer_test"
            ][
                "int8_calibration_allowed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
