import json
import unittest
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]


class TestPhase3OnFieldPolicy(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(
            (
                ROOT
                / "configs"
                / "datasets"
                / "onfield_role_candidate_v1.json"
            ).read_text(
                encoding="utf-8"
            )
        )

    def test_ten_retained_cases(self):
        onfield = self.data[
            "onfield"
        ]

        self.assertEqual(
            onfield[
                "retained_subject_count"
            ],
            10,
        )

        self.assertEqual(
            onfield[
                "retained_storage_ids"
            ],
            [
                str(value)
                for value
                in range(
                    1001,
                    1011,
                )
            ],
        )

    def test_activity_only(self):
        onfield = self.data[
            "onfield"
        ]

        self.assertEqual(
            onfield[
                "labels"
            ],
            [
                "Activity",
            ],
        )

        self.assertFalse(
            onfield[
                "contains_falling"
            ]
        )

    def test_rejected_cases_never_used(self):
        rejected = self.data[
            "rejected_historical_cases"
        ]

        self.assertEqual(
            rejected[
                "storage_ids"
            ],
            [
                "999",
                "1000",
            ],
        )

        self.assertFalse(
            rejected[
                "use_in_new_project"
            ]
        )

        self.assertFalse(
            rejected[
                "use_as_augmentation"
            ]
        )

        self.assertFalse(
            self.data[
                "leakage_policy"
            ][
                "rejected_999_1000_allowed_anywhere"
            ]
        )

    def test_onfield_not_in_primary_folds(self):
        self.assertFalse(
            self.data[
                "primary_fivefold"
            ][
                "include_onfield"
            ]
        )


if __name__ == "__main__":
    unittest.main()
