import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

D = json.loads(
    (
        ROOT
        / "manifests/"
          "phase_3n_final_protocol_freeze_v3.json"
    ).read_text(
        encoding="utf-8"
    )
)


class TestPhase3FinalProtocol(
    unittest.TestCase
):

    def test_frozen(self):
        self.assertEqual(
            D["status"],
            "PASS",
        )

        self.assertEqual(
            D[
                "freeze_status"
            ],
            "PHASE_3_PROTOCOL_FROZEN",
        )

    def test_primary_counts(self):
        p = D[
            "primary_protocol"
        ]

        self.assertEqual(
            p["subjects"],
            61,
        )

        self.assertEqual(
            p["trials"],
            6309,
        )

        self.assertEqual(
            p["windows"],
            273830,
        )

        self.assertEqual(
            p["window_ms"],
            300,
        )

        self.assertEqual(
            p["overlap_percent"],
            50,
        )

    def test_dataset_index_convention(self):
        c = D[
            "event_frame_position_conventions"
        ]

        self.assertTrue(
            c["pass"]
        )

        observed = {
            (
                item[
                    "dataset"
                ],
                item[
                    "onset_frame_minus_position"
                ],
                item[
                    "impact_frame_minus_position"
                ],
            ):
                item[
                    "count"
                ]
            for item
            in c["observed"]
        }

        self.assertEqual(
            observed[
                (
                    "UNIVR",
                    0,
                    0,
                )
            ],
            573,
        )

        self.assertEqual(
            observed[
                (
                    "KFALL",
                    1,
                    1,
                )
            ],
            2346,
        )

    def test_historical_routes(self):
        r = D[
            "route_reconstruction"
        ]

        self.assertEqual(
            r[
                "annotated_event_trial_count"
            ],
            2919,
        )

        self.assertEqual(
            r[
                "piecewise_exact_event_count"
            ],
            2918,
        )

        self.assertEqual(
            r[
                "piecewise_mismatch_count"
            ],
            1,
        )

        self.assertEqual(
            r[
                "full_trial_fallback_qualified_count"
            ],
            1,
        )

        self.assertEqual(
            r[
                "unresolved_route_count"
            ],
            0,
        )

        self.assertEqual(
            r[
                "mapped_window_count"
            ],
            273830,
        )

    def test_exception_is_retained(self):
        e = D[
            "historical_route_exception"
        ]

        self.assertEqual(
            e[
                "event_id"
            ],
            "KFALL_106_T27_R05",
        )

        self.assertTrue(
            e[
                "full_trial_fallback_route_match"
            ]
        )

        self.assertEqual(
            e[
                "classification_labels"
            ],
            "retained unchanged",
        )

    def test_frame_clock(self):
        c = D[
            "event_frame_clock_audit"
        ]

        self.assertEqual(
            c["event_count"],
            2919,
        )

        self.assertEqual(
            c[
                "missing_framecounter_count"
            ],
            0,
        )

        self.assertEqual(
            c[
                "event_position_framecounter_mismatch_count"
            ],
            0,
        )

        self.assertEqual(
            c[
                "framecounter_discontinuity_event_count"
            ],
            0,
        )

    def test_no_model_execution(self):
        b = D[
            "scientific_boundary"
        ]

        self.assertFalse(
            b["model_trained"]
        )

        self.assertFalse(
            b["predictions_executed"]
        )

        self.assertFalse(
            b[
                "int8_calibration_executed"
            ]
        )

        self.assertFalse(
            b["faults_injected"]
        )


if __name__ == "__main__":
    unittest.main()
