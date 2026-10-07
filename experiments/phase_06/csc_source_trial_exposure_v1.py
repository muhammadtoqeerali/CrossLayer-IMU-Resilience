"""Pure Phase-6 CSC source-trial exposure geometry.

This module performs no array reads, sensor-fault execution, model loading,
compute-fault execution, or model forward.

It converts already-frozen Phase-4H source-trial fault-instance metadata plus
the frozen historical retained-window endpoints into the exact Phase-6D-R1
sensor-exposed evaluation-window set required by the Phase-6D-R2
observable-first selection rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

WINDOW_LENGTH_SAMPLES = 30

FINITE_EPISODE_FAMILIES = frozenset(
    {
        "drift",
        "stuck_channel",
        "dropout",
        "frame_loss",
    }
)

UNTIL_END_FAMILIES = frozenset(
    {
        "jitter",
        "delay",
        "orientation",
    }
)

SEQUENCE_FAMILIES = (
    FINITE_EPISODE_FAMILIES
    | UNTIL_END_FAMILIES
)

PINNED_FILES = {
    "phase6d_r1":
        (
            "configs/evaluation/"
            "phase6d_r1_csc_pairing_clarification_v1.json",
            "60b557b16cb7c9d75ea7fd67299c264624b024fe962d436ce336827031faeea0",
        ),

    "phase6d_r2":
        (
            "configs/evaluation/"
            "phase6d_r2_csc_structural_omission_amendment_v1.json",
            "0c67b4c291966f35d2649b5c6065bed1ee5fe076bf0e8f209336348e5bd6d16e",
        ),

    "phase4h_sampling":
        (
            "experiments/phase_04/sensor_fi_sampling_v3.py",
            "a78c1dbdafdfe1b074f66c4a9bcb493aef89cfd95eb13620bfc0c9f3c5c20f74",
        ),

    "phase4h_operators":
        (
            "experiments/phase_04/sensor_fi_operators.py",
            "6559ddc1fcdca352fbd09bf003e28919bb355a76b2ef52e6985a99188fd69eaa",
        ),

    "phase4h_executor":
        (
            "experiments/phase_04/sensor_fi_outer_executor_v1.py",
            "350f3356b5eed65d7e318effdb2c152f4f022e117fac73fcc0a8ee4d1871e72d",
        ),
}


def sha256_file(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def _load_json(
    path: str | Path,
) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def validate_pinned_exposure_contract() -> dict[str, Any]:
    """Validate all bytes and frozen text rules this helper implements."""

    observed: dict[str, str] = {}

    for name, (
        relative_path,
        expected_sha,
    ) in PINNED_FILES.items():
        path = (
            ROOT
            / relative_path
        )

        actual_sha = sha256_file(
            path
        )

        if actual_sha != expected_sha:
            raise ValueError(
                f"pinned exposure dependency changed: {name}"
            )

        observed[
            name
        ] = actual_sha

    r1 = _load_json(
        ROOT
        / PINNED_FILES[
            "phase6d_r1"
        ][0]
    )

    r2 = _load_json(
        ROOT
        / PINNED_FILES[
            "phase6d_r2"
        ][0]
    )

    exposure = r1[
        "sensor_exposed_window_rule"
    ]

    geometry = r1[
        "source_trial_window_geometry"
    ]

    eligibility = r2[
        "source_trial_candidate_eligibility"
    ]

    selection = r2[
        "source_trial_selection_rule"
    ]

    if exposure != {
        "geometry_only":
            True,

        "model_outputs_used":
            False,

        "ordered_set":
            (
                "ascending stored evaluation window_index "
                "within the same subject/task/trial"
            ),

        "performance_outcomes_used":
            False,

        "predicate":
            (
                "window_start < fault_stop_exclusive "
                "and fault_start < window_stop_exclusive"
            ),
    }:
        raise ValueError(
            "Phase6D-R1 sensor-exposure rule changed"
        )

    if geometry != {
        "historical_window_end_semantics":
            "exclusive_stop",

        "window_length_samples":
            30,

        "window_start":
            "historical_window_end - 30",

        "window_stop_exclusive":
            "historical_window_end",

        "window_stride_samples":
            15,

        "window_support_inclusive":
            "[historical_window_end-30,historical_window_end-1]",
    }:
        raise ValueError(
            "Phase6D-R1 retained-window geometry changed"
        )

    if (
        eligibility[
            "eligible_predicate"
        ]
        != "sensor_exposed_window_count >= 1"
    ):
        raise ValueError(
            "Phase6D-R2 eligibility predicate changed"
        )

    if (
        eligibility[
            "evaluation_geometry"
        ]
        != "exact frozen Phase-6D-R1 retained-window geometry"
    ):
        raise ValueError(
            "Phase6D-R2 evaluation geometry changed"
        )

    if (
        eligibility[
            "sensor_active_support"
        ]
        != "exact frozen Phase-6D-R1 sensor active-support rule"
    ):
        raise ValueError(
            "Phase6D-R2 active-support binding changed"
        )

    expected_order = [
        "generate exact frozen Phase-4H candidate instances",
        "compute geometry-only retained-window exposure",
        "retain candidates with sensor_exposed_window_count >= 1",
        (
            "if eligible set is nonempty apply unchanged "
            "Phase-6D SHA256 hash-min ranking"
        ),
        (
            "if eligible set is empty mark "
            "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR"
        ),
        (
            "only after retained sensor selection assign "
            "frozen Phase-6D compute stratum"
        ),
    ]

    if selection[
        "order"
    ] != expected_order:
        raise ValueError(
            "Phase6D-R2 selection order changed"
        )

    return {
        "status":
            "PINNED_EXPOSURE_CONTRACT_VALID",

        "module_sha256":
            observed,

        "window_length_samples":
            WINDOW_LENGTH_SAMPLES,

        "exposure_predicate":
            exposure[
                "predicate"
            ],

        "eligibility_predicate":
            eligibility[
                "eligible_predicate"
            ],
    }


def source_trial_fault_active_support(
    *,
    instance: Mapping[str, Any],
    source_length: int,
) -> tuple[int, int]:
    """Return frozen [fault_start, fault_stop_exclusive) support."""

    source_length = int(
        source_length
    )

    if source_length <= 0:
        raise ValueError(
            "source_length must be positive"
        )

    if (
        instance.get(
            "parent_kind"
        )
        != "source_trial"
    ):
        raise ValueError(
            "active-support helper requires source_trial parent"
        )

    family = str(
        instance.get(
            "family",
            ""
        )
    )

    if family not in SEQUENCE_FAMILIES:
        raise ValueError(
            f"unsupported source-trial family: {family}"
        )

    onset = int(
        instance.get(
            "onset_sample",
            0,
        )
    )

    if not (
        0
        <= onset
        < source_length
    ):
        raise ValueError(
            "fault onset outside source trial"
        )

    duration = instance.get(
        "duration_samples"
    )

    persistence = str(
        instance.get(
            "persistence",
            ""
        )
    )

    if family in FINITE_EPISODE_FAMILIES:
        if persistence != "transient":
            raise ValueError(
                "finite Phase4H sequence family must be transient"
            )

        if duration is None:
            raise ValueError(
                "finite Phase4H sequence family requires duration"
            )

        duration = int(
            duration
        )

        if duration <= 0:
            raise ValueError(
                "fault duration must be positive"
            )

        fault_stop_exclusive = (
            onset
            + duration
        )

        if fault_stop_exclusive > source_length:
            raise ValueError(
                "fault support exceeds source trial"
            )

    else:
        if persistence != "until_end":
            raise ValueError(
                "full-trial Phase4H sequence family must persist until_end"
            )

        if onset != 0:
            raise ValueError(
                "full-trial Phase4H sequence family must start at zero"
            )

        if duration is not None:
            raise ValueError(
                "full-trial Phase4H sequence family must have null duration"
            )

        fault_stop_exclusive = (
            source_length
        )

    return (
        onset,
        int(
            fault_stop_exclusive
        ),
    )


def source_trial_exposed_window_indices(
    *,
    instance: Mapping[str, Any],
    historical_window_ends: Sequence[int],
    source_length: int,
) -> list[int]:
    """Return exact R1 sensor-exposed stored evaluation window indices."""

    source_length = int(
        source_length
    )

    (
        fault_start,
        fault_stop_exclusive,
    ) = source_trial_fault_active_support(
        instance=instance,
        source_length=source_length,
    )

    exposed: list[int] = []

    for window_index, raw_end in enumerate(
        historical_window_ends
    ):
        window_stop_exclusive = int(
            raw_end
        )

        window_start = (
            window_stop_exclusive
            - WINDOW_LENGTH_SAMPLES
        )

        if window_start < 0:
            raise ValueError(
                "historical evaluation window starts before source trial"
            )

        if window_stop_exclusive > source_length:
            raise ValueError(
                "historical evaluation window exceeds source trial"
            )

        if (
            window_start
            < fault_stop_exclusive
            and fault_start
            < window_stop_exclusive
        ):
            exposed.append(
                int(
                    window_index
                )
            )

    return exposed


def source_trial_exposure_census(
    *,
    candidates: Sequence[Mapping[str, Any]],
    historical_window_ends: Sequence[int],
    source_length: int,
) -> dict[str, Any]:
    """Produce the R2 exposure maps consumed by source-trial selection."""

    indices_by_fault_id: dict[str, list[int]] = {}
    count_by_fault_id: dict[str, int] = {}

    for instance in candidates:
        fault_id = str(
            instance.get(
                "fault_id",
                ""
            )
        )

        if not fault_id:
            raise ValueError(
                "source-trial candidate has empty fault_id"
            )

        if fault_id in indices_by_fault_id:
            raise ValueError(
                f"duplicate source-trial fault_id: {fault_id}"
            )

        indices = (
            source_trial_exposed_window_indices(
                instance=instance,
                historical_window_ends=historical_window_ends,
                source_length=source_length,
            )
        )

        indices_by_fault_id[
            fault_id
        ] = indices

        count_by_fault_id[
            fault_id
        ] = len(
            indices
        )

    return {
        "schema_version":
            "phase6i_source_trial_exposure_census_v1",

        "geometry_only":
            True,

        "execution_performed":
            False,

        "sensor_exposed_window_indices_by_fault_id":
            indices_by_fault_id,

        "exposure_count_by_fault_id":
            count_by_fault_id,
    }


def main() -> int:
    validated = validate_pinned_exposure_contract()

    print(
        "PHASE6I_SOURCE_TRIAL_EXPOSURE_BINDINGS=PASS"
    )

    print(
        "PINNED_MODULE_COUNT="
        + str(
            len(
                validated[
                    "module_sha256"
                ]
            )
        )
    )

    print(
        "OUTER_ARRAY_READ=False"
    )

    print(
        "SENSOR_FAULT_EXECUTED=False"
    )

    print(
        "MODEL_LOADED=False"
    )

    print(
        "COMPUTE_FAULT_EXECUTED=False"
    )

    print(
        "MODEL_FORWARD_EXECUTED=False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
