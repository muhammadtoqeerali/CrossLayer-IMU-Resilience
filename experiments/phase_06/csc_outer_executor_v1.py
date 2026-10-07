#!/usr/bin/env python3
"""Phase-6E metadata-first CSC outer executor.

This module deliberately cannot execute a CSC model forward.

Its only responsibilities in this implementation stage are:

1. validate the frozen Phase-6E plan and all bound upstream artifacts;
2. reproduce deterministic sensor-selection metadata;
3. reproduce deterministic compute-stratum assignment;
4. reproduce Phase-6D-R1 transient-window selection;
5. reproduce frozen Phase-5 compute coordinates;
6. derive temporal sensor/compute overlap metadata;
7. refuse execution until a separately qualified execution implementation
   is frozen.

No sensor operator, model loader, Torch runtime, compute-fault execution core,
outer performance result, validation result, or OnField result is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import compute_fi_outer_sampling_v1 as compute_sampling


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PLAN = (
    ROOT
    / "configs/evaluation/phase6e_csc_execution_plan_v1.json"
)

EXPECTED_PLAN_SHA256 = (
    "888d90f720682d19cfed6ea211d84b12"
    "b4ea8fb1408aaef36cffdb7301975bea"
)

EXPECTED_PAIR_SURFACE_BINDING_SHA256 = (
    "3b39a909ea140db75da52315abcf5efe"
    "2748fcdb2d31f0fc98ddf766b000af53"
)

EXPECTED_COMPUTE_COORDINATE_BINDING_SHA256 = (
    "2117c7b07084566607ccac46db150faa"
    "e81d28bef51d0dcab7c700627d8eb435"
)

EXPECTED_TEMPORAL_WORKLOAD_BINDING_SHA256 = (
    "5b447bca2f45d4889bcf0dc86d6170c"
    "40a1028171a9f1b0ea7a28e2884f12ba8"
)

EXPECTED_MODEL_INDEPENDENT_PAIR_COUNT = 4_237_835
EXPECTED_PAIR_MEMBER_COUNT = 21_793_038

EXECUTION_ENABLED = False

AUTHORITATIVE_API_BINDINGS = {
    "phase4h_executor": {
        "path":
            "experiments/phase_04/sensor_fi_outer_executor_v1.py",

        "sha256":
            "350f3356b5eed65d7e318effdb2c152f"
            "4f022e117fac73fcc0a8ee4d1871e72d",
    },

    "phase4h_runner": {
        "path":
            "experiments/phase_04/sensor_fi_devcal_runner_v2.py",

        "sha256":
            "541af4663293ae5fbaf5286fc3dea767"
            "9d511fdabb50738863880e4712a9b86a",
    },

    "phase4h_operators": {
        "path":
            "experiments/phase_04/sensor_fi_operators.py",

        "sha256":
            "6559ddc1fcdca352fbd09bf003e28919"
            "bb355a76b2ef52e6985a99188fd69eaa",
    },

    "phase4h_sampling_v3": {
        "path":
            "experiments/phase_04/sensor_fi_sampling_v3.py",

        "sha256":
            "a78c1dbdafdfe1b074f66c4a9bcb493"
            "aef89cfd95eb13620bfc0c9f3c5c20f74",
    },

    "phase5_executor": {
        "path":
            "experiments/phase_05/compute_fi_outer_fleet_executor_v1.py",

        "sha256":
            "82424cf132a7272c9dc8a7ffd2271472"
            "b19990a209490f9dbc94bec7a7af9188",
    },

    "phase5_sampling": {
        "path":
            "experiments/phase_05/compute_fi_outer_sampling_v1.py",

        "sha256":
            "8886b74043ef9ba32f2cb8e999d318c3"
            "40c239d818caad53f430594546d2417d",
    },
}


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_file(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def require_hash(
    path: str | Path,
    expected: str,
    label: str,
) -> None:
    actual = sha256_file(
        path
    )

    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: "
            f"expected={expected} actual={actual}"
        )


def _dependency_path(
    plan: Mapping[str, Any],
    key: str,
) -> Path:
    record = plan[
        "frozen_dependencies"
    ][
        key
    ]

    return (
        ROOT
        / record[
            "path"
        ]
    )


def validate_frozen_plan(
    plan_path: str | Path = DEFAULT_PLAN,
) -> dict[str, Any]:
    """Validate the frozen Phase-6E plan without reading outer arrays."""

    plan_path = Path(
        plan_path
    )

    require_hash(
        plan_path,
        EXPECTED_PLAN_SHA256,
        "Phase-6E plan",
    )

    plan = load_json(
        plan_path
    )

    if (
        plan[
            "status"
        ]
        != "FROZEN_PRE_EXECUTION_CSC_PLAN"
    ):
        raise ValueError(
            "unexpected Phase-6E plan status"
        )

    if (
        plan[
            "execution_gate"
        ][
            "status"
        ]
        != "PLAN_FROZEN_EXECUTION_NOT_YET_PERFORMED"
    ):
        raise ValueError(
            "unexpected Phase-6E execution-gate status"
        )

    dependencies = plan[
        "frozen_dependencies"
    ]

    for name, record in dependencies.items():
        if not isinstance(
            record,
            dict,
        ):
            continue

        if (
            "path" not in record
            or "sha256" not in record
        ):
            continue

        require_hash(
            ROOT
            / record[
                "path"
            ],
            record[
                "sha256"
            ],
            f"Phase-6E dependency {name}",
        )

    for name, record in AUTHORITATIVE_API_BINDINGS.items():
        require_hash(
            ROOT
            / record[
                "path"
            ],
            record[
                "sha256"
            ],
            f"authoritative API {name}",
        )

    bindings = plan[
        "derivation_bindings"
    ]

    if (
        bindings[
            "pair_surface_binding_sha256"
        ]
        != EXPECTED_PAIR_SURFACE_BINDING_SHA256
    ):
        raise ValueError(
            "pair-surface binding changed"
        )

    if (
        bindings[
            "combined_compute_coordinate_binding_sha256"
        ]
        != EXPECTED_COMPUTE_COORDINATE_BINDING_SHA256
    ):
        raise ValueError(
            "compute-coordinate binding changed"
        )

    if (
        bindings[
            "temporal_workload_binding_sha256"
        ]
        != EXPECTED_TEMPORAL_WORKLOAD_BINDING_SHA256
    ):
        raise ValueError(
            "temporal-workload binding changed"
        )

    if (
        int(
            plan[
                "pair_surface"
            ][
                "model_independent_csc_pair_count"
            ]
        )
        != EXPECTED_MODEL_INDEPENDENT_PAIR_COUNT
    ):
        raise ValueError(
            "model-independent pair count changed"
        )

    if (
        int(
            plan[
                "model_variant_surface"
            ][
                "combined_seed_variant_expanded_pair_member_count"
            ]
        )
        != EXPECTED_PAIR_MEMBER_COUNT
    ):
        raise ValueError(
            "pair-member count changed"
        )

    strata = plan[
        "compute_strata"
    ]

    if len(
        strata
    ) != 28:
        raise ValueError(
            "compute-stratum count changed"
        )

    if [
        int(
            row[
                "index"
            ]
        )
        for row in strata
    ] != list(
        range(
            28
        )
    ):
        raise ValueError(
            "compute-stratum ordering changed"
        )

    phase6d = load_json(
        _dependency_path(
            plan,
            "phase6d_config",
        )
    )

    r1 = load_json(
        _dependency_path(
            plan,
            "phase6d_r1_config",
        )
    )

    r2 = load_json(
        _dependency_path(
            plan,
            "phase6d_r2_config",
        )
    )

    phase5d = load_json(
        _dependency_path(
            plan,
            "phase5d_outer_protocol",
        )
    )

    if (
        r2[
            "source_trial_candidate_eligibility"
        ][
            "eligible_predicate"
        ]
        != "sensor_exposed_window_count >= 1"
    ):
        raise ValueError(
            "Phase-6D-R2 eligibility rule changed"
        )

    return {
        "plan":
            plan,

        "phase6d":
            phase6d,

        "phase6d_r1":
            r1,

        "phase6d_r2":
            r2,

        "phase5d":
            phase5d,
    }


def sensor_selection_rank(
    *,
    instance: Mapping[str, Any],
    fold: int,
    subject: int,
    family: str,
    severity: str,
    selection_namespace: str,
) -> tuple[str, str]:
    """Exact Phase-6D SHA256 hash-min ranking key."""

    payload = {
        "namespace":
            selection_namespace,

        "fold":
            int(
                fold
            ),

        "subject":
            int(
                subject
            ),

        "parent_kind":
            instance[
                "parent_kind"
            ],

        "parent_sequence_id":
            instance[
                "parent_sequence_id"
            ],

        "family":
            family,

        "severity":
            severity,

        "fault_id":
            instance[
                "fault_id"
            ],

        "replay_id":
            instance[
                "replay_id"
            ],
    }

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        digest,
        str(
            instance[
                "fault_id"
            ]
        ),
    )


def select_stored_window_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    fold: int,
    subject: int,
    family: str,
    severity: str,
    phase6d: Mapping[str, Any],
) -> dict[str, Any]:
    """Phase-6D hash-min selection; R2 does not alter stored windows."""

    if not candidates:
        raise ValueError(
            "stored-window candidate set must not be empty"
        )

    namespace = phase6d[
        "sensor_surface"
    ][
        "selected_instance_rule"
    ][
        "selection_namespace"
    ]

    selected = min(
        candidates,
        key=lambda row: sensor_selection_rank(
            instance=row,
            fold=fold,
            subject=subject,
            family=family,
            severity=severity,
            selection_namespace=namespace,
        ),
    )

    return dict(
        selected
    )


def select_source_trial_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    exposure_count_by_fault_id: Mapping[str, int],
    fold: int,
    subject: int,
    family: str,
    severity: str,
    phase6d: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact R2 observable-first source-trial selection.

    If no frozen candidate is observable, return the frozen structural-
    omission status. No replacement, fallback, or rebalancing occurs.
    """

    if not candidates:
        raise ValueError(
            "source-trial candidate set must not be empty"
        )

    namespace = phase6d[
        "sensor_surface"
    ][
        "selected_instance_rule"
    ][
        "selection_namespace"
    ]

    eligible = [
        row
        for row in candidates
        if int(
            exposure_count_by_fault_id.get(
                str(
                    row[
                        "fault_id"
                    ]
                ),
                0,
            )
        ) >= 1
    ]

    if not eligible:
        return {
            "status":
                "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR",

            "selected":
                None,
        }

    selected = min(
        eligible,
        key=lambda row: sensor_selection_rank(
            instance=row,
            fold=fold,
            subject=subject,
            family=family,
            severity=severity,
            selection_namespace=namespace,
        ),
    )

    return {
        "status":
            "ELIGIBLE_SELECTED",

        "selected":
            dict(
                selected
            ),
    }


def assign_compute_stratum(
    *,
    sensor_instance: Mapping[str, Any],
    phase6d: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Exact Phase-6D sensor-instance -> one-of-28 assignment."""

    strata = phase6d[
        "compute_surface"
    ][
        "compute_strata"
    ]

    if len(
        strata
    ) != 28:
        raise ValueError(
            "frozen compute-stratum surface is not 28"
        )

    payload = {
        "namespace":
            phase6d[
                "compute_stratum_assignment"
            ][
                "namespace"
            ],

        "sensor_fault_id":
            sensor_instance[
                "fault_id"
            ],

        "sensor_replay_id":
            sensor_instance[
                "replay_id"
            ],
    }

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).digest()

    index = (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        % len(
            strata
        )
    )

    return (
        index,
        dict(
            strata[
                index
            ]
        ),
    )


def source_trial_transient_window_index(
    *,
    sensor_replay_id: str,
    target_name: str,
    exposed_window_indices: Sequence[int],
    phase6d_r1: Mapping[str, Any],
) -> int:
    """Exact Phase-6D-R1 transient compute-window rule."""

    windows = [
        int(
            index
        )
        for index in exposed_window_indices
    ]

    if not windows:
        raise ValueError(
            "ABORT_PLAN_DERIVATION_NO_FALLBACK"
        )

    if windows != sorted(
        set(
            windows
        )
    ):
        raise ValueError(
            "exposed windows must be unique ascending trial-local indices"
        )

    rule = phase6d_r1[
        "source_trial_transient_compute_window_selection"
    ]

    payload = {
        "namespace":
            rule[
                "namespace"
            ],

        "sensor_replay_id":
            sensor_replay_id,

        "target_name":
            target_name,
    }

    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).digest()

    offset = (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        % len(
            windows
        )
    )

    return windows[
        offset
    ]


def target_by_name(
    *,
    phase5d: Mapping[str, Any],
    target_name: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in phase5d[
            "target_inventory"
        ]
        if row[
            "target_name"
        ] == target_name
    ]

    if len(
        matches
    ) != 1:
        raise ValueError(
            f"target lookup cardinality != 1: {target_name}"
        )

    return dict(
        matches[
            0
        ]
    )


def build_compute_coordinate(
    *,
    fold: int,
    subject: int,
    task: int,
    trial: int,
    trial_window_count: int,
    target: Mapping[str, Any],
    persistence: str,
    transient_window_index: int | None,
) -> dict[str, Any]:
    """Reproduce the frozen Phase-5 model-independent coordinate."""

    if persistence == "transient_one_inference":
        if transient_window_index is None:
            raise ValueError(
                "transient coordinate requires window index"
            )

        parent_kind = "window"
        payload_window = int(
            transient_window_index
        )

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        if transient_window_index is not None:
            raise ValueError(
                "persistent coordinate is trial-bound"
            )

        parent_kind = "trial"
        payload_window = None

    else:
        raise ValueError(
            f"unknown persistence: {persistence}"
        )

    payload = compute_sampling.canonical_sampling_payload(
        partition="outer_test",
        fold=int(
            fold
        ),
        subject=int(
            subject
        ),
        task=int(
            task
        ),
        trial=int(
            trial
        ),
        parent_kind=parent_kind,
        window_index=payload_window,
        representation_class=target[
            "representation_class"
        ],
        target_name=target[
            "target_name"
        ],
        target_role=target[
            "target_role"
        ],
        persistence=persistence,
        replicate_index=0,
    )

    sid = compute_sampling.sampling_instance_id(
        payload
    )

    element_index = compute_sampling.derive_element_index(
        payload,
        target_numel=int(
            target[
                "target_numel_per_inference"
            ]
        ),
    )

    bit_position = compute_sampling.derive_bit_position(
        payload,
        eligible_bits=target[
            "eligible_bit_positions"
        ],
    )

    if persistence == "transient_one_inference":
        inference_index = (
            compute_sampling.transient_inference_index(
                payload
            )
        )

        if (
            inference_index
            != transient_window_index
        ):
            raise ValueError(
                "Phase-5 transient inference index differs from selected window"
            )

    else:
        inference_index = (
            compute_sampling.derive_persistent_onset_index(
                payload,
                trial_window_count=int(
                    trial_window_count
                ),
            )
        )

    return {
        "payload":
            payload,

        "sampling_instance_id":
            sid,

        "parent_kind":
            parent_kind,

        "element_index":
            int(
                element_index
            ),

        "bit_position":
            int(
                bit_position
            ),

        "inference_index":
            int(
                inference_index
            ),
    }


def temporal_accounting(
    *,
    sensor_exposed_window_indices: Sequence[int],
    trial_window_count: int,
    persistence: str,
    compute_inference_index: int,
) -> dict[str, Any]:
    """Return model-independent sensor/compute temporal geometry."""

    trial_window_count = int(
        trial_window_count
    )

    if trial_window_count <= 0:
        raise ValueError(
            "trial_window_count must be positive"
        )

    exposed = [
        int(
            value
        )
        for value in sensor_exposed_window_indices
    ]

    if not exposed:
        raise ValueError(
            "retained CSC pair must have sensor exposure"
        )

    if exposed != sorted(
        set(
            exposed
        )
    ):
        raise ValueError(
            "sensor exposure indices must be unique ascending"
        )

    if (
        exposed[
            0
        ] < 0
        or exposed[
            -1
        ] >= trial_window_count
    ):
        raise ValueError(
            "sensor exposure index outside trial"
        )

    compute_inference_index = int(
        compute_inference_index
    )

    if not (
        0
        <= compute_inference_index
        < trial_window_count
    ):
        raise ValueError(
            "compute inference/onset index outside trial"
        )

    if persistence == "transient_one_inference":
        compute_active = [
            compute_inference_index
        ]

        if (
            compute_inference_index
            not in exposed
        ):
            raise ValueError(
                "frozen transient compute window must be sensor-exposed"
            )

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        compute_active = list(
            range(
                compute_inference_index,
                trial_window_count,
            )
        )

    else:
        raise ValueError(
            f"unknown persistence: {persistence}"
        )

    exposed_set = set(
        exposed
    )

    active_set = set(
        compute_active
    )

    overlap = sorted(
        exposed_set
        & active_set
    )

    union = sorted(
        exposed_set
        | active_set
    )

    return {
        "sensor_exposed_window_count":
            len(
                exposed
            ),

        "compute_active_window_count":
            len(
                compute_active
            ),

        "temporal_overlap":
            bool(
                overlap
            ),

        "temporal_overlap_window_count":
            len(
                overlap
            ),

        "sensor_compute_union_window_count":
            len(
                union
            ),

        "overlap_window_indices":
            overlap,
    }


def derive_pair_metadata(
    *,
    sensor_parent_kind: str,
    sensor_instance: Mapping[str, Any],
    sensor_exposed_window_indices: Sequence[int],
    fold: int,
    subject: int,
    task: int,
    trial: int,
    trial_window_count: int,
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one frozen CSC pair's metadata with zero fault/model execution."""

    if sensor_parent_kind not in {
        "stored_window",
        "source_trial",
    }:
        raise ValueError(
            "unsupported sensor_parent_kind"
        )

    plan = validated[
        "plan"
    ]

    phase6d = validated[
        "phase6d"
    ]

    r1 = validated[
        "phase6d_r1"
    ]

    phase5d = validated[
        "phase5d"
    ]

    stratum_index, stratum = assign_compute_stratum(
        sensor_instance=sensor_instance,
        phase6d=phase6d,
    )

    target = target_by_name(
        phase5d=phase5d,
        target_name=stratum[
            "target_name"
        ],
    )

    persistence = stratum[
        "persistence"
    ]

    exposed = [
        int(
            value
        )
        for value in sensor_exposed_window_indices
    ]

    if persistence == "transient_one_inference":
        if sensor_parent_kind == "stored_window":
            if len(
                exposed
            ) != 1:
                raise ValueError(
                    "stored-window transient pair requires one parent window"
                )

            transient_window = exposed[
                0
            ]

        else:
            transient_window = (
                source_trial_transient_window_index(
                    sensor_replay_id=str(
                        sensor_instance[
                            "replay_id"
                        ]
                    ),
                    target_name=stratum[
                        "target_name"
                    ],
                    exposed_window_indices=exposed,
                    phase6d_r1=r1,
                )
            )

    else:
        transient_window = None

    coordinate = build_compute_coordinate(
        fold=fold,
        subject=subject,
        task=task,
        trial=trial,
        trial_window_count=trial_window_count,
        target=target,
        persistence=persistence,
        transient_window_index=transient_window,
    )

    timing = temporal_accounting(
        sensor_exposed_window_indices=exposed,
        trial_window_count=trial_window_count,
        persistence=persistence,
        compute_inference_index=coordinate[
            "inference_index"
        ],
    )

    eligible_variants = list(
        stratum[
            "eligible_model_variants"
        ]
    )

    checkpoint_seeds = list(
        plan[
            "model_variant_surface"
        ][
            "ptq_v7"
        ][
            "checkpoint_seeds"
        ]
    )

    if checkpoint_seeds != [
        42,
        123,
        2025,
    ]:
        raise ValueError(
            "checkpoint-seed surface changed"
        )

    return {
        "schema_version":
            "phase6e_csc_pair_metadata_v1",

        "execution_performed":
            False,

        "sensor_parent_kind":
            sensor_parent_kind,

        "sensor_fault_id":
            sensor_instance[
                "fault_id"
            ],

        "sensor_replay_id":
            sensor_instance[
                "replay_id"
            ],

        "compute_stratum_index":
            int(
                stratum_index
            ),

        "target_name":
            stratum[
                "target_name"
            ],

        "persistence":
            persistence,

        "eligible_model_variants":
            eligible_variants,

        "checkpoint_seeds":
            checkpoint_seeds,

        "pair_member_multiplier":
            (
                len(
                    eligible_variants
                )
                * len(
                    checkpoint_seeds
                )
            ),

        "compute_coordinate":
            coordinate,

        "temporal_accounting":
            timing,
    }


def execute_shard(
    *_args: Any,
    **_kwargs: Any,
) -> None:
    """Hard execution gate for the metadata-first implementation."""

    raise RuntimeError(
        "PHASE6E_EXECUTION_DISABLED_METADATA_ONLY: "
        "no sensor fault, compute fault, or model forward is authorized "
        "by csc_outer_executor_v1.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "validate-config"
    )

    sub.add_parser(
        "execution-status"
    )

    args = parser.parse_args()

    if args.command == "validate-config":
        validated = validate_frozen_plan()

        print(
            "PHASE6E_METADATA_EXECUTOR_VALIDATE_CONFIG=PASS"
        )

        print(
            "MODEL_INDEPENDENT_CSC_PAIR_COUNT="
            + str(
                validated[
                    "plan"
                ][
                    "pair_surface"
                ][
                    "model_independent_csc_pair_count"
                ]
            )
        )

        print(
            "PAIR_MEMBER_COUNT="
            + str(
                validated[
                    "plan"
                ][
                    "model_variant_surface"
                ][
                    "combined_seed_variant_expanded_pair_member_count"
                ]
            )
        )

        print(
            "PAIR_FILES_MATERIALIZED=False"
        )

        print(
            "OUTER_ARRAY_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "MODEL_FORWARD_EXECUTED=False"
        )

        print(
            "FAULT_EXECUTION_EXECUTED=False"
        )

        return

    print(
        "PHASE6E_METADATA_EXECUTOR_EXECUTION_ENABLED=False"
    )

    print(
        "EXECUTION_REQUIRES_SEPARATE_QUALIFIED_IMPLEMENTATION=True"
    )


if __name__ == "__main__":
    main()
