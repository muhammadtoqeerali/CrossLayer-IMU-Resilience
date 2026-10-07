#!/usr/bin/env python3
"""Phase-6G pre-forward CSC execution adapter.

This module freezes the *shape of the future execution call* without executing
it.  It bridges already-frozen Phase-6E pair metadata to the exact Phase-4H
sensor and Phase-5 compute execution contracts discovered prospectively.

It intentionally does not import:
  * torch,
  * numpy,
  * sensor_fi_operators,
  * sensor_fi_outer_executor_v1,
  * compute_fi_outer_fleet_executor_v1,
  * compute_fi_execution_harness,
  * compute_fi_fault_only_execution_v1.

No outer arrays, model checkpoint, PTQ state, sensor operator, compute fault,
or model forward can be reached from this implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import csc_outer_executor_v1 as metadata_executor


ROOT = Path(__file__).resolve().parents[2]

EXECUTION_ENABLED = False

FROZEN_METADATA_EXECUTOR_SHA256 = (
    "99320b5811c5e72940bcfc779fa530df"
    "88fb8b7a2f8662711b8f56e4cf5d61fd"
)

FROZEN_PHASE6F_CONFIG_SHA256 = (
    "9bcb887c06701a9a1598b8500b2becd0"
    "59479a5f465c1f11180cd7f551b29d1b"
)

PINNED_API_BINDINGS = {
    "phase4h_executor": {
        "path":
            "experiments/phase_04/sensor_fi_outer_executor_v1.py",

        "sha256":
            "350f3356b5eed65d7e318effdb2c152f"
            "4f022e117fac73fcc0a8ee4d1871e72d",

        "required_symbols":
            [
                "condition_windows",
                "clean_windows",
                "build_parent_instance_maps",
            ],
    },

    "phase4h_runner": {
        "path":
            "experiments/phase_04/sensor_fi_devcal_runner_v2.py",

        "sha256":
            "541af4663293ae5fbaf5286fc3dea767"
            "9d511fdabb50738863880e4712a9b86a",

        "required_symbols":
            [
                "rewindow_sequence",
            ],
    },

    "phase4h_operators": {
        "path":
            "experiments/phase_04/sensor_fi_operators.py",

        "sha256":
            "6559ddc1fcdca352fbd09bf003e28919"
            "bb355a76b2ef52e6985a99188fd69eaa",

        "required_symbols":
            [
                "apply_fault",
            ],
    },

    "phase5_executor": {
        "path":
            "experiments/phase_05/compute_fi_outer_fleet_executor_v1.py",

        "sha256":
            "82424cf132a7272c9dc8a7ffd2271472"
            "b19990a209490f9dbc94bec7a7af9188",

        "required_symbols":
            [
                "_window_tensor",
                "load_model_bundle",
                "build_outer_identity",
                "execute_fault_sequence",
                "execution_active_mask",
            ],
    },

    "phase5_execution_harness": {
        "path":
            "experiments/phase_05/compute_fi_execution_harness.py",

        "sha256":
            "68418e44d50f7c8192777ffe6c7f7f5f"
            "0dc0cab2243788f627d07879ef19005b",

        "required_symbols":
            [
                "load_frozen_fp32_model",
                "build_frozen_ptq_eager",
                "clone_state_dict",
            ],
    },

    "phase5_fault_only_execution": {
        "path":
            "experiments/phase_05/compute_fi_fault_only_execution_v1.py",

        "sha256":
            "f8bb4095bbfedc69c24fa4cc6e79ccf"
            "4914e14fa502ce080a06de01e740112cb",

        "required_symbols":
            [
                "run_fp32_fault_only",
                "run_ptq_activation_buffer_fault_only",
                "PTQWeightFaultOnlySession",
            ],
    },

    "phase5_contract": {
        "path":
            "experiments/phase_05/compute_fi_contract.py",

        "sha256":
            "335efb0090318eb492e3ae1856dedbb4"
            "360840b2e49285f12a6ec14163791557",

        "required_symbols":
            [
                "FaultIdentity",
                "validate_fault_identity",
            ],
    },
}


PHASE4H_RUNNER_V1_SOURCE_HELPERS = {
    "module_alias":
        "V1",

    "owner_module":
        "sensor_fi_devcal_runner",

    "path":
        "experiments/phase_04/sensor_fi_devcal_runner.py",

    "sha256":
        "bee7b43423723d5ccdbf237d6dbcf706"
        "33d9db489d2aca35d1aca26c5100d398",

    "exports": {
        "load_source_trial":
            "load_source_trial",

        "source_trial_path":
            "source_trial_path",
    },
}


INPUT_CONVERSION_CONTRACT = {
    "source_window_dtype":
        "numpy.float32",

    "conversion":
        "torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0)",

    "batch_dimension":
        0,

    "transpose":
        False,

    "permute":
        False,

    "copy_required_by_adapter":
        False,
}


MODEL_BUNDLE_CONTRACT = {
    "fp32": {
        "loader":
            "load_frozen_fp32_model",

        "ptq_clean_state":
            None,

        "model_eval":
            True,
    },

    "ptq_v7": {
        "fp32_base_loader":
            "load_frozen_fp32_model",

        "ptq_builder":
            "build_frozen_ptq_eager",

        "clean_state_capture":
            "clone_state_dict(ptq_model.state_dict())",

        "model_eval":
            True,
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


def _top_level_symbols(
    path: Path,
) -> set[str]:
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    return {
        node.name
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    }


def _validate_phase4h_runner_v2_exports(
    path: Path,
) -> dict[str, Any]:
    """Validate v2-local rewindowing plus the exact V1 helper exports."""

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    module_aliases: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(
            node,
            ast.Import,
        ):
            continue

        for alias in node.names:
            local = (
                alias.asname
                or alias.name.split(".")[-1]
            )

            module_aliases[
                local
            ] = alias.name

    expected_alias = (
        PHASE4H_RUNNER_V1_SOURCE_HELPERS[
            "module_alias"
        ]
    )

    expected_module = (
        PHASE4H_RUNNER_V1_SOURCE_HELPERS[
            "owner_module"
        ]
    )

    if (
        module_aliases.get(
            expected_alias
        )
        != expected_module
    ):
        raise ValueError(
            "Phase4H runner-v2 V1 module alias changed"
        )

    expected_exports = dict(
        PHASE4H_RUNNER_V1_SOURCE_HELPERS[
            "exports"
        ]
    )

    observed_exports: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if len(
            node.targets
        ) != 1:
            continue

        lhs = node.targets[
            0
        ]

        rhs = node.value

        if not (
            isinstance(
                lhs,
                ast.Name,
            )
            and lhs.id in expected_exports
        ):
            continue

        if not (
            isinstance(
                rhs,
                ast.Attribute,
            )
            and isinstance(
                rhs.value,
                ast.Name,
            )
            and rhs.value.id
            == expected_alias
        ):
            raise ValueError(
                f"Phase4H runner-v2 export changed: {lhs.id}"
            )

        observed_exports[
            lhs.id
        ] = rhs.attr

    if observed_exports != expected_exports:
        raise ValueError(
            "Phase4H runner-v2 source-helper exports changed: "
            f"expected={expected_exports} observed={observed_exports}"
        )

    v1_path = (
        ROOT
        / PHASE4H_RUNNER_V1_SOURCE_HELPERS[
            "path"
        ]
    )

    actual_v1_sha = sha256_file(
        v1_path
    )

    expected_v1_sha = (
        PHASE4H_RUNNER_V1_SOURCE_HELPERS[
            "sha256"
        ]
    )

    if actual_v1_sha != expected_v1_sha:
        raise ValueError(
            "Phase4H runner-v1 source-helper hash changed"
        )

    v1_symbols = _top_level_symbols(
        v1_path
    )

    missing = sorted(
        set(
            expected_exports.values()
        )
        - v1_symbols
    )

    if missing:
        raise ValueError(
            "Phase4H runner-v1 source-helper definitions missing: "
            f"{missing}"
        )

    return {
        "binding_style":
            "LOCAL_REWINDOW_PLUS_V1_EXPORTED_SOURCE_HELPERS",

        "v2_module_alias":
            expected_alias,

        "v1_owner_module":
            expected_module,

        "v1_path":
            PHASE4H_RUNNER_V1_SOURCE_HELPERS[
                "path"
            ],

        "v1_sha256":
            actual_v1_sha,

        "exports":
            observed_exports,
    }


def validate_pinned_bindings() -> dict[str, Any]:
    """Validate byte-frozen modules and required symbols without importing them."""

    phase6f_path = (
        ROOT
        / "configs/evaluation/phase6f_csc_metadata_executor_binding_v1.json"
    )

    if (
        sha256_file(
            phase6f_path
        )
        != FROZEN_PHASE6F_CONFIG_SHA256
    ):
        raise ValueError(
            "Phase-6F binding config hash mismatch"
        )

    metadata_path = (
        ROOT
        / "experiments/phase_06/csc_outer_executor_v1.py"
    )

    if (
        sha256_file(
            metadata_path
        )
        != FROZEN_METADATA_EXECUTOR_SHA256
    ):
        raise ValueError(
            "frozen metadata executor hash mismatch"
        )

    validated = metadata_executor.validate_frozen_plan()

    module_results = {}

    for name, binding in PINNED_API_BINDINGS.items():
        path = ROOT / binding[
            "path"
        ]

        actual = sha256_file(
            path
        )

        if actual != binding[
            "sha256"
        ]:
            raise ValueError(
                f"pinned API hash mismatch for {name}: "
                f"expected={binding['sha256']} actual={actual}"
            )

        symbols = _top_level_symbols(
            path
        )

        missing = sorted(
            set(
                binding[
                    "required_symbols"
                ]
            )
            - symbols
        )

        if missing:
            raise ValueError(
                f"pinned API symbols missing from {name}: {missing}"
            )

        module_results[
            name
        ] = {
            "path":
                binding[
                    "path"
                ],

            "sha256":
                actual,

            "required_symbols":
                list(
                    binding[
                        "required_symbols"
                    ]
                ),
        }

        if name == "phase4h_runner":
            module_results[
                name
            ][
                "exported_source_helpers"
            ] = _validate_phase4h_runner_v2_exports(
                path
            )

    return {
        "phase6e_validated":
            validated,

        "modules":
            module_results,
    }


def _target_record(
    *,
    validated: Mapping[str, Any],
    target_name: str,
) -> dict[str, Any]:
    return metadata_executor.target_by_name(
        phase5d=validated[
            "phase6e_validated"
        ][
            "phase5d"
        ],
        target_name=target_name,
    )


def compute_execution_route(
    *,
    model_variant: str,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one frozen target/model member to the exact Phase-5 fault route."""

    representation = str(
        target[
            "representation_class"
        ]
    )

    if model_variant == "fp32":
        if representation not in {
            "fp32_activation",
            "fp32_buffer",
        }:
            raise ValueError(
                "FP32 member is not eligible for this representation"
            )

        return {
            "primitive":
                "run_fp32_fault_only",

            "requires_ptq_clean_state":
                False,

            "ptq_clean_state_restored_after_sequence":
                False,
        }

    if model_variant != "ptq_v7":
        raise ValueError(
            f"unsupported model variant: {model_variant}"
        )

    if representation == "int8_persistent_weight":
        return {
            "primitive":
                "PTQWeightFaultOnlySession.run_fault_only",

            "requires_ptq_clean_state":
                True,

            "ptq_clean_state_restored_after_sequence":
                True,

            "restoration":
                "execute_fault_sequence finally -> session.reset_clean()",
        }

    if representation in {
        "quantized_activation",
        "quantized_buffer",
        "fp32_activation",
        "fp32_buffer",
    }:
        return {
            "primitive":
                "run_ptq_activation_buffer_fault_only",

            "requires_ptq_clean_state":
                False,

            "ptq_clean_state_restored_after_sequence":
                False,
        }

    raise ValueError(
        f"unsupported PTQ representation: {representation}"
    )


def compute_window_sequence(
    *,
    pair_metadata: Mapping[str, Any],
    trial_window_count: int,
) -> list[int]:
    """Exact compute-active window sequence for one pair-member."""

    persistence = str(
        pair_metadata[
            "persistence"
        ]
    )

    inference_index = int(
        pair_metadata[
            "compute_coordinate"
        ][
            "inference_index"
        ]
    )

    trial_window_count = int(
        trial_window_count
    )

    if persistence == "transient_one_inference":
        indices = [
            inference_index
        ]

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        indices = list(
            range(
                inference_index,
                trial_window_count,
            )
        )

    else:
        raise ValueError(
            f"unknown persistence: {persistence}"
        )

    if not indices:
        raise ValueError(
            "compute execution sequence must not be empty"
        )

    if indices != sorted(
        indices
    ):
        raise ValueError(
            "compute execution indices must be monotonic"
        )

    if (
        indices[
            0
        ] < 0
        or indices[
            -1
        ] >= trial_window_count
    ):
        raise ValueError(
            "compute execution index outside trial"
        )

    return indices


def sensor_active_mask_on_compute_sequence(
    *,
    pair_metadata: Mapping[str, Any],
    sensor_exposed_window_indices: Sequence[int],
    trial_window_count: int,
) -> list[bool]:
    """Mark which compute-active windows are simultaneously sensor-faulted."""

    compute_indices = compute_window_sequence(
        pair_metadata=pair_metadata,
        trial_window_count=trial_window_count,
    )

    sensor_exposed = {
        int(
            index
        )
        for index in sensor_exposed_window_indices
    }

    return [
        int(
            index
        ) in sensor_exposed
        for index in compute_indices
    ]


def build_execution_request(
    *,
    pair_metadata: Mapping[str, Any],
    sensor_exposed_window_indices: Sequence[int],
    trial_window_count: int,
    model_variant: str,
    checkpoint_seed: int,
    validated_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct one prospective pair-member execution request.

    This is metadata only.  It does not read the trial or execute either fault.
    """

    eligible = list(
        pair_metadata[
            "eligible_model_variants"
        ]
    )

    if model_variant not in eligible:
        raise ValueError(
            "model variant is not eligible for this pair"
        )

    seeds = list(
        pair_metadata[
            "checkpoint_seeds"
        ]
    )

    checkpoint_seed = int(
        checkpoint_seed
    )

    if checkpoint_seed not in seeds:
        raise ValueError(
            "checkpoint seed is not frozen for this pair"
        )

    target = _target_record(
        validated=validated_bindings,
        target_name=str(
            pair_metadata[
                "target_name"
            ]
        ),
    )

    route = compute_execution_route(
        model_variant=model_variant,
        target=target,
    )

    compute_indices = compute_window_sequence(
        pair_metadata=pair_metadata,
        trial_window_count=trial_window_count,
    )

    sensor_mask = (
        sensor_active_mask_on_compute_sequence(
            pair_metadata=pair_metadata,
            sensor_exposed_window_indices=sensor_exposed_window_indices,
            trial_window_count=trial_window_count,
        )
    )

    overlap_indices = [
        index
        for index, active in zip(
            compute_indices,
            sensor_mask,
        )
        if active
    ]

    timing = pair_metadata[
        "temporal_accounting"
    ]

    if (
        len(
            overlap_indices
        )
        != int(
            timing[
                "temporal_overlap_window_count"
            ]
        )
    ):
        raise ValueError(
            "request overlap disagrees with frozen pair metadata"
        )

    persistence = str(
        pair_metadata[
            "persistence"
        ]
    )

    identity_count = (
        1
        if persistence
        in {
            "transient_one_inference",
            "persistent_from_onset_until_trial_end",
        }
        else 0
    )

    if identity_count != 1:
        raise ValueError(
            "unsupported persistence identity cardinality"
        )

    payload = {
        "schema_version":
            "phase6g_csc_pre_forward_execution_request_v1",

        "execution_enabled":
            False,

        "sensor_fault_id":
            pair_metadata[
                "sensor_fault_id"
            ],

        "sensor_replay_id":
            pair_metadata[
                "sensor_replay_id"
            ],

        "sensor_parent_kind":
            pair_metadata[
                "sensor_parent_kind"
            ],

        "target_name":
            pair_metadata[
                "target_name"
            ],

        "representation_class":
            target[
                "representation_class"
            ],

        "target_role":
            target[
                "target_role"
            ],

        "persistence":
            persistence,

        "model_variant":
            model_variant,

        "checkpoint_seed":
            checkpoint_seed,

        "compute_sampling_instance_id":
            pair_metadata[
                "compute_coordinate"
            ][
                "sampling_instance_id"
            ],

        "element_index":
            int(
                pair_metadata[
                    "compute_coordinate"
                ][
                    "element_index"
                ]
            ),

        "bit_position":
            int(
                pair_metadata[
                    "compute_coordinate"
                ][
                    "bit_position"
                ]
            ),

        "onset_or_inference_index":
            int(
                pair_metadata[
                    "compute_coordinate"
                ][
                    "inference_index"
                ]
            ),

        "trial_window_count":
            int(
                trial_window_count
            ),

        "sensor_exposed_window_indices":
            [
                int(
                    index
                )
                for index in sensor_exposed_window_indices
            ],

        "compute_execution_window_indices":
            compute_indices,

        "sensor_active_mask_on_compute_sequence":
            sensor_mask,

        "simultaneous_overlap_window_indices":
            overlap_indices,

        "phase4h_sensor_path": {
            "executor_entrypoint":
                "condition_windows",

            "operator_entrypoint":
                "apply_fault",

            "sensor_corruption_precedes_window_tensor_conversion":
                True,
        },

        "phase5_input_conversion":
            dict(
                INPUT_CONVERSION_CONTRACT
            ),

        "phase5_model_bundle":
            dict(
                MODEL_BUNDLE_CONTRACT[
                    model_variant
                ]
            ),

        "phase5_compute_route":
            route,

        "phase5_fault_sequence_contract": {
            "function":
                "execute_fault_sequence",

            "identity_count":
                identity_count,

            "transient_one_identity_per_input":
                persistence
                == "transient_one_inference",

            "persistent_exactly_one_identity":
                persistence
                == "persistent_from_onset_until_trial_end",

            "persistent_all_post_onset_mutations_must_be_active":
                persistence
                == "persistent_from_onset_until_trial_end",
        },

        "clean_reference_contract": {
            "inside_fault_loop":
                False,

            "reuse_or_generate_before_fault_execution":
                True,
        },
    }

    request_id = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        **payload,

        "execution_request_id":
            request_id,
    }


def validate_execution_request(
    request: Mapping[str, Any],
) -> None:
    """Validate one pre-forward request without importing execution runtime."""

    if request[
        "execution_enabled"
    ] is not False:
        raise ValueError(
            "pre-forward request must remain execution-disabled"
        )

    persistence = request[
        "persistence"
    ]

    indices = [
        int(
            value
        )
        for value in request[
            "compute_execution_window_indices"
        ]
    ]

    if not indices:
        raise ValueError(
            "empty compute execution sequence"
        )

    if indices != sorted(
        indices
    ):
        raise ValueError(
            "non-monotonic compute sequence"
        )

    onset = int(
        request[
            "onset_or_inference_index"
        ]
    )

    count = int(
        request[
            "trial_window_count"
        ]
    )

    if persistence == "transient_one_inference":
        if indices != [
            onset
        ]:
            raise ValueError(
                "transient execution must contain exactly its frozen window"
            )

        if request[
            "phase5_fault_sequence_contract"
        ][
            "identity_count"
        ] != 1:
            raise ValueError(
                "transient request requires exactly one identity"
            )

    elif (
        persistence
        == "persistent_from_onset_until_trial_end"
    ):
        if indices != list(
            range(
                onset,
                count,
            )
        ):
            raise ValueError(
                "persistent execution must be exact onset-to-end suffix"
            )

        if request[
            "phase5_fault_sequence_contract"
        ][
            "identity_count"
        ] != 1:
            raise ValueError(
                "persistent request requires exactly one identity"
            )

    else:
        raise ValueError(
            "unknown persistence"
        )

    mask = list(
        request[
            "sensor_active_mask_on_compute_sequence"
        ]
    )

    if len(
        mask
    ) != len(
        indices
    ):
        raise ValueError(
            "sensor-active mask length mismatch"
        )

    overlap = [
        int(
            value
        )
        for value in request[
            "simultaneous_overlap_window_indices"
        ]
    ]

    expected_overlap = [
        index
        for index, active in zip(
            indices,
            mask,
        )
        if bool(
            active
        )
    ]

    if overlap != expected_overlap:
        raise ValueError(
            "overlap indices disagree with active mask"
        )

    if (
        request[
            "phase5_input_conversion"
        ]
        != INPUT_CONVERSION_CONTRACT
    ):
        raise ValueError(
            "Phase-5 input conversion contract changed"
        )

    route = request[
        "phase5_compute_route"
    ]

    if (
        request[
            "representation_class"
        ]
        == "int8_persistent_weight"
    ):
        if (
            request[
                "model_variant"
            ]
            != "ptq_v7"
        ):
            raise ValueError(
                "int8 persistent weight route must be PTQ-v7"
            )

        if not route[
            "requires_ptq_clean_state"
        ]:
            raise ValueError(
                "weight route lost PTQ clean-state requirement"
            )

        if not route[
            "ptq_clean_state_restored_after_sequence"
        ]:
            raise ValueError(
                "weight route lost clean-state restoration"
            )

    if request[
        "clean_reference_contract"
    ][
        "inside_fault_loop"
    ] is not False:
        raise ValueError(
            "clean reference forward must remain outside fault loop"
        )

    supplied_id = str(
        request[
            "execution_request_id"
        ]
    )

    payload = dict(
        request
    )

    payload.pop(
        "execution_request_id"
    )

    expected_id = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    if supplied_id != expected_id:
        raise ValueError(
            "execution request ID mismatch"
        )


def execute_request(
    *_args: Any,
    **_kwargs: Any,
) -> None:
    """Hard stop: execution is not authorized at Phase-6G pre-forward stage."""

    raise RuntimeError(
        "PHASE6G_EXECUTION_DISABLED_PRE_FORWARD_ADAPTER: "
        "this module may construct/validate requests only"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "validate-bindings"
    )

    sub.add_parser(
        "execution-status"
    )

    args = parser.parse_args()

    if args.command == "validate-bindings":
        validated = validate_pinned_bindings()

        print(
            "PHASE6G_PRE_FORWARD_ADAPTER_BINDINGS=PASS"
        )

        print(
            "PINNED_MODULE_COUNT="
            + str(
                len(
                    validated[
                        "modules"
                    ]
                )
            )
        )

        print(
            "EXECUTION_ENABLED=False"
        )

        print(
            "OUTER_ARRAY_READ=False"
        )

        print(
            "MODEL_LOADED=False"
        )

        print(
            "SENSOR_FAULT_EXECUTED=False"
        )

        print(
            "COMPUTE_FAULT_EXECUTED=False"
        )

        print(
            "MODEL_FORWARD_EXECUTED=False"
        )

        return

    print(
        "PHASE6G_PRE_FORWARD_ADAPTER_EXECUTION_ENABLED=False"
    )

    print(
        "EXECUTION_REQUIRES_SEPARATE_AUTHORIZATION_AND_FREEZE=True"
    )


if __name__ == "__main__":
    main()
