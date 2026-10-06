from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from compute_fi_contract import FaultIdentity
from compute_fi_execution_harness import (
    build_frozen_ptq_eager,
    load_frozen_fp32_model,
    outputs_bitwise_equal,
    validate_exact_single_bit_mutation,
)
from compute_fi_outer_executor_v1 import (
    artifact_is_reusable,
    commit_atomic_artifact,
    clean_output_record,
    paired_execution_record,
    prepare_atomic_artifact,
    sha256_file,
    tensor_has_nonfinite,
    write_json,
    write_jsonl,
)
from compute_fi_shard_runtime_v1 import (
    active_mask,
    run_activation_buffer_sequence,
    run_ptq_weight_sequence,
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "phase5h_phase4_devcal",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "unable to load Phase-4 devcal runner"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def qualification_payload(
    *,
    fold,
    subject,
    task,
    trial,
    window_index,
    representation_class,
    target_name,
    target_role,
    persistence,
):
    return {
        "namespace":
            "crosslayer-phase5h-training-calibration-shard-fixture-v1",

        "partition":
            "training_calibration",

        "fold":
            int(fold),

        "subject":
            int(subject),

        "task":
            int(task),

        "trial":
            int(trial),

        "window_index":
            (
                None
                if window_index is None
                else int(window_index)
            ),

        "representation_class":
            representation_class,

        "target_name":
            target_name,

        "target_role":
            target_role,

        "persistence":
            persistence,

        "replicate_index":
            0,
    }


def qualification_sid(payload):
    return hashlib.sha256(
        canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


def coordinates(
    *,
    sid,
    target,
):
    element = (
        int(
            sid[:16],
            16,
        )
        % int(
            target[
                "target_numel_per_inference"
            ]
        )
    )

    bits = [
        int(x)
        for x in target[
            "eligible_bit_positions"
        ]
    ]

    bit = bits[
        int(
            sid[16:32],
            16,
        )
        % len(bits)
    ]

    return (
        element,
        bit,
    )


def make_identity(
    *,
    model_variant,
    seed,
    fold,
    target,
    element,
    bit,
    inference_index,
    persistence,
):
    return FaultIdentity(
        protocol="phase5h_training_calibration_shard_qualification_v1",
        model_variant=model_variant,
        checkpoint_seed=int(seed),
        fold=int(fold),
        fault_family=target[
            "fault_family"
        ],
        representation_class=target[
            "representation_class"
        ],
        target_name=target[
            "target_name"
        ],
        target_role=target[
            "target_role"
        ],
        element_index=int(element),
        bit_position=int(bit),
        inference_index=int(
            inference_index
        ),
        persistence=persistence,
        multiplicity=1,
        replicate_index=0,
    )


def effect_record(
    *,
    outer_instance_id,
    sid,
    identity,
    parent,
    pair,
):
    if pair.mutation.active:
        validate_exact_single_bit_mutation(
            pair.mutation
        )

    return {
        "outer_instance_id":
            outer_instance_id,

        "sampling_instance_id":
            sid,

        "phase5a_fault_id":
            identity.fault_id(),

        "parent":
            dict(parent),

        "current_inference_index":
            int(
                parent[
                    "window_index"
                ]
            ),

        "active":
            bool(
                pair.mutation.active
            ),

        "clean_output_sha256":
            hashlib.sha256(
                pair.clean_output
                .detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            ).hexdigest(),

        "faulted_output_sha256":
            hashlib.sha256(
                pair.faulted_output
                .detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            ).hexdigest(),

        "faulted_output_nonfinite":
            tensor_has_nonfinite(
                pair.faulted_output
            ),

        "changed_indices":
            [
                int(x)
                for x in pair.mutation.changed_indices
            ],
    }


def write_atomic_fixture(
    *,
    root,
    artifact_id,
    artifact_kind,
    plan_sha,
    executor_sha,
    files,
    coverage,
):
    state, temp, success = prepare_atomic_artifact(
        output_root=root,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        recompute_partial=True,
    )

    if state != "compute":
        raise RuntimeError(
            "fixture unexpectedly reused during clean qualification"
        )

    if success is not None:
        raise RuntimeError(
            "unexpected success payload before fixture compute"
        )

    hashes = {}

    for name, value in files.items():
        target = temp / name

        if name.endswith(
            ".jsonl"
        ):
            hashes[name] = write_jsonl(
                target,
                value,
            )

        else:
            hashes[name] = write_json(
                target,
                value,
            )

    commit_atomic_artifact(
        output_root=root,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        temp_dir=temp,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
        output_hashes=hashes,
        coverage=coverage,
    )

    if not artifact_is_reusable(
        output_root=root,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        plan_sha256=plan_sha,
        executor_sha256=executor_sha,
    ):
        raise RuntimeError(
            "atomic fixture not reusable after commit"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--phase5d",
        required=True,
    )
    parser.add_argument(
        "--phase5e-plan",
        required=True,
    )
    parser.add_argument(
        "--phase5g-config",
        required=True,
    )
    parser.add_argument(
        "--calibration",
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
    )
    parser.add_argument(
        "--fp32-freeze",
        required=True,
    )
    parser.add_argument(
        "--ptq-all15",
        required=True,
    )
    parser.add_argument(
        "--phase4-devcal",
        required=True,
    )
    parser.add_argument(
        "--phase5f-core",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    cfg = json.loads(
        Path(args.config).read_text()
    )

    phase5d = json.loads(
        Path(args.phase5d).read_text()
    )

    phase5g = json.loads(
        Path(args.phase5g_config).read_text()
    )

    if cfg[
        "partition"
    ] != "training_calibration":
        raise ValueError(
            "qualification partition changed"
        )

    if phase5g[
        "execution_gate"
    ][
        "outer_execution_enabled"
    ] is not False:
        raise ValueError(
            "outer gate unexpectedly enabled"
        )

    phase4 = load_module(
        Path(args.phase4_devcal)
    )

    phase4.assert_partition(
        "training_calibration",
        execution_stage="qualification",
    )

    for prohibited in (
        "validation",
        "outer_test",
        "onfield",
    ):
        rejected = False

        try:
            phase4.assert_partition(
                prohibited,
                execution_stage="qualification",
            )

        except ValueError:
            rejected = True

        if not rejected:
            raise RuntimeError(
                f"prohibited partition accepted: {prohibited}"
            )

    calibration = json.loads(
        Path(args.calibration).read_text()
    )

    fold = 1
    seed = 42

    fold_record = next(
        row
        for row in calibration[
            "folds"
        ]
        if int(
            row[
                "fold"
            ]
        )
        == fold
    )

    selected = sorted(
        fold_record[
            "selection"
        ],
        key=lambda row: (
            int(
                row[
                    "subject"
                ]
            ),
            int(
                row[
                    "task"
                ]
            ),
            int(
                row[
                    "trial"
                ]
            ),
            int(
                row[
                    "window_index"
                ]
            ),
        ),
    )

    anchor = selected[0]

    subject = int(
        anchor[
            "subject"
        ]
    )
    task = int(
        anchor[
            "task"
        ]
    )
    trial = int(
        anchor[
            "trial"
        ]
    )
    anchor_window = int(
        anchor[
            "window_index"
        ]
    )

    trial_path = (
        Path(args.dataset_root)
        / str(subject)
        / str(task)
        / str(trial)
        / "segments.npy"
    )

    # Training-calibration fixture payload read is explicitly allowed by the
    # Phase-5H frozen qualification contract. This is not outer-test data.
    trial_windows = np.load(
        trial_path,
        allow_pickle=False,
    )

    if (
        trial_windows.ndim != 3
        or tuple(
            trial_windows.shape[1:]
        )
        != (
            30,
            9,
        )
    ):
        raise RuntimeError(
            f"unexpected fixture trial shape: {trial_windows.shape}"
        )

    n_windows = int(
        trial_windows.shape[0]
    )

    if n_windows < 5:
        raise RuntimeError(
            "qualification trial has fewer than 5 windows"
        )

    start = min(
        max(
            0,
            anchor_window - 2,
        ),
        n_windows - 5,
    )

    indices = list(
        range(
            start,
            start + 5,
        )
    )

    if anchor_window not in indices:
        raise RuntimeError(
            "anchor calibration identity is outside fixture sequence"
        )

    onset = indices[2]

    inputs = []
    parents = []

    for window_index in indices:
        window_np = np.asarray(
            trial_windows[
                window_index
            ],
            dtype=np.float32,
        )

        x = torch.from_numpy(
            window_np
        ).unsqueeze(
            0
        )

        if tuple(
            x.shape
        ) != (
            1,
            30,
            9,
        ):
            raise RuntimeError(
                f"unexpected fixture window shape: {tuple(x.shape)}"
            )

        inputs.append(x)

        parents.append({
            "partition":
                "training_calibration",

            "fold":
                fold,

            "subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "window_index":
                int(window_index),
        })

    distinct_trial_row = next(
        row
        for row in selected
        if (
            int(row["subject"]),
            int(row["task"]),
            int(row["trial"]),
        )
        != (
            subject,
            task,
            trial,
        )
    )

    next_trial_np, _ = phase4.load_stored_window(
        Path(args.dataset_root),
        distinct_trial_row,
    )

    next_trial_input = torch.from_numpy(
        np.asarray(
            next_trial_np,
            dtype=np.float32,
        )
    ).unsqueeze(
        0
    )

    fp32_model, fp32_meta = load_frozen_fp32_model(
        Path(args.fp32_freeze),
        seed=seed,
        fold=fold,
    )

    ptq_manifest = json.loads(
        Path(args.ptq_all15).read_text()
    )

    ptq_member = next(
        row
        for row in ptq_manifest[
            "members"
        ]
        if int(row["seed"]) == seed
        and int(row["fold"]) == fold
    )

    ptq_state = Path(
        ptq_member[
            "artifacts"
        ][
            "state_dict"
        ][
            "path"
        ]
    )

    ptq_script_path = Path(
        ptq_member[
            "artifacts"
        ][
            "torchscript"
        ][
            "path"
        ]
    )

    if sha256_file(
        ptq_state
    ) != ptq_member[
        "artifacts"
    ][
        "state_dict"
    ][
        "sha256"
    ]:
        raise RuntimeError(
            "PTQ state hash mismatch"
        )

    if sha256_file(
        ptq_script_path
    ) != ptq_member[
        "artifacts"
    ][
        "torchscript"
    ][
        "sha256"
    ]:
        raise RuntimeError(
            "PTQ TorchScript hash mismatch"
        )

    ptq_model = build_frozen_ptq_eager(
        fp32_model,
        ptq_state,
    )

    ptq_script = torch.jit.load(
        str(ptq_script_path),
        map_location="cpu",
    )

    ptq_script.eval()

    for x in inputs:
        with torch.no_grad():
            eager = ptq_model(
                x.clone()
            )

            scripted = ptq_script(
                x.clone()
            )

        if not outputs_bitwise_equal(
            eager,
            scripted,
        ):
            raise RuntimeError(
                "PTQ eager/TorchScript mismatch in sequence fixture"
            )

    target_by_name = {
        row[
            "target_name"
        ]:
            row
        for row in phase5d[
            "target_inventory"
        ]
    }

    target_cases = [
        (
            "fp32",
            "fc1_output_fp32",
        ),
        (
            "fp32",
            "flatten_buffer_fp32",
        ),
        (
            "ptq_v7",
            "fc1_output_fp32",
        ),
        (
            "ptq_v7",
            "flatten_buffer_fp32",
        ),
        (
            "ptq_v7",
            "conv2_quantized_input",
        ),
        (
            "ptq_v7",
            "post_fc2_quantized_buffer",
        ),
        (
            "ptq_v7",
            "conv_2.0.weight",
        ),
    ]

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_sha = sha256_file(
        Path(args.phase5e_plan)
    )

    executor_sha = sha256_file(
        Path(args.phase5f_core)
    )

    clean_cache_ids = {
        "fp32":
            "phase5h-fixture-clean-fp32-seed42-fold1",

        "ptq_v7":
            "phase5h-fixture-clean-ptq-v7-seed42-fold1",
    }

    clean_models = {
        "fp32":
            fp32_model,

        "ptq_v7":
            ptq_model,
    }

    for variant, model in clean_models.items():
        records = []

        for parent, x in zip(
            parents,
            inputs,
        ):
            with torch.no_grad():
                output = model(
                    x.clone()
                )

            records.append(
                clean_output_record(
                    clean_cache_id=clean_cache_ids[
                        variant
                    ],
                    parent=parent,
                    model_variant=variant,
                    checkpoint_seed=seed,
                    fold=fold,
                    input_tensor=x,
                    output_tensor=output,
                )
            )

        write_atomic_fixture(
            root=output_root,
            artifact_id=clean_cache_ids[
                variant
            ],
            artifact_kind="clean_cache",
            plan_sha=plan_sha,
            executor_sha=executor_sha,
            files={
                "records.jsonl":
                    records,

                "coverage.json":
                    {
                        "complete":
                            True,

                        "record_count":
                            len(records),

                        "sequence_window_indices":
                            indices,
                    },
            },
            coverage={
                "complete":
                    True,

                "record_count":
                    len(records),
            },
        )

    fixture_results = []
    all_instance_ids = []
    coordinate_ledger = {}

    for model_variant, target_name in target_cases:
        target = target_by_name[
            target_name
        ]

        for persistence in (
            "transient_one_inference",
            "persistent_from_onset_until_trial_end",
        ):
            shard_id = (
                "phase5h-fixture-"
                f"{model_variant}-"
                f"{target['representation_class']}-"
                f"{target_name.replace('.', '_')}-"
                + (
                    "transient"
                    if persistence
                    == "transient_one_inference"
                    else "persistent"
                )
            )

            identities = []
            sids = []

            if persistence == "transient_one_inference":
                for current in indices:
                    payload = qualification_payload(
                        fold=fold,
                        subject=subject,
                        task=task,
                        trial=trial,
                        window_index=current,
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
                    )

                    sid = qualification_sid(
                        payload
                    )

                    element, bit = coordinates(
                        sid=sid,
                        target=target,
                    )

                    identities.append(
                        make_identity(
                            model_variant=model_variant,
                            seed=seed,
                            fold=fold,
                            target=target,
                            element=element,
                            bit=bit,
                            inference_index=current,
                            persistence=persistence,
                        )
                    )

                    sids.append(sid)

                    coordinate_ledger.setdefault(
                        (
                            target_name,
                            persistence,
                            current,
                        ),
                        {},
                    )[
                        model_variant
                    ] = (
                        element,
                        bit,
                    )

            else:
                payload = qualification_payload(
                    fold=fold,
                    subject=subject,
                    task=task,
                    trial=trial,
                    window_index=None,
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
                )

                sid = qualification_sid(
                    payload
                )

                element, bit = coordinates(
                    sid=sid,
                    target=target,
                )

                identities = [
                    make_identity(
                        model_variant=model_variant,
                        seed=seed,
                        fold=fold,
                        target=target,
                        element=element,
                        bit=bit,
                        inference_index=onset,
                        persistence=persistence,
                    )
                ]

                sids = [sid]

                coordinate_ledger.setdefault(
                    (
                        target_name,
                        persistence,
                        None,
                    ),
                    {},
                )[
                    model_variant
                ] = (
                    element,
                    bit,
                )

            if target[
                "representation_class"
            ] == "int8_persistent_weight":
                clean_state = torch.load(
                    ptq_state,
                    map_location="cpu",
                    weights_only=False,
                )

                fault_model = build_frozen_ptq_eager(
                    fp32_model,
                    ptq_state,
                )

                pairs, reset_equal = run_ptq_weight_sequence(
                    fault_model=fault_model,
                    clean_model=ptq_model,
                    clean_state_dict=clean_state,
                    inputs=inputs,
                    inference_indices=indices,
                    identities=identities,
                    reset_probe_input=next_trial_input,
                )

                if not reset_equal:
                    raise RuntimeError(
                        "PTQ weight state leaked across trial reset"
                    )

            else:
                model = (
                    fp32_model
                    if model_variant == "fp32"
                    else ptq_model
                )

                pairs = run_activation_buffer_sequence(
                    model=model,
                    inputs=inputs,
                    inference_indices=indices,
                    identities=identities,
                )

                reset_equal = True

            mask = active_mask(
                pairs
            )

            expected_mask = (
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
                if persistence
                == "transient_one_inference"
                else [
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            )

            if mask != expected_mask:
                raise RuntimeError(
                    f"active mask mismatch shard={shard_id}: "
                    f"{mask} != {expected_mask}"
                )

            instance_records = []
            window_effects = []

            if persistence == "transient_one_inference":
                for parent, identity, sid, pair in zip(
                    parents,
                    identities,
                    sids,
                    pairs,
                ):
                    record = paired_execution_record(
                        shard_id=shard_id,
                        clean_cache_id=clean_cache_ids[
                            model_variant
                        ],
                        sampling_instance_id_value=sid,
                        parent=parent,
                        identity=identity,
                        pair=pair,
                    )

                    instance_records.append(
                        record
                    )

                    all_instance_ids.append(
                        record[
                            "outer_instance_id"
                        ]
                    )

                    window_effects.append(
                        effect_record(
                            outer_instance_id=record[
                                "outer_instance_id"
                            ],
                            sid=sid,
                            identity=identity,
                            parent=parent,
                            pair=pair,
                        )
                    )

            else:
                identity = identities[0]
                sid = sids[0]

                active_index = mask.index(
                    True
                )

                instance = paired_execution_record(
                    shard_id=shard_id,
                    clean_cache_id=clean_cache_ids[
                        model_variant
                    ],
                    sampling_instance_id_value=sid,
                    parent=parents[
                        active_index
                    ],
                    identity=identity,
                    pair=pairs[
                        active_index
                    ],
                )

                instance_records.append(
                    instance
                )

                all_instance_ids.append(
                    instance[
                        "outer_instance_id"
                    ]
                )

                for parent, pair in zip(
                    parents,
                    pairs,
                ):
                    window_effects.append(
                        effect_record(
                            outer_instance_id=instance[
                                "outer_instance_id"
                            ],
                            sid=sid,
                            identity=identity,
                            parent=parent,
                            pair=pair,
                        )
                    )

            expected_instances = (
                5
                if persistence
                == "transient_one_inference"
                else 1
            )

            if len(
                instance_records
            ) != expected_instances:
                raise RuntimeError(
                    f"instance-record count mismatch: {shard_id}"
                )

            if len(
                window_effects
            ) != 5:
                raise RuntimeError(
                    f"window-effect count mismatch: {shard_id}"
                )

            write_atomic_fixture(
                root=output_root,
                artifact_id=shard_id,
                artifact_kind="fault_shard",
                plan_sha=plan_sha,
                executor_sha=executor_sha,
                files={
                    "instance_records.jsonl":
                        instance_records,

                    "window_effects.jsonl":
                        window_effects,

                    "coverage.json":
                        {
                            "complete":
                                True,

                            "model_variant":
                                model_variant,

                            "representation_class":
                                target[
                                    "representation_class"
                                ],

                            "target_name":
                                target_name,

                            "persistence":
                                persistence,

                            "active_mask":
                                mask,

                            "instance_record_count":
                                len(
                                    instance_records
                                ),

                            "window_effect_record_count":
                                len(
                                    window_effects
                                ),

                            "clean_cache_id":
                                clean_cache_ids[
                                    model_variant
                                ],

                            "weight_reset_equal":
                                bool(
                                    reset_equal
                                ),
                        },
                },
                coverage={
                    "complete":
                        True,

                    "instance_record_count":
                        len(
                            instance_records
                        ),

                    "window_effect_record_count":
                        len(
                            window_effects
                        ),
                },
            )

            fixture_results.append({
                "shard_id":
                    shard_id,

                "model_variant":
                    model_variant,

                "representation_class":
                    target[
                        "representation_class"
                    ],

                "target_name":
                    target_name,

                "persistence":
                    persistence,

                "active_mask":
                    mask,

                "instance_record_count":
                    len(
                        instance_records
                    ),

                "window_effect_record_count":
                    len(
                        window_effects
                    ),

                "clean_cache_id":
                    clean_cache_ids[
                        model_variant
                    ],

                "weight_reset_equal":
                    bool(
                        reset_equal
                    ),
            })

    if len(
        fixture_results
    ) != 14:
        raise RuntimeError(
            f"expected 14 fault-shard fixtures, got {len(fixture_results)}"
        )

    if len(
        all_instance_ids
    ) != 42:
        raise RuntimeError(
            f"expected 42 fixture execution IDs, got {len(all_instance_ids)}"
        )

    if len(
        all_instance_ids
    ) != len(
        set(all_instance_ids)
    ):
        raise RuntimeError(
            "fixture outer_instance_id collision"
        )

    for target_name in (
        "fc1_output_fp32",
        "flatten_buffer_fp32",
    ):
        for persistence in (
            "transient_one_inference",
            "persistent_from_onset_until_trial_end",
        ):
            keys = (
                [
                    (
                        target_name,
                        persistence,
                        current,
                    )
                    for current in indices
                ]
                if persistence
                == "transient_one_inference"
                else [
                    (
                        target_name,
                        persistence,
                        None,
                    )
                ]
            )

            for key in keys:
                values = coordinate_ledger[
                    key
                ]

                if set(values) != {
                    "fp32",
                    "ptq_v7",
                }:
                    raise RuntimeError(
                        f"missing cross-variant coordinate binding: {key}"
                    )

                if values[
                    "fp32"
                ] != values[
                    "ptq_v7"
                ]:
                    raise RuntimeError(
                        f"cross-variant coordinate mismatch: {key}"
                    )

    result = {
        "schema_version":
            "phase5h_compute_fi_shard_style_execution_qualification_result_v1",

        "status":
            "QUALIFIED_COMPLETE_SHARD_STYLE_EXECUTION",

        "qualification_partition":
            "training_calibration",

        "fixture": {
            "fold":
                fold,

            "seed":
                seed,

            "subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "window_indices":
                indices,

            "persistent_onset_inference_index":
                onset,

            "sequence_length":
                5,
        },

        "counts": {
            "clean_cache_artifacts":
                2,

            "fault_shard_artifacts":
                14,

            "transient_shard_artifacts":
                7,

            "persistent_shard_artifacts":
                7,

            "outer_instance_fixture_records":
                42,

            "window_effect_records":
                70,
        },

        "coverage": {
            "fp32_model_representations": [
                "fp32_activation",
                "fp32_buffer",
            ],

            "ptq_model_representations": [
                "fp32_activation",
                "fp32_buffer",
                "quantized_activation",
                "quantized_buffer",
                "int8_persistent_weight",
            ],

            "persistence_modes": [
                "transient_one_inference",
                "persistent_from_onset_until_trial_end",
            ],

            "transient_active_mask": [
                True,
                True,
                True,
                True,
                True,
            ],

            "persistent_active_mask": [
                False,
                False,
                True,
                True,
                True,
            ],

            "common_fp32_coordinates_reused_across_variants":
                True,

            "clean_cache_shared_between_persistence_modes":
                True,

            "ptq_weight_cross_trial_reset":
                True,

            "ptq_eager_torchscript_clean_bitwise_equal_all_sequence_windows":
                True,

            "all_atomic_artifacts_reusable":
                True,

            "outer_instance_ids_unique":
                True,
        },

        "source_artifacts": {
            "fp32_checkpoint_sha256":
                fp32_meta[
                    "checkpoint_sha256"
                ],

            "ptq_state_sha256":
                ptq_member[
                    "artifacts"
                ][
                    "state_dict"
                ][
                    "sha256"
                ],

            "ptq_torchscript_sha256":
                ptq_member[
                    "artifacts"
                ][
                    "torchscript"
                ][
                    "sha256"
                ],
        },

        "fixture_results":
            fixture_results,

        "scientific_boundary": {
            "training_calibration_payload_used":
                True,

            "outer_test_payload_used":
                False,

            "outer_execution_gate_enabled":
                False,

            "outer_model_forward_executed":
                False,

            "outer_fault_execution_executed":
                False,

            "outer_shard_executed":
                False,

            "outer_prediction_read":
                False,

            "onfield_used":
                False,

            "phase5d_sampling_changed":
                False,

            "phase5d_identity_changed":
                False,

            "phase5e_plan_changed":
                False,

            "CC_result_generated":
                False,

            "CSC_result_generated":
                False,
        },
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE5H_STATUS=QUALIFIED_COMPLETE_SHARD_STYLE_EXECUTION"
    )
    print(
        "TRAINING_CALIBRATION_SEQUENCE_LENGTH=5"
    )
    print(
        "CLEAN_CACHE_ARTIFACTS=2"
    )
    print(
        "FAULT_SHARD_ARTIFACTS=14"
    )
    print(
        "TRANSIENT_SHARDS=7"
    )
    print(
        "PERSISTENT_SHARDS=7"
    )
    print(
        "FIXTURE_OUTER_INSTANCE_IDS=42"
    )
    print(
        "WINDOW_EFFECT_RECORDS=70"
    )
    print(
        "TRANSIENT_ACTIVE_MASK=True,True,True,True,True"
    )
    print(
        "PERSISTENT_ACTIVE_MASK=False,False,True,True,True"
    )
    print(
        "COMMON_FP32_COORDINATE_REUSE=PASS"
    )
    print(
        "PTQ_WEIGHT_CROSS_TRIAL_RESET=PASS"
    )
    print(
        "ATOMIC_ARTIFACT_REUSE=PASS"
    )


if __name__ == "__main__":
    main()
