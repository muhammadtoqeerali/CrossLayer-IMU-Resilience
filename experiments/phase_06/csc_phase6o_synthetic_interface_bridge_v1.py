"""Phase6O prospective CSC execution-interface bridge, synthetic only.

Exercises the actual frozen Phase6J:
    condition_sensor_parent
    execute_bound_compute_fault_sequence
    synthetic_execute_pair_member
    run_model_member_stream

Every dependency that could access signals, operate on tensors,
load checkpoints, or perform inference is substituted with a
self-contained synthetic implementation operating on string tokens.

The real frozen Phase6M representative metadata and Phase6G
pre-forward requests are used. No raw sensor/label/probability
payload is read, no actual model is loaded, and no CSC result or
Phase6H success artifact is created.

This qualifies integration interfaces, not scientific CSC execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import csc_execution_adapter_v1 as adapter
import csc_execution_runtime_v1 as runtime
import csc_phase6k_metadata_fleet_auditor_v1 as auditor
import csc_phase6m_canonical_request_witness_v1 as witness_module
import csc_phase6m_pre_forward_orchestrator_v1 as preflight_module
import csc_phase6n_canonical_member_stream_v1 as canonical

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

EXECUTION_BLOCK = "PHASE6O_REAL_CSC_EXECUTION_NOT_AUTHORIZED"
BRIDGE_ABORT = "PHASE6O_SYNTHETIC_BRIDGE_ABORT"

PINS = {
    "manifests/phase_6n_canonical_member_evidence_freeze_v1.json":
        "3d7585fdf355b7ced8e692a5076e6b90a26eea25ef43060968eaaea2b598dc8e",
    "manifests/phase_6m_csc_pre_forward_evidence_freeze_v1.json":
        "a77b437cf669894b4423caa65f9f094f27074b58e862261376c169d04b1a9607",
    "manifests/phase_6l_producer_aware_cache_evidence_freeze_v1.json":
        "d1b355fe0d4ff13d7eebb6a6f14ba7af72f47da06519c9fbce14290c6d9c4ba5",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_execution_adapter_v1.py":
        "2d8d218b0d2214e099dab28b769eeb9fb90d6fa6420b50351b21e54cc4bd1fb5",
    "experiments/phase_06/csc_phase6m_canonical_request_witness_v1.py":
        "6b3d8028b31ad52e10a0fb4e4060e0b7bf8866177cced7630c825adeb91fd774",
    "experiments/phase_06/csc_phase6m_pre_forward_orchestrator_v1.py":
        "c19eb646ca3e5ad584c737af7d4a36383588b8df07430ba3ab1dd368e386af0e",
    "experiments/phase_06/csc_phase6n_canonical_member_stream_v1.py":
        "55ae23c5f84431a9b41c34fcfd82ff7fe3ca3c3690163b07c11299d7e9b504c7",
    "manifests/phase_6m_pre_forward_evidence_v1/phase6m_canonical_request_synthetic_witness_v1.json":
        "7ad0376d72161702cd36e60117c12fa2a85bf4405e5e032681525f40ce023d46",
    "manifests/phase_6o_interface_evidence_v1/phase6o_prospective_csc_execution_contract_inventory_v1.json":
        "c31352b05e88fb7c11417660081b041c3e33a68f84e0390c18f1d5a804ed77bf",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check_pins() -> None:
    for relative, expected in PINS.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(
                BRIDGE_ABORT + ": frozen source: " + relative
            )

    for module, relative in (
        (
            runtime,
            "experiments/phase_06/csc_execution_runtime_v1.py",
        ),
        (
            adapter,
            "experiments/phase_06/csc_execution_adapter_v1.py",
        ),
        (
            witness_module,
            "experiments/phase_06/csc_phase6m_canonical_request_witness_v1.py",
        ),
        (
            canonical,
            "experiments/phase_06/csc_phase6n_canonical_member_stream_v1.py",
        ),
    ):
        if Path(module.__file__).resolve() != (
            ROOT / relative
        ).resolve():
            raise RuntimeError(
                BRIDGE_ABORT + ": module origin"
            )

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and PRODUCTION_BODY_RELEASED is False
        and runtime.EXECUTION_AUTHORIZED is False
        and preflight_module.EXECUTION_AUTHORIZED is False
        and canonical.EXECUTION_AUTHORIZED is False
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": execution authorization"
        )


class SyntheticExposure:
    """Only returns previously audited frozen exposure indices."""

    def __init__(self, indices):
        self.indices = list(indices)
        self.validations = 0

    def validate_pinned_exposure_contract(self):
        self.validations += 1

    def source_trial_exposed_window_indices(
        self,
        *,
        instance,
        historical_window_ends,
        source_length,
    ):
        assert int(source_length) > 0
        assert historical_window_ends
        assert instance["synthetic_operator_fixture"] is True
        return list(self.indices)


class SyntheticOperators:
    """Never invokes a real Phase4H fault operator."""

    def __init__(self, events):
        self.events = events
        self.calls = 0

    def apply_fault(
        self,
        parent,
        instance,
        *,
        reference_scales=None,
    ):
        if instance.get("synthetic_operator_fixture") is not True:
            raise RuntimeError(
                BRIDGE_ABORT + ": non-synthetic sensor instance"
            )

        if not isinstance(parent, str):
            raise RuntimeError(
                BRIDGE_ABORT + ": non-synthetic parent input"
            )

        self.calls += 1
        self.events.append("synthetic_sensor_operator")

        return (
            "SYNTHETIC_CONDITIONED:" + parent,
            {"family": instance["family"], "synthetic": True},
        )


class SyntheticRunner:
    """Simulates rewindowing without reading actual IMU data."""

    def __init__(self, window_count, events):
        self.window_count = int(window_count)
        self.events = events
        self.calls = 0

    def rewindow_sequence(
        self,
        corrupted_source,
        stored_labels,
        *,
        fall_start_frame=None,
    ):
        if not (
            isinstance(corrupted_source, str)
            and corrupted_source.startswith("SYNTHETIC_CONDITIONED:")
            and stored_labels == "SYNTHETIC_LABEL_FIXTURE"
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": invalid synthetic rewindow inputs"
            )

        self.events.append("synthetic_rewindow")
        self.calls += 1

        return [
            f"SYNTHETIC_REWINDOWED:{index}"
            for index in range(self.window_count)
        ]


class SyntheticPhase5Executor:
    """Simulated Phase5 executor; does not use tensors or a model."""

    def __init__(self, events):
        self.events = events
        self.window_tensor_calls = 0
        self.sequence_calls = 0
        self.mask_calls = 0

    def _window_tensor(self, trial_windows, index):
        value = trial_windows[int(index)]

        if not (
            isinstance(value, str)
            and value.startswith("SYNTHETIC_")
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": non-synthetic trial window"
            )

        self.window_tensor_calls += 1
        self.events.append("synthetic_tensor_conversion")
        return value

    def execute_fault_sequence(
        self,
        *,
        model,
        inputs,
        inference_indices,
        identities,
        ptq_clean_state=None,
    ):
        if model != "SYNTHETIC_MODEL_BUNDLE":
            raise RuntimeError(
                BRIDGE_ABORT + ": non-synthetic model"
            )

        if not (
            len(identities) == 1
            and identities[0] == "SYNTHETIC_COMPUTE_IDENTITY"
            and len(inputs) == len(inference_indices)
            and all(isinstance(value, str) for value in inputs)
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": synthetic compute identity"
            )

        self.sequence_calls += 1
        self.events.append("synthetic_compute_sequence")

        return [
            {
                "synthetic_execution_index": int(index),
                "active": True,
            }
            for index in inference_indices
        ]

    def execution_active_mask(self, executions):
        self.mask_calls += 1

        return [
            bool(item["active"])
            for item in executions
        ]


def synthetic_condition(
    witness,
    *,
    events,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Run frozen conditioning logic with synthetic-only dependencies."""

    kind = witness["sensor_parent_kind"]
    exposed = list(witness["sensor_exposed_window_indices"])
    count = int(witness["trial_window_count"])

    instance = {
        "synthetic_operator_fixture": True,
        "family": witness["sensor_family"],
        "fault_id": witness["sensor_fault_id"],
        "replay_id": witness["sensor_replay_id"],
    }

    operator = SyntheticOperators(events)
    runner = SyntheticRunner(count, events)
    exposure = SyntheticExposure(exposed)

    stored_windows = (
        [
            f"SYNTHETIC_STORED:{index}"
            for index in range(count)
        ]
        if kind == "stored_window"
        else None
    )

    result = runtime.condition_sensor_parent(
        parent_kind=kind,
        instance=instance,
        parent_local_index=int(witness["parent_local_index"]),
        trial_window_count=count,
        stored_windows=stored_windows,
        source_trial=(
            "SYNTHETIC_SOURCE_TRIAL"
            if kind == "source_trial"
            else None
        ),
        stored_labels=(
            "SYNTHETIC_LABEL_FIXTURE"
            if kind == "source_trial"
            else None
        ),
        fall_start_frame=None,
        historical_window_ends=(
            list(range(1, count + 1))
            if kind == "source_trial"
            else None
        ),
        source_length=(
            count if kind == "source_trial" else None
        ),
        reference_scales=None,
        operators=operator,
        runner=runner,
        exposure_helper=exposure,
    )

    if not (
        operator.calls == 1
        and exposure.validations == 1
        and result["sensor_exposed_window_indices"] == exposed
        and set(result["conditioned_windows_by_index"]) == set(exposed)
        and result["operator_audit"]["synthetic"] is True
        and runner.calls == int(kind == "source_trial")
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": synthetic conditioning contract"
        )

    conditioned = result["conditioned_windows_by_index"]

    if not all(
        isinstance(value, str)
        and value.startswith("SYNTHETIC_")
        for value in conditioned.values()
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": non-synthetic conditioned result"
        )

    return conditioned, {
        "parent_kind": kind,
        "operator_calls": operator.calls,
        "rewindow_calls": runner.calls,
        "source_exposure_stub_used": kind == "source_trial",
    }


def synthetic_execute_one(
    *,
    bundle,
    job,
) -> dict[str, Any]:
    """Run actual frozen integration functions with synthetic fixtures."""

    request = job["request"]
    witness = job["witness"]
    clean = job["clean"]

    adapter.validate_execution_request(request)

    if not (
        request["execution_enabled"] is False
        and request["model_variant"]
            == bundle["synthetic_member_identity"]["model_variant"]
        and int(request["checkpoint_seed"])
            == int(bundle["synthetic_member_identity"]["checkpoint_seed"])
        and clean["clean_cache_id"] == job["clean_cache_id"]
        and clean["execution_authorized"] is False
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": request or cache identity"
        )

    exposed = list(request["sensor_exposed_window_indices"])
    compute = list(request["compute_execution_window_indices"])

    events = []

    conditioned, conditioning_audit = synthetic_condition(
        witness,
        events=events,
    )

    windows = {
        int(index): f"SYNTHETIC_CLEAN:{index}"
        for index in sorted(set(exposed) | set(compute))
    }

    windows.update(conditioned)

    fake_executor = SyntheticPhase5Executor(events)

    def reference_hook(index, window):
        if not (
            index in exposed
            and isinstance(window, str)
            and window.startswith("SYNTHETIC_")
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": synthetic reference window"
            )

        events.append("synthetic_reference")

        return witness_module._reference_summary(index, window)

    def fault_hook(indices, inputs, request_arg):
        if (
            list(indices) != compute
            or request_arg["execution_request_id"]
                != request["execution_request_id"]
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": frozen compute sequence route"
            )

        phase5_result = runtime.execute_bound_compute_fault_sequence(
            request=request_arg,
            trial_windows=windows,
            identities=["SYNTHETIC_COMPUTE_IDENTITY"],
            model_bundle={
                "model": bundle["model"],
                "ptq_clean_state": bundle["ptq_clean_state"],
            },
            phase5_executor=fake_executor,
        )

        if not (
            phase5_result["execution_window_indices"] == compute
            and len(phase5_result["executions"]) == len(compute)
            and phase5_result["execution_active_mask"]
                == [True] * len(compute)
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": synthetic Phase5 results"
            )

        return witness_module._fault_summaries(
            indices,
            inputs,
            request_arg,
        )

    reference_cache_id = runtime.sensor_reference_cache_id(
        fold=5,
        subject=9,
        task=int(witness["task"]),
        trial=int(witness["trial"]),
        sensor_parent_kind=witness["sensor_parent_kind"],
        parent_local_index=int(witness["parent_local_index"]),
        sensor_fault_id=witness["sensor_fault_id"],
        sensor_replay_id=witness["sensor_replay_id"],
        model_variant=request["model_variant"],
        checkpoint_seed=int(request["checkpoint_seed"]),
    )

    shard_id = runtime.shard_id(
        fold=5,
        subject=9,
        sensor_family=witness["sensor_family"],
    )

    result = runtime.synthetic_execute_pair_member(
        request=request,
        shard_id_value=shard_id,
        fold=5,
        subject=9,
        task=int(witness["task"]),
        trial=int(witness["trial"]),
        parent_local_index=int(witness["parent_local_index"]),
        phase5_clean_cache_id=clean["clean_cache_id"],
        sensor_reference_cache_id_value=reference_cache_id,
        windows_by_index=windows,
        hooks=runtime.SyntheticHooks(
            reference_forward=reference_hook,
            fault_sequence=fault_hook,
        ),
    )

    if not (
        len(result["sensor_reference_rows"]) == len(exposed)
        and len(result["csc_fault_rows"]) == len(compute)
        and fake_executor.window_tensor_calls == len(compute)
        and fake_executor.sequence_calls == 1
        and fake_executor.mask_calls == 1
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": frozen synthetic record lifecycle"
        )

    reference_positions = [
        index for index, event in enumerate(events)
        if event == "synthetic_reference"
    ]

    compute_positions = [
        index for index, event in enumerate(events)
        if event == "synthetic_compute_sequence"
    ]

    conversion_positions = [
        index for index, event in enumerate(events)
        if event == "synthetic_tensor_conversion"
    ]

    if not (
        len(reference_positions) == len(exposed)
        and len(compute_positions) == 1
        and len(conversion_positions) == len(compute)
        and max(reference_positions) < min(conversion_positions)
        and max(conversion_positions) < compute_positions[0]
        and events[0] == "synthetic_sensor_operator"
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": synthetic operation ordering"
        )

    exposed_set = set(exposed)
    kinds = [
        row["reference_kind"]
        for row in result["csc_fault_rows"]
    ]

    expected_kinds = [
        (
            "phase6_sensor_reference"
            if index in exposed_set
            else "phase5_clean_cache"
        )
        for index in compute
    ]

    if kinds != expected_kinds:
        raise RuntimeError(
            BRIDGE_ABORT + ": reference provenance route"
        )

    overlap = len(set(exposed) & set(compute))

    if result["pair_member"]["zero_temporal_overlap"] is not (
        overlap == 0
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": zero-overlap provenance"
        )

    return {
        "execution_request_id": request["execution_request_id"],
        "sensor_family": witness["sensor_family"],
        "sensor_parent_kind": witness["sensor_parent_kind"],
        "model_variant": request["model_variant"],
        "checkpoint_seed": int(request["checkpoint_seed"]),
        "clean_cache_id": clean["clean_cache_id"],
        "synthetic_sensor_reference_rows": len(exposed),
        "synthetic_compute_fault_rows": len(compute),
        "simultaneous_overlap_rows": overlap,
        "synthetic_operator_calls":
            conditioning_audit["operator_calls"],
        "synthetic_rewindow_calls":
            conditioning_audit["rewindow_calls"],
        "synthetic_phase5_sequence_calls":
            fake_executor.sequence_calls,
        "execution_authorized": False,
    }


def qualify_synthetic_bridge() -> dict[str, Any]:
    """Validate all 189 frozen subject-9 witness requests."""

    check_pins()

    archived = json.loads(
        (
            ROOT
            / "manifests/phase_6m_pre_forward_evidence_v1/"
            "phase6m_canonical_request_synthetic_witness_v1.json"
        ).read_text(encoding="utf-8")
    )

    if not (
        archived["qualification_status"]
        == "FROZEN_SUBJECT_REQUEST_WITNESSES_PASS"
        and archived["distinct_pair_witness_count"] == 37
        and archived["request_counts"]["requests"] == 189
        and archived["execution_authorized"] is False
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": frozen prior witness"
        )

    previous_ids = {
        row["execution_request_id"]
        for row in archived["requests"]
    }

    if len(previous_ids) != 189:
        raise RuntimeError(
            BRIDGE_ABORT + ": historical request IDs"
        )

    preflight = preflight_module.FrozenCSCPreForwardPlan()
    context = auditor.FrozenContext()
    capture = witness_module.CapturingSubjectAudit(context, 9)

    audited = capture.run()

    if not (
        capture.fold == 5
        and len(audited["rows"]) == 12
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": subject-9 reconstruction"
        )

    examples = {
        row["sensor_fault_id"]: row
        for row in capture.examples_by_tag.values()
    }

    if len(examples) != 37:
        raise RuntimeError(
            BRIDGE_ABORT + ": 37-pair witness surface"
        )

    bindings = adapter.validate_pinned_bindings()

    groups = defaultdict(list)
    built_ids = set()

    for example in examples.values():
        pair = example["pair_metadata"]
        exposed = list(example["sensor_exposed_window_indices"])

        member_surface = canonical.validate_member_surface(pair)

        for variant, seed in member_surface:
            request = adapter.build_execution_request(
                pair_metadata=pair,
                sensor_exposed_window_indices=exposed,
                trial_window_count=int(
                    example["trial_window_count"]
                ),
                model_variant=variant,
                checkpoint_seed=seed,
                validated_bindings=bindings,
            )

            adapter.validate_execution_request(request)

            request_id = request["execution_request_id"]

            if request_id in built_ids:
                raise RuntimeError(
                    BRIDGE_ABORT + ": duplicate request"
                )

            built_ids.add(request_id)

            clean = preflight._cache_by_member[
                (5, 9, variant, seed)
            ]

            if clean["execution_authorized"] is not False:
                raise RuntimeError(
                    BRIDGE_ABORT + ": clean cache execution state"
                )

            family = example["sensor_family"]

            shard_id = runtime.shard_id(
                fold=5,
                subject=9,
                sensor_family=family,
            )

            if clean["clean_cache_id"] not in (
                preflight.shard(shard_id)[
                    "prospective_candidate_clean_cache_ids"
                ]
            ):
                raise RuntimeError(
                    BRIDGE_ABORT + ": clean-cache owner"
                )

            identity = canonical.canonical_pair_identity(
                task=int(example["task"]),
                trial=int(example["trial"]),
                kind=example["sensor_parent_kind"],
                parent_local_index=int(
                    example["parent_local_index"]
                ),
                severity=example["severity"],
                replay_id=example["sensor_replay_id"],
            )

            sort_record = {
                **identity,
                "model_variant": variant,
                "checkpoint_seed": seed,
            }

            groups[(family, variant, seed)].append({
                "request": request,
                "witness": example,
                "clean": clean,
                "clean_cache_id": clean["clean_cache_id"],
                "canonical_sort_record": sort_record,
            })

    if not (
        built_ids == previous_ids
        and len(groups) == 69
        and sum(len(group) for group in groups.values()) == 189
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": previous request reconciliation"
        )

    loaded = Counter()
    released = Counter()
    executed = Counter()
    counters = Counter()
    observed_ids = set()
    stream_rows = []

    for key in sorted(groups):
        family, variant, seed = key

        jobs = sorted(
            groups[key],
            key=lambda job: runtime.canonical_pair_member_sort_key(
                job["canonical_sort_record"]
            ),
        )

        sort_keys = [
            runtime.canonical_pair_member_sort_key(
                job["canonical_sort_record"]
            )
            for job in jobs
        ]

        if sort_keys != sorted(set(sort_keys)):
            raise RuntimeError(
                BRIDGE_ABORT + ": canonical stream order"
            )

        def synthetic_loader(identity):
            if not (
                int(identity["fold"]) == 5
                and int(identity["subject"]) == 9
                and identity["model_variant"] == variant
                and int(identity["checkpoint_seed"]) == seed
            ):
                raise RuntimeError(
                    BRIDGE_ABORT + ": model-member identity"
                )

            loaded[key] += 1

            def release():
                released[key] += 1

            return {
                "model": "SYNTHETIC_MODEL_BUNDLE",
                "ptq_clean_state": "SYNTHETIC_PTQ_STATE",
                "synthetic_member_identity": dict(identity),
                "release": release,
            }

        def synthetic_pair_executor(bundle, job):
            value = synthetic_execute_one(
                bundle=bundle,
                job=job,
            )

            executed[key] += 1
            return value

        outputs = runtime.run_model_member_stream(
            fold=5,
            subject=9,
            model_variant=variant,
            checkpoint_seed=seed,
            pair_jobs=jobs,
            load_model_bundle_hook=synthetic_loader,
            pair_executor_hook=synthetic_pair_executor,
        )

        expected_ids = [
            job["request"]["execution_request_id"]
            for job in jobs
        ]

        output_ids = [
            row["execution_request_id"]
            for row in outputs
        ]

        if not (
            output_ids == expected_ids
            and loaded[key] == 1
            and released[key] == 1
            and executed[key] == len(jobs)
        ):
            raise RuntimeError(
                BRIDGE_ABORT + ": stream lifecycle"
            )

        for row in outputs:
            request_id = row["execution_request_id"]

            if request_id in observed_ids:
                raise RuntimeError(
                    BRIDGE_ABORT + ": repeated output ID"
                )

            observed_ids.add(request_id)

            counters["executed_synthetic_hooks"] += 1
            counters["synthetic_reference_rows"] += row[
                "synthetic_sensor_reference_rows"
            ]
            counters["synthetic_fault_rows"] += row[
                "synthetic_compute_fault_rows"
            ]
            counters["synthetic_overlap_rows"] += row[
                "simultaneous_overlap_rows"
            ]
            counters["synthetic_operator_calls"] += row[
                "synthetic_operator_calls"
            ]
            counters["synthetic_rewindow_calls"] += row[
                "synthetic_rewindow_calls"
            ]
            counters["synthetic_phase5_calls"] += row[
                "synthetic_phase5_sequence_calls"
            ]
            counters[
                "parent_kind_" + row["sensor_parent_kind"]
            ] += 1

        stream_rows.append({
            "runtime_shard_id": runtime.shard_id(
                fold=5,
                subject=9,
                sensor_family=family,
            ),
            "sensor_family": family,
            "model_variant": variant,
            "checkpoint_seed": seed,
            "synthetic_bundle_load_count": loaded[key],
            "synthetic_bundle_release_count": released[key],
            "synthetic_pair_count": len(outputs),
            "execution_authorized": False,
        })

    if not (
        observed_ids == previous_ids
        and len(stream_rows) == 69
        and counters["executed_synthetic_hooks"] == 189
        and counters["synthetic_reference_rows"] == 5310
        and counters["synthetic_fault_rows"] == 13989
        and counters["synthetic_operator_calls"] == 189
        and counters["synthetic_phase5_calls"] == 189
        and counters["parent_kind_source_trial"] > 0
        and counters["parent_kind_stored_window"] > 0
        and sum(loaded.values()) == 69
        and sum(released.values()) == 69
    ):
        raise RuntimeError(
            BRIDGE_ABORT + ": synthetic end-to-end census"
        )

    return {
        "qualification_status":
            "FROZEN_PHASE6J_SYNTHETIC_INTERFACE_BRIDGE_PASS",
        "qualification_scope":
            "REAL_FROZEN_ORCHESTRATION_FUNCTIONS_WITH_SYNTHETIC_FIXTURES",
        "subject": 9,
        "fold": 5,
        "sensor_family_count": 12,
        "compute_strata_covered": 28,
        "prior_frozen_request_ids_reconciled": len(observed_ids),
        "synthetic_member_streams": len(stream_rows),
        "synthetic_bundle_loads": sum(loaded.values()),
        "synthetic_bundle_releases": sum(released.values()),
        "synthetic_hooks_executed":
            counters["executed_synthetic_hooks"],
        "synthetic_counts": dict(counters),
        "streams": stream_rows,
        "real_sensor_inputs_consumed": False,
        "real_labels_consumed": False,
        "real_faults_applied": False,
        "real_checkpoint_loaded": False,
        "real_ptq_state_restoration_qualified": False,
        "actual_model_forward_performed": False,
        "real_phase6h_artifacts_created": False,
        "full_fleet_csc_execution_qualified": False,
        "historical_phase6e_digests_reproduced": False,
        "real_csc_execution_performed": False,
        "execution_authorized": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    """An unconditional hard stop, even if supplied a fake gate."""
    raise RuntimeError(EXECUTION_BLOCK)
