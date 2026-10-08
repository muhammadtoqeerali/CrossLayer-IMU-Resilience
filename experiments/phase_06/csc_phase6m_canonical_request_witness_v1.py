"""Phase6M canonical CSC metadata-request qualification witness.

The frozen Phase6K metadata auditor enumerates one complete subject.
Its 12 shard descriptor hashes must exactly reproduce the independent
Phase6K audit.

Representative frozen pairs covering all 28 compute strata, all
12 families, both parent kinds and persistence modes are expanded
into actual Phase6G pre-forward requests using the frozen variant
eligibility and checkpoint seed surface.

The resulting requests exercise frozen Phase6J synthetic orchestration.
There is no real signal input, checkpoint load, fault mutation,
model forward or CSC execution authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import csc_execution_adapter_v1 as adapter
import csc_execution_runtime_v1 as runtime
import csc_outer_executor_v1 as csc
import csc_phase6k_metadata_fleet_auditor_v1 as auditor
import csc_phase6m_pre_forward_orchestrator_v1 as phase6m

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False

EXECUTION_BLOCK = "PHASE6M_WITNESS_REAL_EXECUTION_NOT_AUTHORIZED"

SUBJECT = 9
FOLD = 5

NUMERIC_REL = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_numeric_workload_report_v1.json"
)

NUMERIC_SHA = (
    "e9b977b40dfdeb2a25a9ea041e5c3f22041a58251cfba79b8d9b055df2c725c2"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_tags(
    *,
    family: str,
    kind: str,
    persistence: str,
    stratum: int,
    overlap: int,
    eligible_variants: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    tags = [
        ("stratum", int(stratum)),
        ("family", str(family)),
        ("parent_persistence", str(kind), str(persistence)),
    ]

    if overlap == 0:
        tags.append(("zero_overlap", str(kind)))

    for variant in eligible_variants:
        tags.append(("eligible_variant", str(variant)))

    return tuple(tags)


class CapturingSubjectAudit(auditor.SubjectAudit):
    """Capture only representative pairs while auditing the full subject."""

    def __init__(self, context: Any, subject: int):
        super().__init__(context, subject)
        self.examples_by_tag: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add_pair(
        self,
        key,
        family,
        level,
        kind,
        instance,
        exposed,
    ):
        # Preserve exact frozen Phase6K counting and descriptor hashing.
        super().add_pair(
            key,
            family,
            level,
            kind,
            instance,
            exposed,
        )

        index, stratum = csc.assign_compute_stratum(
            sensor_instance=instance,
            phase6d=self.ctx.phase6d,
        )

        persistence = str(stratum["persistence"])

        # The complete metadata pair is required only for a new
        # coverage tag. No need to retain millions of records.
        preliminary_tags = (
            ("stratum", int(index)),
            ("family", str(family)),
            ("parent_persistence", str(kind), persistence),
        )

        if all(
            tag in self.examples_by_tag
            for tag in preliminary_tags
        ):
            # Zero-overlap witnesses are also required, so do not
            # skip persistent pairs without inspecting their timing.
            if persistence != "persistent_from_onset_until_trial_end":
                return

        subject, task, trial = key
        window_count = int(self.ctx.p5[key]["window_count"])

        pair = csc.derive_pair_metadata(
            sensor_parent_kind=kind,
            sensor_instance=instance,
            sensor_exposed_window_indices=exposed,
            fold=self.fold,
            subject=subject,
            task=task,
            trial=trial,
            trial_window_count=window_count,
            validated=self.ctx.validated,
        )

        timing = pair["temporal_accounting"]
        overlap = int(timing["temporal_overlap_window_count"])

        tags = _sample_tags(
            family=family,
            kind=kind,
            persistence=persistence,
            stratum=int(index),
            overlap=overlap,
            eligible_variants=tuple(
                pair["eligible_model_variants"]
            ),
        )

        if all(tag in self.examples_by_tag for tag in tags):
            return

        parent_local_index = (
            int(exposed[0])
            if kind == "stored_window"
            else -1
        )

        witness = {
            "fold": self.fold,
            "subject": subject,
            "task": task,
            "trial": trial,
            "sensor_family": str(family),
            "severity": str(level),
            "sensor_parent_kind": str(kind),
            "parent_local_index": parent_local_index,
            "sensor_fault_id": str(instance["fault_id"]),
            "sensor_replay_id": str(instance["replay_id"]),
            "sensor_exposed_window_indices": tuple(
                int(value) for value in exposed
            ),
            "trial_window_count": window_count,
            "pair_metadata": pair,
        }

        for tag in tags:
            self.examples_by_tag.setdefault(tag, witness)


def _reference_summary(index: int, _window: Any):
    return {
        "input_sha256": f"synthetic-input-{index}",
        "output_sha256": f"synthetic-ref-output-{index}",
        "output_nonfinite": False,
        "output_values": [0.0],
        "output_float32_hex": ["00000000"],
        "softmax_values": [1.0],
    }


def _fault_summaries(indices, windows, request):
    assert len(indices) == len(windows)

    return [
        {
            "input_sha256": f"synthetic-fault-input-{index}",
            "faulted_output_sha256": f"synthetic-fault-output-{index}",
            "faulted_output_nonfinite": False,
            "faulted_output_values": [0.0],
            "faulted_output_float32_hex": ["00000000"],
            "faulted_softmax_values": [1.0],
            "mutation": {"active": True},
        }
        for index in indices
    ]


def qualify_subject(
    *,
    subject: int = SUBJECT,
) -> dict[str, Any]:
    """Metadata-only full-subject audit plus synthetic request witnesses."""
    if subject != SUBJECT:
        raise ValueError("only frozen subject 9 witness is qualified")

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and runtime.EXECUTION_AUTHORIZED is False
        and phase6m.EXECUTION_AUTHORIZED is False
        and csc.EXECUTION_ENABLED is False
    ):
        raise RuntimeError(EXECUTION_BLOCK)

    if digest(ROOT / NUMERIC_REL) != NUMERIC_SHA:
        raise RuntimeError("PHASE6M_FROZEN_NUMERIC_SHA_MISMATCH")

    preflight = phase6m.FrozenCSCPreForwardPlan()
    context = auditor.FrozenContext()

    assert context.subject_folds[SUBJECT] == {FOLD}

    audited = CapturingSubjectAudit(context, SUBJECT)
    actual = audited.run()

    expected_numeric = json.loads(
        (ROOT / NUMERIC_REL).read_text(encoding="utf-8")
    )

    expected_rows = {
        row["sensor_family"]: row
        for row in expected_numeric["inventory"]["rows"]
        if int(row["subject"]) == SUBJECT
    }

    observed_rows = {
        row["sensor_family"]: row
        for row in actual["rows"]
    }

    assert len(observed_rows) == len(expected_rows) == 12

    for family, row in observed_rows.items():
        frozen = expected_rows[family]

        for field in (
            "runtime_shard_id",
            "fold",
            "subject",
            "sensor_family",
            "sensor_parent_kind",
            "pair_count",
            "pair_member_count",
            "structural_omission_count",
            "zero_overlap_pair_count",
            "sensor_reference_member_windows",
            "compute_faulted_member_windows",
            "simultaneous_overlap_member_windows",
            "union_member_windows",
            "new_phase6k_descriptor_sha256",
        ):
            if row[field] != frozen[field]:
                raise RuntimeError(
                    "PHASE6M_SHARD_DESCRIPTOR_MISMATCH: "
                    + family + ":" + field
                )

        matched = preflight.shard(row["runtime_shard_id"])

        assert matched["pair_count"] == row["pair_count"]
        assert matched["pair_member_count"] == row["pair_member_count"]
        assert matched["execution_authorized"] is False

    tags = audited.examples_by_tag

    if {
        int(key[1])
        for key in tags
        if key[0] == "stratum"
    } != set(range(28)):
        raise RuntimeError("PHASE6M_STRATUM_COVERAGE_INCOMPLETE")

    if len({
        key[1] for key in tags if key[0] == "family"
    }) != 12:
        raise RuntimeError("PHASE6M_FAMILY_COVERAGE_INCOMPLETE")

    bindings = adapter.validate_pinned_bindings()

    # Deduplicate examples selected through multiple coverage tags.
    examples = {
        witness["sensor_fault_id"]: witness
        for witness in tags.values()
    }

    requests = []
    counts = Counter()

    for witness in sorted(
        examples.values(),
        key=lambda value: (
            value["sensor_family"],
            value["task"],
            value["trial"],
            value["sensor_fault_id"],
        ),
    ):
        pair = witness["pair_metadata"]
        exposed = list(witness["sensor_exposed_window_indices"])

        if pair["execution_performed"] is not False:
            raise RuntimeError("frozen pair execution flag changed")

        eligible = list(pair["eligible_model_variants"])
        seeds = list(pair["checkpoint_seeds"])

        if seeds != [42, 123, 2025]:
            raise RuntimeError("checkpoint-seed surface changed")

        for variant in eligible:
            for seed in seeds:
                key = (
                    FOLD,
                    SUBJECT,
                    variant,
                    int(seed),
                )

                clean = preflight._cache_by_member[key]

                request = adapter.build_execution_request(
                    pair_metadata=pair,
                    sensor_exposed_window_indices=exposed,
                    trial_window_count=witness["trial_window_count"],
                    model_variant=variant,
                    checkpoint_seed=seed,
                    validated_bindings=bindings,
                )

                adapter.validate_execution_request(request)

                if not (
                    request["execution_enabled"] is False
                    and request["model_variant"] == variant
                    and request["checkpoint_seed"] == seed
                ):
                    raise RuntimeError(
                        "PHASE6M_EXECUTION_REQUEST_IDENTITY_INVALID"
                    )

                compute = [
                    int(value)
                    for value in request[
                        "compute_execution_window_indices"
                    ]
                ]

                overlap = [
                    int(value)
                    for value in request[
                        "simultaneous_overlap_window_indices"
                    ]
                ]

                expected_overlap = sorted(set(exposed) & set(compute))

                if overlap != expected_overlap:
                    raise RuntimeError(
                        "PHASE6M_TEMPORAL_OVERLAP_MISMATCH"
                    )

                sensor_cache_id = runtime.sensor_reference_cache_id(
                    fold=FOLD,
                    subject=SUBJECT,
                    task=witness["task"],
                    trial=witness["trial"],
                    sensor_parent_kind=witness["sensor_parent_kind"],
                    parent_local_index=witness["parent_local_index"],
                    sensor_fault_id=witness["sensor_fault_id"],
                    sensor_replay_id=witness["sensor_replay_id"],
                    model_variant=variant,
                    checkpoint_seed=seed,
                )

                windows_by_index = {
                    index: f"synthetic-window-{index}"
                    for index in (set(exposed) | set(compute))
                }

                events = []

                def reference_hook(index, window):
                    events.append(("reference", int(index)))
                    return _reference_summary(index, window)

                def compute_hook(indices, windows, request_arg):
                    events.append(("compute_sequence", len(indices)))
                    return _fault_summaries(
                        indices,
                        windows,
                        request_arg,
                    )

                synthetic = runtime.synthetic_execute_pair_member(
                    request=request,
                    shard_id_value=runtime.shard_id(
                        fold=FOLD,
                        subject=SUBJECT,
                        sensor_family=witness["sensor_family"],
                    ),
                    fold=FOLD,
                    subject=SUBJECT,
                    task=witness["task"],
                    trial=witness["trial"],
                    parent_local_index=witness["parent_local_index"],
                    phase5_clean_cache_id=clean["clean_cache_id"],
                    sensor_reference_cache_id_value=sensor_cache_id,
                    windows_by_index=windows_by_index,
                    hooks=runtime.SyntheticHooks(
                        reference_forward=reference_hook,
                        fault_sequence=compute_hook,
                    ),
                )

                assert len(synthetic["sensor_reference_rows"]) == len(exposed)
                assert len(synthetic["csc_fault_rows"]) == len(compute)
                assert len(events) == len(exposed) + 1
                assert all(event[0] == "reference" for event in events[:-1])
                assert events[-1][0] == "compute_sequence"

                actual_kinds = [
                    row["reference_kind"]
                    for row in synthetic["csc_fault_rows"]
                ]

                expected_kinds = [
                    (
                        "phase6_sensor_reference"
                        if index in set(exposed)
                        else "phase5_clean_cache"
                    )
                    for index in compute
                ]

                assert actual_kinds == expected_kinds

                assert synthetic["pair_member"][
                    "zero_temporal_overlap"
                ] is (len(overlap) == 0)

                counts["requests"] += 1
                counts["sensor_reference_rows"] += len(exposed)
                counts["compute_fault_rows"] += len(compute)

                requests.append({
                    "execution_request_id":
                        request["execution_request_id"],
                    "sensor_family":
                        witness["sensor_family"],
                    "sensor_parent_kind":
                        witness["sensor_parent_kind"],
                    "compute_stratum_index":
                        pair["compute_stratum_index"],
                    "persistence":
                        pair["persistence"],
                    "model_variant":
                        variant,
                    "checkpoint_seed":
                        seed,
                    "clean_cache_id":
                        clean["clean_cache_id"],
                    "producer_kind":
                        clean["producer_kind"],
                    "reference_window_count":
                        len(exposed),
                    "compute_window_count":
                        len(compute),
                    "overlap_window_count":
                        len(overlap),
                    "zero_temporal_overlap":
                        len(overlap) == 0,
                    "synthetic_orchestration_passed":
                        True,
                    "execution_authorized":
                        False,
                })

    if not requests:
        raise RuntimeError("PHASE6M_NO_SYNTHETIC_REQUESTS")

    if len({
        row["execution_request_id"] for row in requests
    }) != len(requests):
        raise RuntimeError("PHASE6M_DUPLICATE_REQUEST_ID")

    return {
        "qualification_status":
            "FROZEN_SUBJECT_REQUEST_WITNESSES_PASS",
        "subject": SUBJECT,
        "fold": FOLD,
        "full_subject_shard_count": 12,
        "full_subject_pair_count": int(actual["totals"]["pair_count"]),
        "full_subject_pair_member_count":
            int(actual["totals"]["pair_member_count"]),
        "frozen_shard_descriptor_sha256_matches": 12,
        "covered_compute_strata": sorted({
            row["compute_stratum_index"] for row in requests
        }),
        "covered_sensor_families": sorted({
            row["sensor_family"] for row in requests
        }),
        "coverage_tag_count": len(tags),
        "distinct_pair_witness_count": len(examples),
        "request_counts": dict(counts),
        "requests": requests,
        "execution_authorized": False,
        "model_forward_performed": False,
        "real_csc_execution_performed": False,
        "historical_phase6e_digests_reproduced": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
