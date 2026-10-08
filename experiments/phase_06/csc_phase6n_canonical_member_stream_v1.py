"""Phase6N full-subject canonical pair-member metadata qualification.

Re-enumerates all frozen model-independent CSC pairs for subject 9,
expands every eligible pair into its exact frozen model-variant/seed
members, creates and validates each Phase6G request, and checks
canonical Phase6J ordering and Phase6K per-shard workload.

This module does NOT read numerical signal/label payloads, load models,
apply faults, parse clean probability records, run a model forward,
write CSC execution artifacts, or authorize CSC execution.

The frozen Phase6K metadata auditor counts source CSV records without
parsing their sample values; this qualification uses that same frozen
metadata-only source-length procedure.

Its newly calculated stream hashes are Phase6N evidence digests,
NOT reconstructed historical Phase6E digests.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import csc_execution_adapter_v1 as adapter
import csc_execution_runtime_v1 as runtime
import csc_outer_executor_v1 as csc
import csc_phase6k_metadata_fleet_auditor_v1 as auditor
import csc_phase6m_pre_forward_orchestrator_v1 as phase6m

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

EXECUTION_BLOCK = "PHASE6N_REAL_CSC_EXECUTION_NOT_AUTHORIZED"
STREAM_ABORT = "PHASE6N_CANONICAL_MEMBER_STREAM_ABORT"

SUBJECT = 9
FOLD = 5

NUMERIC_REL = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_numeric_workload_report_v1.json"
)

WITNESS_REL = (
    "manifests/phase_6m_pre_forward_evidence_v1/"
    "phase6m_canonical_request_synthetic_witness_v1.json"
)

SOURCE_PINS = {
    NUMERIC_REL:
        "e9b977b40dfdeb2a25a9ea041e5c3f22041a58251cfba79b8d9b055df2c725c2",
    WITNESS_REL:
        "7ad0376d72161702cd36e60117c12fa2a85bf4405e5e032681525f40ce023d46",
    "manifests/phase_6m_csc_pre_forward_evidence_freeze_v1.json":
        "a77b437cf669894b4423caa65f9f094f27074b58e862261376c169d04b1a9607",
    "manifests/phase_6l_producer_aware_cache_evidence_freeze_v1.json":
        "d1b355fe0d4ff13d7eebb6a6f14ba7af72f47da06519c9fbce14290c6d9c4ba5",
    "manifests/phase_6k_csc_numeric_evidence_freeze_v1.json":
        "b486264d2caeafc9c899bbd9cb0bf6e758ff952582466cffaacb7c8f511b0bfe",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_execution_adapter_v1.py":
        "2d8d218b0d2214e099dab28b769eeb9fb90d6fa6420b50351b21e54cc4bd1fb5",
    "experiments/phase_06/csc_phase6m_pre_forward_orchestrator_v1.py":
        "c19eb646ca3e5ad584c737af7d4a36383588b8df07430ba3ab1dd368e386af0e",
    "experiments/phase_06/csc_phase6m_canonical_request_witness_v1.py":
        "6b3d8028b31ad52e10a0fb4e4060e0b7bf8866177cced7630c825adeb91fd774",
}

VARIANTS = ("fp32", "ptq_v7")
SEEDS = (42, 123, 2025)

COUNT_FIELDS = (
    "pair_member_count",
    "sensor_reference_member_windows",
    "compute_faulted_member_windows",
    "simultaneous_overlap_member_windows",
    "union_member_windows",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_member_surface(
    pair: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    """Return only frozen-eligible model/seed identities."""

    variants = list(pair["eligible_model_variants"])
    seeds = list(pair["checkpoint_seeds"])

    if (
        not variants
        or len(variants) != len(set(variants))
        or not set(variants).issubset(set(VARIANTS))
        or seeds != list(SEEDS)
    ):
        raise RuntimeError(
            STREAM_ABORT + ": frozen variant/seed surface"
        )

    return tuple(
        (variant, seed)
        for variant in VARIANTS
        if variant in variants
        for seed in SEEDS
    )


def canonical_pair_identity(
    *,
    task: int,
    trial: int,
    kind: str,
    parent_local_index: int,
    severity: str,
    replay_id: str,
) -> dict[str, Any]:
    """Minimal record consumed by the actual frozen sort primitive."""

    if kind not in ("stored_window", "source_trial"):
        raise RuntimeError(STREAM_ABORT + ": parent kind")

    if kind == "source_trial":
        if parent_local_index != -1:
            raise RuntimeError(
                STREAM_ABORT + ": source-trial parent index"
            )
    elif parent_local_index < 0:
        raise RuntimeError(
            STREAM_ABORT + ": stored-window parent index"
        )

    return {
        "task": int(task),
        "trial": int(trial),
        "sensor_parent_kind": str(kind),
        "parent_local_index": int(parent_local_index),
        "severity": str(severity),
        "sensor_replay_id": str(replay_id),
    }


class CompleteSubjectAudit(auditor.SubjectAudit):
    """Capture complete pair metadata while retaining frozen audit checks."""

    def __init__(self, context: Any, subject: int):
        super().__init__(context, subject)
        self.entries_by_family = defaultdict(list)

    def add_pair(
        self,
        key,
        family,
        level,
        kind,
        instance,
        exposed,
    ):
        # Every original frozen Phase6K assertion still runs.
        super().add_pair(
            key,
            family,
            level,
            kind,
            instance,
            exposed,
        )

        subject, task, trial = key

        window_count = int(
            self.ctx.p5[key]["window_count"]
        )

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

        if (
            pair["execution_performed"] is not False
            or pair["sensor_fault_id"] != instance["fault_id"]
            or pair["sensor_replay_id"] != instance["replay_id"]
        ):
            raise RuntimeError(
                STREAM_ABORT + ": canonical pair provenance"
            )

        parent_index = (
            int(exposed[0])
            if kind == "stored_window"
            else -1
        )

        identity = canonical_pair_identity(
            task=task,
            trial=trial,
            kind=kind,
            parent_local_index=parent_index,
            severity=level,
            replay_id=instance["replay_id"],
        )

        self.entries_by_family[family].append(
            (
                identity,
                pair,
                tuple(int(index) for index in exposed),
                window_count,
            )
        )


def _check_sources() -> None:
    for relative, expected in SOURCE_PINS.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(
                STREAM_ABORT + ": frozen file hash: " + relative
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
            phase6m,
            "experiments/phase_06/csc_phase6m_pre_forward_orchestrator_v1.py",
        ),
    ):
        if Path(module.__file__).resolve() != (
            ROOT / relative
        ).resolve():
            raise RuntimeError(
                STREAM_ABORT + ": imported module origin"
            )

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and PRODUCTION_BODY_RELEASED is False
        and runtime.EXECUTION_AUTHORIZED is False
        and phase6m.EXECUTION_AUTHORIZED is False
        and csc.EXECUTION_ENABLED is False
    ):
        raise RuntimeError(
            STREAM_ABORT + ": execution authorization state"
        )


def qualify_complete_subject(
    *,
    subject: int = SUBJECT,
    progress: bool = False,
) -> dict[str, Any]:
    """Expand every selected pair-member for frozen subject 9.

    All outputs are counters, metadata identities and independent
    SHA-256 digests. Request payloads are validated then discarded.
    """

    if subject != SUBJECT:
        raise RuntimeError(
            STREAM_ABORT + ": unqualified subject"
        )

    _check_sources()

    preflight = phase6m.FrozenCSCPreForwardPlan()

    numeric = json.loads(
        (ROOT / NUMERIC_REL).read_text(encoding="utf-8")
    )

    previous = json.loads(
        (ROOT / WITNESS_REL).read_text(encoding="utf-8")
    )

    if not (
        numeric["execution_authorized"] is False
        and previous["execution_authorized"] is False
        and previous["real_csc_execution_performed"] is False
        and previous["subject"] == SUBJECT
        and previous["fold"] == FOLD
        and previous["distinct_pair_witness_count"] == 37
        and len(previous["requests"]) == 189
    ):
        raise RuntimeError(
            STREAM_ABORT + ": historical qualification"
        )

    previous_request_ids = {
        row["execution_request_id"]
        for row in previous["requests"]
    }

    if len(previous_request_ids) != 189:
        raise RuntimeError(
            STREAM_ABORT + ": prior request identity census"
        )

    context = auditor.FrozenContext()

    if context.subject_folds[SUBJECT] != {FOLD}:
        raise RuntimeError(
            STREAM_ABORT + ": frozen subject fold"
        )

    capture = CompleteSubjectAudit(context, SUBJECT)
    full = capture.run()

    if int(full["fold"]) != FOLD:
        raise RuntimeError(
            STREAM_ABORT + ": audited fold"
        )

    frozen_rows = {
        row["sensor_family"]: row
        for row in numeric["inventory"]["rows"]
        if int(row["subject"]) == SUBJECT
    }

    actual_rows = {
        row["sensor_family"]: row
        for row in full["rows"]
    }

    if not (
        len(frozen_rows) == 12
        and len(actual_rows) == 12
        and set(frozen_rows) == set(actual_rows)
        and set(capture.entries_by_family) == set(actual_rows)
    ):
        raise RuntimeError(
            STREAM_ABORT + ": frozen family census"
        )

    validated_bindings = adapter.validate_pinned_bindings()

    stream_results = []
    shard_results = []
    global_counts = Counter()
    strata_seen = set()
    request_ids = set()
    witness_ids_seen = set()
    pair_count_total = 0

    for family in sorted(actual_rows):
        frozen = frozen_rows[family]
        actual = actual_rows[family]

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
            if frozen[field] != actual[field]:
                raise RuntimeError(
                    STREAM_ABORT + ": Phase6K shard mismatch: "
                    + family + ":" + field
                )

        shard_id = actual["runtime_shard_id"]
        route = preflight.shard(shard_id)

        if not (
            route["fold"] == FOLD
            and route["subject"] == SUBJECT
            and route["pair_count"] == actual["pair_count"]
            and route["pair_member_count"] == actual["pair_member_count"]
            and route["execution_authorized"] is False
        ):
            raise RuntimeError(
                STREAM_ABORT + ": Phase6M shard ownership"
            )

        pairs = sorted(
            capture.entries_by_family[family],
            key=lambda item: runtime.canonical_pair_sort_key(item[0]),
        )

        if len(pairs) != int(actual["pair_count"]):
            raise RuntimeError(
                STREAM_ABORT + ": pair cardinality"
            )

        pair_count_total += len(pairs)

        pair_sort_keys = [
            runtime.canonical_pair_sort_key(item[0])
            for item in pairs
        ]

        if len(set(pair_sort_keys)) != len(pair_sort_keys):
            raise RuntimeError(
                STREAM_ABORT + ": duplicate canonical pair order key"
            )

        zero_overlap_pairs = sum(
            int(
                item[1]["temporal_accounting"][
                    "temporal_overlap_window_count"
                ]
            ) == 0
            for item in pairs
        )

        if zero_overlap_pairs != int(
            frozen["zero_overlap_pair_count"]
        ):
            raise RuntimeError(
                STREAM_ABORT + ": zero-overlap pair count"
            )

        family_totals = Counter()
        family_stream_count = 0

        # Full frozen Phase6J canonical member order:
        # variant -> checkpoint seed -> canonical selected pair.
        for variant in VARIANTS:
            for seed in SEEDS:
                stream_hash = hashlib.sha256()
                stream_totals = Counter()
                previous_sort_key = None
                first_request_id = None
                last_request_id = None

                cache = preflight._cache_by_member[
                    (FOLD, SUBJECT, variant, seed)
                ]

                if cache["execution_authorized"] is not False:
                    raise RuntimeError(
                        STREAM_ABORT + ": cache execution state"
                    )

                if cache["clean_cache_id"] not in (
                    route["prospective_candidate_clean_cache_ids"]
                ):
                    raise RuntimeError(
                        STREAM_ABORT + ": cache owner"
                    )

                for identity, pair, exposed, window_count in pairs:
                    if (variant, seed) not in validate_member_surface(pair):
                        continue

                    strata_seen.add(
                        int(pair["compute_stratum_index"])
                    )

                    sort_record = {
                        **identity,
                        "model_variant": variant,
                        "checkpoint_seed": seed,
                    }

                    sort_key = runtime.canonical_pair_member_sort_key(
                        sort_record
                    )

                    if (
                        previous_sort_key is not None
                        and sort_key <= previous_sort_key
                    ):
                        raise RuntimeError(
                            STREAM_ABORT + ": noncanonical member order"
                        )

                    previous_sort_key = sort_key

                    request = adapter.build_execution_request(
                        pair_metadata=pair,
                        sensor_exposed_window_indices=exposed,
                        trial_window_count=window_count,
                        model_variant=variant,
                        checkpoint_seed=seed,
                        validated_bindings=validated_bindings,
                    )

                    adapter.validate_execution_request(request)

                    if not (
                        request["execution_enabled"] is False
                        and request["model_variant"] == variant
                        and request["checkpoint_seed"] == seed
                        and request["sensor_fault_id"]
                        == pair["sensor_fault_id"]
                        and request["sensor_replay_id"]
                        == pair["sensor_replay_id"]
                        and request["sensor_parent_kind"]
                        == pair["sensor_parent_kind"]
                    ):
                        raise RuntimeError(
                            STREAM_ABORT + ": request metadata mismatch"
                        )

                    request_id = request["execution_request_id"]

                    if request_id in request_ids:
                        raise RuntimeError(
                            STREAM_ABORT + ": duplicate execution request"
                        )

                    request_ids.add(request_id)

                    if request_id in previous_request_ids:
                        witness_ids_seen.add(request_id)

                    compute = [
                        int(index)
                        for index in request[
                            "compute_execution_window_indices"
                        ]
                    ]

                    overlap = [
                        int(index)
                        for index in request[
                            "simultaneous_overlap_window_indices"
                        ]
                    ]

                    exposed_set = set(exposed)
                    compute_set = set(compute)

                    if overlap != sorted(
                        exposed_set & compute_set
                    ):
                        raise RuntimeError(
                            STREAM_ABORT + ": temporal overlap"
                        )

                    union_count = len(
                        exposed_set | compute_set
                    )

                    if union_count != (
                        len(exposed)
                        + len(compute)
                        - len(overlap)
                    ):
                        raise RuntimeError(
                            STREAM_ABORT + ": temporal union"
                        )

                    if request["persistence"] == (
                        "transient_one_inference"
                    ):
                        if not (
                            len(compute) == 1
                            and len(overlap) == 1
                        ):
                            raise RuntimeError(
                                STREAM_ABORT + ": transient geometry"
                            )

                    sensor_cache_id = (
                        runtime.sensor_reference_cache_id(
                            fold=FOLD,
                            subject=SUBJECT,
                            task=identity["task"],
                            trial=identity["trial"],
                            sensor_parent_kind=identity[
                                "sensor_parent_kind"
                            ],
                            parent_local_index=identity[
                                "parent_local_index"
                            ],
                            sensor_fault_id=pair["sensor_fault_id"],
                            sensor_replay_id=pair["sensor_replay_id"],
                            model_variant=variant,
                            checkpoint_seed=seed,
                        )
                    )

                    token = {
                        "schema_version":
                            "phase6n_canonical_member_token_v1",
                        "runtime_shard_id": shard_id,
                        "execution_request_id": request_id,
                        "sensor_fault_id": pair["sensor_fault_id"],
                        "sensor_replay_id": pair["sensor_replay_id"],
                        "model_variant": variant,
                        "checkpoint_seed": seed,
                        "compute_stratum_index":
                            int(pair["compute_stratum_index"]),
                        "clean_cache_id": cache["clean_cache_id"],
                        "expected_producer_sha256":
                            cache["expected_executor_sha256"],
                        "sensor_reference_cache_id":
                            sensor_cache_id,
                        "sensor_reference_window_count":
                            len(exposed),
                        "compute_execution_window_count":
                            len(compute),
                        "simultaneous_overlap_window_count":
                            len(overlap),
                        "union_window_count": union_count,
                        "execution_authorized": False,
                    }

                    stream_hash.update(canonical_line(token))

                    if first_request_id is None:
                        first_request_id = request_id

                    last_request_id = request_id

                    stream_totals["pair_member_count"] += 1
                    stream_totals[
                        "sensor_reference_member_windows"
                    ] += len(exposed)
                    stream_totals[
                        "compute_faulted_member_windows"
                    ] += len(compute)
                    stream_totals[
                        "simultaneous_overlap_member_windows"
                    ] += len(overlap)
                    stream_totals[
                        "union_member_windows"
                    ] += union_count

                if stream_totals["pair_member_count"] == 0:
                    continue

                family_stream_count += 1
                family_totals.update(stream_totals)

                stream_results.append({
                    "runtime_shard_id": shard_id,
                    "sensor_family": family,
                    "fold": FOLD,
                    "subject": SUBJECT,
                    "model_variant": variant,
                    "checkpoint_seed": seed,
                    "clean_cache_id": cache["clean_cache_id"],
                    "producer_kind": cache["producer_kind"],
                    "pair_member_count":
                        stream_totals["pair_member_count"],
                    "sensor_reference_member_windows":
                        stream_totals[
                            "sensor_reference_member_windows"
                        ],
                    "compute_faulted_member_windows":
                        stream_totals[
                            "compute_faulted_member_windows"
                        ],
                    "simultaneous_overlap_member_windows":
                        stream_totals[
                            "simultaneous_overlap_member_windows"
                        ],
                    "union_member_windows":
                        stream_totals["union_member_windows"],
                    "first_execution_request_id": first_request_id,
                    "last_execution_request_id": last_request_id,
                    "phase6n_new_stream_sha256":
                        stream_hash.hexdigest(),
                    "execution_authorized": False,
                })

        for field in COUNT_FIELDS:
            if int(family_totals[field]) != int(frozen[field]):
                raise RuntimeError(
                    STREAM_ABORT + ": expanded frozen workload: "
                    + family + ":" + field
                )

        global_counts.update(family_totals)

        shard_results.append({
            "runtime_shard_id": shard_id,
            "sensor_family": family,
            "sensor_parent_kind":
                frozen["sensor_parent_kind"],
            "pair_count": len(pairs),
            "pair_member_count":
                family_totals["pair_member_count"],
            "stream_count": family_stream_count,
            "zero_overlap_pair_count": zero_overlap_pairs,
            "structural_omission_count":
                frozen["structural_omission_count"],
            "frozen_phase6k_descriptor_sha256":
                actual["new_phase6k_descriptor_sha256"],
            "all_variant_seed_members_qualified": True,
            "execution_authorized": False,
        })

        if progress:
            print(
                "CANONICAL_FAMILY_PASS="
                + family
                + " pairs="
                + str(len(pairs))
                + " members="
                + str(family_totals["pair_member_count"])
                + " streams="
                + str(family_stream_count),
                flush=True,
            )

    if pair_count_total != int(
        full["totals"]["pair_count"]
    ):
        raise RuntimeError(
            STREAM_ABORT + ": subject pair count"
        )

    if global_counts["pair_member_count"] != int(
        full["totals"]["pair_member_count"]
    ):
        raise RuntimeError(
            STREAM_ABORT + ": subject member count"
        )

    if strata_seen != set(range(28)):
        raise RuntimeError(
            STREAM_ABORT + ": 28-stratum coverage"
        )

    if witness_ids_seen != previous_request_ids:
        missing = len(
            previous_request_ids - witness_ids_seen
        )
        raise RuntimeError(
            STREAM_ABORT
            + ": prior witness reconciliation missing="
            + str(missing)
        )

    if len(stream_results) > 72:
        raise RuntimeError(
            STREAM_ABORT + ": unexpected stream census"
        )

    if len(shard_results) != 12:
        raise RuntimeError(
            STREAM_ABORT + ": shard count"
        )

    return {
        "qualification_status":
            "FULL_SUBJECT_9_CANONICAL_PAIR_MEMBER_STREAM_PASS",
        "qualification_scope":
            "ALL_FROZEN_SUBJECT_9_PAIRS_AND_ELIGIBLE_MEMBERS_METADATA_ONLY",
        "fold": FOLD,
        "subject": SUBJECT,
        "shard_count": len(shard_results),
        "complete_model_independent_pair_count":
            pair_count_total,
        "complete_pair_member_request_count":
            global_counts["pair_member_count"],
        "unique_validated_request_count":
            len(request_ids),
        "frozen_phase6k_shard_descriptors_matched": 12,
        "covered_compute_strata": sorted(strata_seen),
        "previous_phase6m_request_ids_reconciled":
            len(witness_ids_seen),
        "previous_phase6m_request_ids_expected": 189,
        "model_member_stream_count": len(stream_results),
        "expanded_member_window_totals":
            dict(global_counts),
        "shards": shard_results,
        "streams": stream_results,
        "phase6n_digest_provenance":
            "NEW_INDEPENDENT_CANONICAL_METADATA_TOKEN_STREAM_V1",
        "original_phase6e_digests_reproduced": False,
        "full_732_shard_canonical_enumeration_qualified": False,
        "actual_sensor_signal_samples_parsed": False,
        "actual_label_arrays_parsed": False,
        "model_loaded": False,
        "compute_fault_mutation_performed": False,
        "model_forward_performed": False,
        "real_csc_execution_performed": False,
        "production_body_released": False,
        "execution_authorized": False,
    }


def execute_shard(
    *,
    shard_id_value: str,
    dataset_root: str | Path,
    phase5_output_root: str | Path,
    output_root: str | Path,
    gate_path: str | Path | None = None,
) -> None:
    """Hard stop; no real CSC execution body exists here."""

    raise RuntimeError(EXECUTION_BLOCK)
