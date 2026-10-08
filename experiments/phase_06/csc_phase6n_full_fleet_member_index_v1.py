"""Phase6N complete-fleet canonical CSC member metadata index.

Reuses the frozen Phase6K subject metadata auditor and the previously
qualified Phase6N complete-subject capture adapter.

Every selected frozen pair is visited, and every frozen eligible
variant/seed member is counted and incorporated into a deterministic
new metadata-only stream digest.

For scalability, Phase6G complete execution requests are built and
validated for frozen compute-stratum representatives per subject,
rather than for all 21,793,038 pair-members.

Therefore this stage qualifies full-fleet canonical member
IDENTITY/ORDER/OWNERSHIP/WORKLOAD, but does not claim validation of
every full-fleet Phase6G request payload.

No raw sensor sample values, labels, model predictions or probability
rows are read. No checkpoint, real fault or model forward is executed.

All generated hashes are NEW Phase6N metadata digests and do not
reproduce historical Phase6E serialized stream digests.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import csc_execution_adapter_v1 as adapter
import csc_execution_runtime_v1 as runtime
import csc_phase6k_metadata_fleet_auditor_v1 as auditor
import csc_phase6m_pre_forward_orchestrator_v1 as phase6m
import csc_phase6n_canonical_member_stream_v1 as phase6n

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

FLEET_ABORT = "PHASE6N_FULL_FLEET_CANONICAL_ABORT"
EXECUTION_BLOCK = "PHASE6N_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

SUBJECT9_REL = (
    "manifests/phase_6n_canonical_evidence_v1/"
    "phase6n_complete_subject9_canonical_member_stream_audit_v1.json"
)

NUMERIC_REL = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_numeric_workload_report_v1.json"
)

SOURCE_PINS = {
    SUBJECT9_REL:
        "d8bbff4006a7d78c11f20c134255ac3a0a9ab7f2081fba20af6861db96bffe2f",
    NUMERIC_REL:
        "e9b977b40dfdeb2a25a9ea041e5c3f22041a58251cfba79b8d9b055df2c725c2",
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
    "experiments/phase_06/csc_phase6n_canonical_member_stream_v1.py":
        "55ae23c5f84431a9b41c34fcfd82ff7fe3ca3c3690163b07c11299d7e9b504c7",
}

EXPECTED_GLOBAL = {
    "pair_count": 4237835,
    "pair_member_count": 21793038,
    "sensor_reference_member_windows": 35167107,
    "compute_faulted_member_windows": 411540372,
    "simultaneous_overlap_member_windows": 19926021,
    "union_member_windows": 426781458,
    "zero_overlap_pair_count": 1018215,
    "structural_omission_count": 2104,
}

MEMBER_FIELDS = (
    "pair_member_count",
    "sensor_reference_member_windows",
    "compute_faulted_member_windows",
    "simultaneous_overlap_member_windows",
    "union_member_windows",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_line(value: Mapping[str, Any]) -> bytes:
    return phase6n.canonical_line(value)


def _check_inputs() -> None:
    for relative, expected in SOURCE_PINS.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(
                FLEET_ABORT + ": pinned source: " + relative
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
            phase6n,
            "experiments/phase_06/csc_phase6n_canonical_member_stream_v1.py",
        ),
    ):
        if Path(module.__file__).resolve() != (
            ROOT / relative
        ).resolve():
            raise RuntimeError(
                FLEET_ABORT + ": unexpected imported module"
            )

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and PRODUCTION_BODY_RELEASED is False
        and runtime.EXECUTION_AUTHORIZED is False
        and phase6m.EXECUTION_AUTHORIZED is False
        and phase6n.EXECUTION_AUTHORIZED is False
    ):
        raise RuntimeError(
            FLEET_ABORT + ": prohibited execution state"
        )


def pair_fingerprint(
    *,
    identity: Mapping[str, Any],
    pair: Mapping[str, Any],
    sensor_exposure_count: int,
    compute_count: int,
    overlap_count: int,
    union_count: int,
) -> bytes:
    """One domain-separated, source-independent pair token digest."""

    token = {
        "schema_version":
            "phase6n_full_fleet_pair_fingerprint_v1",
        "canonical_pair_identity": dict(identity),
        "sensor_fault_id": str(pair["sensor_fault_id"]),
        "sensor_replay_id": str(pair["sensor_replay_id"]),
        "compute_stratum_index":
            int(pair["compute_stratum_index"]),
        "compute_sampling_instance_id":
            str(pair["compute_coordinate"]["sampling_instance_id"]),
        "eligible_model_variants":
            list(pair["eligible_model_variants"]),
        "checkpoint_seeds":
            list(pair["checkpoint_seeds"]),
        "sensor_exposure_window_count":
            int(sensor_exposure_count),
        "compute_execution_window_count":
            int(compute_count),
        "temporal_overlap_window_count":
            int(overlap_count),
        "temporal_union_window_count":
            int(union_count),
    }

    return hashlib.sha256(canonical_line(token)).digest()


def prepare_pair(
    entry: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Validate a selected pair once, before variant/seed expansion."""

    identity, pair, exposed, window_count = entry

    members = phase6n.validate_member_surface(pair)
    variants = frozenset(variant for variant, _ in members)

    timing = pair["temporal_accounting"]

    sensor = int(timing["sensor_exposed_window_count"])
    compute = int(timing["compute_active_window_count"])
    overlap = int(timing["temporal_overlap_window_count"])
    union = int(timing["sensor_compute_union_window_count"])

    if not (
        sensor == len(exposed)
        and 0 <= overlap <= min(sensor, compute)
        and union == sensor + compute - overlap
        and compute >= 1
        and int(window_count) >= 1
        and len(members) == len(variants) * 3
    ):
        raise RuntimeError(
            FLEET_ABORT + ": invalid frozen pair geometry"
        )

    if pair["persistence"] == "transient_one_inference":
        if compute != 1 or overlap != 1:
            raise RuntimeError(
                FLEET_ABORT + ": transient geometry"
            )

    pair_digest = pair_fingerprint(
        identity=identity,
        pair=pair,
        sensor_exposure_count=sensor,
        compute_count=compute,
        overlap_count=overlap,
        union_count=union,
    )

    return (
        identity,
        pair,
        exposed,
        window_count,
        variants,
        pair_digest,
        sensor,
        compute,
        overlap,
        union,
    )


def stream_prefix(
    *,
    shard_id: str,
    variant: str,
    seed: int,
    cache_id: str,
    producer_sha256: str,
) -> bytes:
    """Domain-separated header so streams cannot share identities."""

    return canonical_line({
        "schema_version":
            "phase6n_full_fleet_member_stream_header_v1",
        "shard_id": shard_id,
        "model_variant": str(variant),
        "checkpoint_seed": int(seed),
        "clean_cache_id": str(cache_id),
        "producer_sha256": str(producer_sha256),
        "execution_authorized": False,
    })


def qualify_full_fleet(
    *,
    progress: bool = False,
) -> dict[str, Any]:
    """Re-enumerate 61 subjects and index every eligible member."""

    _check_inputs()

    preflight = phase6m.FrozenCSCPreForwardPlan()
    context = auditor.FrozenContext()

    numeric = json.loads(
        (ROOT / NUMERIC_REL).read_text(encoding="utf-8")
    )
    subject9 = json.loads(
        (ROOT / SUBJECT9_REL).read_text(encoding="utf-8")
    )

    if not (
        numeric["status"] == "PHASE6K_NUMERIC_WORKLOAD_AUDIT_PASS"
        and numeric["execution_authorized"] is False
        and numeric["original_phase6e_digests_reproduced"] is False
        and preflight.shard_count == 732
        and preflight.clean_cache_count == 366
        and subject9["qualification_status"]
        == "FULL_SUBJECT_9_CANONICAL_PAIR_MEMBER_STREAM_PASS"
        and subject9["complete_model_independent_pair_count"] == 38107
        and subject9["complete_pair_member_request_count"] == 195768
        and subject9["model_member_stream_count"] == 72
        and subject9["execution_authorized"] is False
    ):
        raise RuntimeError(FLEET_ABORT + ": frozen evidence")

    frozen_rows = {
        row["runtime_shard_id"]: row
        for row in numeric["inventory"]["rows"]
    }

    if len(frozen_rows) != 732:
        raise RuntimeError(FLEET_ABORT + ": duplicate frozen shard")

    subject9_streams = {
        (
            row["sensor_family"],
            row["model_variant"],
            int(row["checkpoint_seed"]),
        ): row
        for row in subject9["streams"]
    }

    if len(subject9_streams) != 72:
        raise RuntimeError(FLEET_ABORT + ": prior stream census")

    validated_bindings = adapter.validate_pinned_bindings()

    global_counts = Counter()
    subjects = []
    shard_rows = []
    stream_rows = []
    sampled_request_ids = set()

    for subject_number, subject in enumerate(
        sorted(context.subject_folds),
        start=1,
    ):
        folds = context.subject_folds[subject]

        if len(folds) != 1:
            raise RuntimeError(FLEET_ABORT + ": ambiguous subject fold")

        fold = next(iter(folds))
        capture = phase6n.CompleteSubjectAudit(
            context,
            subject,
        )
        actual = capture.run()

        if (
            int(actual["fold"]) != fold
            or len(actual["rows"]) != 12
            or len(capture.entries_by_family) != 12
        ):
            raise RuntimeError(FLEET_ABORT + ": subject census")

        actual_by_family = {
            row["sensor_family"]: row
            for row in actual["rows"]
        }

        if len(actual_by_family) != 12:
            raise RuntimeError(FLEET_ABORT + ": repeated family")

        subject_totals = Counter()
        sampled_by_stratum = {}
        subject_shard_count = 0
        subject_stream_count = 0

        for family in sorted(actual_by_family):
            row = actual_by_family[family]
            shard_id = row["runtime_shard_id"]
            frozen = frozen_rows.get(shard_id)

            if frozen is None:
                raise RuntimeError(
                    FLEET_ABORT + ": absent frozen shard"
                )

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
                if frozen[field] != row[field]:
                    raise RuntimeError(
                        FLEET_ABORT
                        + ": Phase6K shard mismatch: "
                        + shard_id
                        + ":"
                        + field
                    )

            route = preflight.shard(shard_id)

            if not (
                route["fold"] == fold
                and route["subject"] == subject
                and route["pair_count"] == row["pair_count"]
                and route["pair_member_count"] == row["pair_member_count"]
                and route["execution_authorized"] is False
            ):
                raise RuntimeError(
                    FLEET_ABORT + ": invalid shard ownership"
                )

            pairs = sorted(
                capture.entries_by_family[family],
                key=lambda item: runtime.canonical_pair_sort_key(
                    item[0]
                ),
            )

            if len(pairs) != int(row["pair_count"]):
                raise RuntimeError(
                    FLEET_ABORT + ": selected pair count"
                )

            prepared = []
            previous_pair_key = None
            zero_overlap = 0

            for entry in pairs:
                identity = entry[0]

                pair_key = runtime.canonical_pair_sort_key(
                    identity
                )

                if (
                    previous_pair_key is not None
                    and pair_key <= previous_pair_key
                ):
                    raise RuntimeError(
                        FLEET_ABORT + ": noncanonical pair order"
                    )

                previous_pair_key = pair_key

                item = prepare_pair(entry)
                prepared.append(item)

                (
                    _identity,
                    pair,
                    exposed,
                    window_count,
                    variants,
                    pair_digest,
                    sensor,
                    compute,
                    overlap,
                    union,
                ) = item

                if overlap == 0:
                    zero_overlap += 1

                stratum = int(pair["compute_stratum_index"])

                if not 0 <= stratum < 28:
                    raise RuntimeError(
                        FLEET_ABORT + ": unknown compute stratum"
                    )

                sampled_by_stratum.setdefault(
                    stratum,
                    (
                        shard_id,
                        family,
                        pair,
                        exposed,
                        window_count,
                    ),
                )

            if zero_overlap != int(row["zero_overlap_pair_count"]):
                raise RuntimeError(
                    FLEET_ABORT + ": zero-overlap pair census"
                )

            family_totals = Counter()
            family_stream_count = 0

            for variant in phase6n.VARIANTS:
                for seed in phase6n.SEEDS:
                    cache = preflight._cache_by_member.get(
                        (fold, subject, variant, seed)
                    )

                    if not cache or (
                        cache["execution_authorized"] is not False
                        or cache["clean_cache_id"] not in
                        route["prospective_candidate_clean_cache_ids"]
                    ):
                        raise RuntimeError(
                            FLEET_ABORT + ": cache producer ownership"
                        )

                    stream_hash = hashlib.sha256()

                    stream_hash.update(
                        stream_prefix(
                            shard_id=shard_id,
                            variant=variant,
                            seed=seed,
                            cache_id=cache["clean_cache_id"],
                            producer_sha256=cache[
                                "expected_executor_sha256"
                            ],
                        )
                    )

                    stream_totals = Counter()

                    for (
                        identity,
                        pair,
                        exposed,
                        window_count,
                        variants,
                        pair_digest,
                        sensor,
                        compute,
                        overlap,
                        union,
                    ) in prepared:
                        if variant not in variants:
                            continue

                        # Exactly one eligible pair-member for this
                        # frozen variant/seed stream.
                        stream_hash.update(pair_digest)

                        stream_totals["pair_member_count"] += 1
                        stream_totals[
                            "sensor_reference_member_windows"
                        ] += sensor
                        stream_totals[
                            "compute_faulted_member_windows"
                        ] += compute
                        stream_totals[
                            "simultaneous_overlap_member_windows"
                        ] += overlap
                        stream_totals[
                            "union_member_windows"
                        ] += union

                    if not stream_totals["pair_member_count"]:
                        continue

                    family_totals.update(stream_totals)
                    family_stream_count += 1

                    descriptor = {
                        "runtime_shard_id": shard_id,
                        "fold": fold,
                        "subject": subject,
                        "sensor_family": family,
                        "model_variant": variant,
                        "checkpoint_seed": seed,
                        "clean_cache_id": cache["clean_cache_id"],
                        "producer_kind": cache["producer_kind"],
                        "expected_executor_sha256":
                            cache["expected_executor_sha256"],
                        **dict(stream_totals),
                        "phase6n_new_fleet_member_stream_sha256":
                            stream_hash.hexdigest(),
                        "full_phase6g_requests_built_for_this_stream":
                            False,
                        "execution_authorized": False,
                    }

                    if subject == 9:
                        prior = subject9_streams.get(
                            (family, variant, seed)
                        )

                        if prior is None:
                            raise RuntimeError(
                                FLEET_ABORT + ": missing subject9 stream"
                            )

                        for field in MEMBER_FIELDS:
                            if descriptor[field] != prior[field]:
                                raise RuntimeError(
                                    FLEET_ABORT
                                    + ": subject9 independent stream "
                                    + field
                                )

                        if (
                            descriptor["clean_cache_id"]
                            != prior["clean_cache_id"]
                        ):
                            raise RuntimeError(
                                FLEET_ABORT + ": subject9 cache"
                            )

                    stream_rows.append(descriptor)

            for field in MEMBER_FIELDS:
                if family_totals[field] != int(row[field]):
                    raise RuntimeError(
                        FLEET_ABORT
                        + ": expanded shard total: "
                        + shard_id
                        + ":"
                        + field
                    )

            shard_rows.append({
                "runtime_shard_id": shard_id,
                "fold": fold,
                "subject": subject,
                "sensor_family": family,
                "sensor_parent_kind": row["sensor_parent_kind"],
                "model_independent_pair_count":
                    int(row["pair_count"]),
                "pair_member_count":
                    family_totals["pair_member_count"],
                "model_member_stream_count":
                    family_stream_count,
                "zero_overlap_pair_count":
                    zero_overlap,
                "structural_omission_count":
                    int(row["structural_omission_count"]),
                "frozen_phase6k_descriptor_sha256":
                    row["new_phase6k_descriptor_sha256"],
                "execution_authorized": False,
            })

            subject_totals.update(family_totals)
            subject_totals["pair_count"] += len(pairs)
            subject_totals["zero_overlap_pair_count"] += (
                zero_overlap
            )
            subject_totals["structural_omission_count"] += int(
                row["structural_omission_count"]
            )

            subject_shard_count += 1
            subject_stream_count += family_stream_count

            # Release per-family prepared references before
            # processing the next subject-family group.
            del prepared
            del pairs

        if set(sampled_by_stratum) != set(range(28)):
            raise RuntimeError(
                FLEET_ABORT + ": missing subject compute strata"
            )

        sample_requests = 0

        for stratum in sorted(sampled_by_stratum):
            (
                shard_id,
                family,
                pair,
                exposed,
                window_count,
            ) = sampled_by_stratum[stratum]

            for variant, seed in phase6n.validate_member_surface(pair):
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
                    and int(request["checkpoint_seed"]) == seed
                ):
                    raise RuntimeError(
                        FLEET_ABORT + ": sampled Phase6G request"
                    )

                request_id = request["execution_request_id"]

                if request_id in sampled_request_ids:
                    raise RuntimeError(
                        FLEET_ABORT + ": duplicate sampled request ID"
                    )

                sampled_request_ids.add(request_id)
                sample_requests += 1

                cache = preflight._cache_by_member.get(
                    (fold, subject, variant, seed)
                )

                if not cache or (
                    cache["clean_cache_id"] not in
                    preflight.shard(shard_id)[
                        "prospective_candidate_clean_cache_ids"
                    ]
                ):
                    raise RuntimeError(
                        FLEET_ABORT + ": sampled request cache"
                    )

        if subject == 9:
            if not (
                subject_totals["pair_count"] == 38107
                and subject_totals["pair_member_count"] == 195768
                and subject_stream_count == 72
            ):
                raise RuntimeError(
                    FLEET_ABORT + ": subject9 baseline mismatch"
                )

        if not (
            subject_shard_count == 12
            and subject_totals["pair_count"]
                == int(actual["totals"]["pair_count"])
            and subject_totals["pair_member_count"]
                == int(actual["totals"]["pair_member_count"])
        ):
            raise RuntimeError(
                FLEET_ABORT + ": subject workload"
            )

        global_counts.update(subject_totals)

        subjects.append({
            "fold": fold,
            "subject": subject,
            "shard_count": subject_shard_count,
            "model_independent_pair_count":
                subject_totals["pair_count"],
            "canonical_pair_member_count":
                subject_totals["pair_member_count"],
            "canonical_member_stream_count":
                subject_stream_count,
            "phase6g_sample_requests_validated": sample_requests,
            "all_28_compute_strata_sampled": True,
            "execution_authorized": False,
        })

        # Release this subject's potentially large pair capture.
        del capture

        if progress:
            print(
                "FULL_FLEET_SUBJECT="
                + str(subject_number)
                + "/61"
                + " subject="
                + str(subject)
                + " pairs="
                + str(subject_totals["pair_count"])
                + " members="
                + str(subject_totals["pair_member_count"])
                + " sampled_requests="
                + str(sample_requests),
                flush=True,
            )

    if not (
        len(subjects) == 61
        and len(shard_rows) == 732
        and len({row["runtime_shard_id"] for row in shard_rows})
            == 732
        and len(stream_rows) <= 4392
        and global_counts == EXPECTED_GLOBAL
    ):
        raise RuntimeError(
            FLEET_ABORT + ": global Phase6K workload discrepancy"
        )

    by_shard = Counter(
        row["runtime_shard_id"]
        for row in stream_rows
    )

    if set(by_shard) != set(frozen_rows):
        raise RuntimeError(
            FLEET_ABORT + ": incomplete shard stream coverage"
        )

    if not (
        len(sampled_request_ids)
        == sum(
            row["phase6g_sample_requests_validated"]
            for row in subjects
        )
    ):
        raise RuntimeError(
            FLEET_ABORT + ": sampled request census"
        )

    return {
        "qualification_status":
            "FULL_61_SUBJECT_CANONICAL_MEMBER_INDEX_PASS",
        "qualification_scope":
            "COMPLETE_METADATA_MEMBER_ORDER_ELIGIBILITY_AND_CACHE_OWNERSHIP",
        "subject_count": len(subjects),
        "shard_count": len(shard_rows),
        "model_member_stream_count": len(stream_rows),
        "model_independent_pair_count":
            global_counts["pair_count"],
        "canonical_pair_member_count":
            global_counts["pair_member_count"],
        "global_frozen_workload_totals": dict(global_counts),
        "phase6g_sample_request_count":
            len(sampled_request_ids),
        "phase6g_full_fleet_request_payloads_all_validated":
            False,
        "full_pair_member_identity_index_qualified": True,
        "frozen_phase6k_shard_census_reconciled": True,
        "subject9_prior_stream_census_reconciled": True,
        "canonical_stream_hash_schema":
            "NEW_PHASE6N_FULL_FLEET_PAIR_FINGERPRINT_AND_STREAM_HEADER_V1",
        "subjects": subjects,
        "shards": shard_rows,
        "streams": stream_rows,
        "historical_phase6e_digests_reproduced": False,
        "raw_sensor_signal_values_parsed": False,
        "label_arrays_parsed": False,
        "prediction_rows_parsed": False,
        "existing_phase5_cache_files_rehashed_in_this_stage":
            False,
        "real_model_loaded": False,
        "real_sensor_fault_applied": False,
        "real_compute_fault_injected": False,
        "model_forward_performed": False,
        "real_csc_execution_performed": False,
        "production_body_released": False,
        "execution_authorized": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
