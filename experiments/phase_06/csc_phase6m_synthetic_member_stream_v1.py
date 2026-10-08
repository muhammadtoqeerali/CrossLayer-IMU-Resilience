"""Phase6M metadata-only model-member stream qualification.

Groups previously verified frozen subject-9 execution-request IDs
by sensor-family shard, model variant and checkpoint seed.

Calls the exact frozen Phase6J model-member stream helper using
synthetic loader/executor hooks. The hooks never create models,
read signals, alter tensors, inject compute faults or write real
CSC output artifacts.

This is a stream-lifecycle and identity qualification, NOT
canonical full-fleet pair ordering or real CSC execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import csc_execution_runtime_v1 as runtime

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

STREAM_ABORT = "PHASE6M_FROZEN_MEMBER_STREAM_ABORT"
EXECUTION_BLOCK = "PHASE6M_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

PREFLIGHT_REL = (
    "manifests/phase_6m_pre_forward_evidence_v1/"
    "phase6m_prospective_732_shard_pre_forward_qualification_v1.json"
)

WITNESS_REL = (
    "manifests/phase_6m_pre_forward_evidence_v1/"
    "phase6m_canonical_request_synthetic_witness_v1.json"
)

SOURCE_PINS = {
    PREFLIGHT_REL:
        "40645db9f41bdad7f364d577b0eeca162bc4b7d381f2675418f19bb7fce4ce33",
    WITNESS_REL:
        "7ad0376d72161702cd36e60117c12fa2a85bf4405e5e032681525f40ce023d46",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_phase6m_pre_forward_orchestrator_v1.py":
        "c19eb646ca3e5ad584c737af7d4a36383588b8df07430ba3ab1dd368e386af0e",
    "experiments/phase_06/csc_phase6m_canonical_request_witness_v1.py":
        "6b3d8028b31ad52e10a0fb4e4060e0b7bf8866177cced7630c825adeb91fd774",
    "manifests/phase_6l_producer_aware_cache_evidence_freeze_v1.json":
        "d1b355fe0d4ff13d7eebb6a6f14ba7af72f47da06519c9fbce14290c6d9c4ba5",
}

EXPECTED_VARIANTS = ("fp32", "ptq_v7")
EXPECTED_SEEDS = (42, 123, 2025)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


class SyntheticMemberStreamPlan:
    """Verified witness grouping with no production execution interface."""

    def __init__(self):
        for relative, expected in SOURCE_PINS.items():
            if file_sha256(ROOT / relative) != expected:
                raise RuntimeError(
                    STREAM_ABORT + ": pinned SHA256 mismatch: " + relative
                )

        if Path(runtime.__file__).resolve() != (
            ROOT / "experiments/phase_06/csc_execution_runtime_v1.py"
        ).resolve():
            raise RuntimeError(STREAM_ABORT + ": runtime module path")

        if not (
            EXECUTION_AUTHORIZED is False
            and MODEL_FORWARD_AUTHORIZED is False
            and PRODUCTION_BODY_RELEASED is False
            and runtime.EXECUTION_AUTHORIZED is False
        ):
            raise RuntimeError(STREAM_ABORT + ": execution enabled")

        preflight = json.loads(
            (ROOT / PREFLIGHT_REL).read_text(encoding="utf-8")
        )

        witness = json.loads(
            (ROOT / WITNESS_REL).read_text(encoding="utf-8")
        )

        if not (
            preflight["qualification_status"]
            == "FROZEN_SHARD_AND_CACHE_PREFLIGHT_PASS"
            and preflight["execution_authorized"] is False
            and preflight["real_csc_execution_performed"] is False
            and preflight["model_forward_performed"] is False
            and preflight["shard_count"] == 732
            and preflight["clean_cache_count"] == 366
            and witness["qualification_status"]
            == "FROZEN_SUBJECT_REQUEST_WITNESSES_PASS"
            and witness["execution_authorized"] is False
            and witness["real_csc_execution_performed"] is False
            and witness["model_forward_performed"] is False
            and witness["subject"] == 9
            and witness["fold"] == 5
            and witness["frozen_shard_descriptor_sha256_matches"] == 12
            and witness["covered_compute_strata"] == list(range(28))
            and len(witness["covered_sensor_families"]) == 12
        ):
            raise RuntimeError(STREAM_ABORT + ": evidence qualification")

        self._shards = {}

        for row in preflight["shards"]:
            shard_id = row["runtime_shard_id"]

            if shard_id in self._shards:
                raise RuntimeError(STREAM_ABORT + ": duplicate shard")

            self._shards[shard_id] = row

        if len(self._shards) != 732:
            raise RuntimeError(STREAM_ABORT + ": shard census")

        self._caches = {}

        for row in preflight["real_cache_validation"]["rows"]:
            key = (
                int(row["fold"]),
                int(row["subject"]),
                str(row["model_variant"]),
                int(row["checkpoint_seed"]),
            )

            if key in self._caches:
                raise RuntimeError(STREAM_ABORT + ": duplicate cache")

            self._caches[key] = row

        if len(self._caches) != 366:
            raise RuntimeError(STREAM_ABORT + ": cache census")

        self._requests = {}
        self._groups = defaultdict(list)

        for position, row in enumerate(witness["requests"]):
            family = str(row["sensor_family"])
            variant = str(row["model_variant"])
            seed = int(row["checkpoint_seed"])
            request_id = str(row["execution_request_id"])

            if not (
                len(request_id) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in request_id
                )
            ):
                raise RuntimeError(STREAM_ABORT + ": request ID")

            if request_id in self._requests:
                raise RuntimeError(STREAM_ABORT + ": duplicate request ID")

            if variant not in EXPECTED_VARIANTS:
                raise RuntimeError(STREAM_ABORT + ": model variant")

            if seed not in EXPECTED_SEEDS:
                raise RuntimeError(STREAM_ABORT + ": checkpoint seed")

            if not (
                row["execution_authorized"] is False
                and row["synthetic_orchestration_passed"] is True
                and int(row["compute_stratum_index"]) in range(28)
                and bool(row["zero_temporal_overlap"])
                == (int(row["overlap_window_count"]) == 0)
                and int(row["reference_window_count"]) >= 1
                and int(row["compute_window_count"]) >= 1
            ):
                raise RuntimeError(STREAM_ABORT + ": witness semantics")

            shard_id = runtime.shard_id(
                fold=5,
                subject=9,
                sensor_family=family,
            )

            shard = self._shards.get(shard_id)

            if not shard or (
                int(shard["fold"]) != 5
                or int(shard["subject"]) != 9
                or shard["sensor_family"] != family
                or shard["execution_authorized"] is not False
            ):
                raise RuntimeError(STREAM_ABORT + ": shard ownership")

            cache = self._caches.get((5, 9, variant, seed))

            if not cache or (
                cache["clean_cache_id"] != row["clean_cache_id"]
                or cache["producer_kind"] != row["producer_kind"]
                or cache["clean_cache_id"] not in
                shard["prospective_candidate_clean_cache_ids"]
            ):
                raise RuntimeError(STREAM_ABORT + ": cache ownership")

            selected = {
                **row,
                "shard_id": shard_id,
                "fold": 5,
                "subject": 9,
                "witness_position": position,
                "execution_authorized": False,
            }

            self._requests[request_id] = selected
            self._groups[(family, variant, seed)].append(selected)

        self._witness = witness

        if not (
            len(self._requests) == 189
            and len(self._groups) >= 12
            and len(self._groups) <= 72
            and sum(
                len(rows)
                for rows in self._groups.values()
            ) == 189
            and sum(
                int(row["reference_window_count"])
                for row in self._requests.values()
            ) == 5310
            and sum(
                int(row["compute_window_count"])
                for row in self._requests.values()
            ) == 13989
            and {
                int(row["compute_stratum_index"])
                for row in self._requests.values()
            } == set(range(28))
            and {
                row["sensor_family"]
                for row in self._requests.values()
            } == set(witness["covered_sensor_families"])
        ):
            raise RuntimeError(STREAM_ABORT + ": witness census")

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def member_stream_count(self) -> int:
        return len(self._groups)

    def request(self, request_id: str) -> dict[str, Any]:
        row = self._requests.get(request_id)

        if row is None:
            raise RuntimeError(STREAM_ABORT + ": unknown witness request")

        return dict(row)

    def exercise_synthetic_member_streams(self) -> dict[str, Any]:
        """Use frozen Phase6J stream lifecycle with synthetic-only hooks."""
        loaded = Counter()
        released = Counter()
        executed = Counter()
        stream_rows = []
        output_request_ids = set()

        for key in sorted(self._groups):
            family, variant, seed = key
            rows = self._groups[key]

            expected_ids = [
                row["execution_request_id"]
                for row in rows
            ]

            jobs = [
                {
                    "request": {
                        "execution_enabled": False,
                        "model_variant": variant,
                        "checkpoint_seed": seed,
                        "execution_request_id":
                            row["execution_request_id"],
                    },
                    "witness": row,
                }
                for row in rows
            ]

            # These are synthetic stream-identity proxies. Full Phase6G
            # execution-request validation was performed in the
            # preceding frozen subject-9 witness qualification.
            # This stage does not claim full canonical pair sorting.
            def synthetic_loader(identity):
                if not (
                    int(identity["fold"]) == 5
                    and int(identity["subject"]) == 9
                    and identity["model_variant"] == variant
                    and int(identity["checkpoint_seed"]) == seed
                ):
                    raise RuntimeError(
                        STREAM_ABORT + ": stream loader identity"
                    )

                loaded[key] += 1

                def synthetic_release():
                    released[key] += 1

                return {
                    "synthetic_stream_identity": dict(identity),
                    "release": synthetic_release,
                }

            def synthetic_pair_executor(bundle, job):
                identity = bundle["synthetic_stream_identity"]
                request = job["request"]
                witness_row = job["witness"]

                if not (
                    request["execution_enabled"] is False
                    and request["model_variant"]
                    == identity["model_variant"]
                    and int(request["checkpoint_seed"])
                    == int(identity["checkpoint_seed"])
                    and witness_row["sensor_family"] == family
                    and witness_row["model_variant"] == variant
                    and int(witness_row["checkpoint_seed"]) == seed
                ):
                    raise RuntimeError(
                        STREAM_ABORT + ": cross-stream request"
                    )

                executed[key] += 1

                return {
                    "execution_request_id":
                        request["execution_request_id"],
                    "shard_id": witness_row["shard_id"],
                    "synthetic_only": True,
                    "execution_authorized": False,
                }

            outputs = runtime.run_model_member_stream(
                fold=5,
                subject=9,
                model_variant=variant,
                checkpoint_seed=seed,
                pair_jobs=jobs,
                load_model_bundle_hook=synthetic_loader,
                pair_executor_hook=synthetic_pair_executor,
            )

            actual_ids = [
                item["execution_request_id"]
                for item in outputs
            ]

            if not (
                actual_ids == expected_ids
                and loaded[key] == 1
                and released[key] == 1
                and executed[key] == len(rows)
                and all(
                    item["synthetic_only"] is True
                    and item["execution_authorized"] is False
                    for item in outputs
                )
            ):
                raise RuntimeError(
                    STREAM_ABORT + ": synthetic stream lifecycle"
                )

            for request_id in actual_ids:
                if request_id in output_request_ids:
                    raise RuntimeError(
                        STREAM_ABORT + ": repeated output request"
                    )
                output_request_ids.add(request_id)

            stream_rows.append({
                "shard_id": runtime.shard_id(
                    fold=5,
                    subject=9,
                    sensor_family=family,
                ),
                "sensor_family": family,
                "model_variant": variant,
                "checkpoint_seed": seed,
                "synthetic_bundle_load_count": 1,
                "synthetic_bundle_release_count": 1,
                "synthetic_request_count": len(rows),
                "execution_authorized": False,
            })

        if output_request_ids != set(self._requests):
            raise RuntimeError(
                STREAM_ABORT + ": missing stream outputs"
            )

        return {
            "qualification_status":
                "ALL_WITNESS_MODEL_MEMBER_STREAMS_SYNTHETIC_PASS",
            "subject": 9,
            "fold": 5,
            "member_stream_count": len(stream_rows),
            "synthetic_bundle_load_count": sum(loaded.values()),
            "synthetic_bundle_release_count": sum(released.values()),
            "synthetic_pair_execution_hook_count":
                sum(executed.values()),
            "distinct_request_ids_seen": len(output_request_ids),
            "canonical_full_fleet_order_qualified": False,
            "actual_model_loaded": False,
            "real_csc_execution_performed": False,
            "execution_authorized": False,
            "streams": stream_rows,
        }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
