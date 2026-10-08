"""Prospective Phase6M CSC pre-forward orchestration.

This module prepares frozen subject-family shard ownership and validates
existing Phase5 clean-cache artifacts through the provenance-pinned
Phase6L v2 adapter.

It does not enumerate actual pair members, load checkpoints, read IMU
signals or labels, inject faults, execute model forwards, or authorize
a production CSC shard.

The next prospective implementation must bind actual canonical
pair-member enumeration and the frozen Phase6J orchestration primitives
before any execution qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import csc_execution_runtime_v1 as runtime
import csc_phase6l_producer_aware_cache_v1 as phase6l_v1
import csc_phase6l_producer_aware_cache_v2 as phase6l_v2

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
REAL_CSC_BODY_RELEASED = False

PREFLIGHT_ABORT = "PHASE6M_PREFLIGHT_ABORT"
UNKNOWN_SHARD = "PHASE6M_UNKNOWN_FROZEN_SHARD"
EXECUTION_BLOCK = "PHASE6M_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

NUMERIC_PATH = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_numeric_workload_report_v1.json"
)
CROSSWALK_PATH = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_shard_clean_cache_ownership_crosswalk_v1.json"
)
PHASE6K_FREEZE_PATH = (
    "manifests/phase_6k_csc_numeric_evidence_freeze_v1.json"
)
PHASE6L_FREEZE_PATH = (
    "manifests/phase_6l_producer_aware_cache_evidence_freeze_v1.json"
)

PINS = {
    NUMERIC_PATH:
        "e9b977b40dfdeb2a25a9ea041e5c3f22041a58251cfba79b8d9b055df2c725c2",
    CROSSWALK_PATH:
        "abc0c7c2230ba77781b19d5306a0fb3a4dd6916df49ba2cc8f0ffb93aee66312",
    PHASE6K_FREEZE_PATH:
        "b486264d2caeafc9c899bbd9cb0bf6e758ff952582466cffaacb7c8f511b0bfe",
    PHASE6L_FREEZE_PATH:
        "d1b355fe0d4ff13d7eebb6a6f14ba7af72f47da06519c9fbce14290c6d9c4ba5",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_phase6l_producer_aware_cache_v1.py":
        "bfb034367519521b93a8ae043dd181e5d185526815728b9b21150960f1c06a14",
    "experiments/phase_06/csc_phase6l_producer_aware_cache_v2.py":
        "6fc1b63a1358f8019fb413f29f9b77a6d5eeebdb8af725fd180bd0cbabd3f755",
}

EXPECTED_TOTALS = {
    "pair_count": 4237835,
    "pair_member_count": 21793038,
    "sensor_reference_member_windows": 35167107,
    "compute_faulted_member_windows": 411540372,
    "simultaneous_overlap_member_windows": 19926021,
    "union_member_windows": 426781458,
    "zero_overlap_pair_count": 1018215,
    "structural_omission_count": 2104,
}

EXPECTED_MODEL_MEMBERS = {
    (variant, seed)
    for variant in ("fp32", "ptq_v7")
    for seed in (42, 123, 2025)
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _load_bound(relative: str) -> dict[str, Any]:
    path = ROOT / relative

    if file_sha256(path) != PINS[relative]:
        raise RuntimeError(PREFLIGHT_ABORT + ": " + relative)

    return json.loads(path.read_text(encoding="utf-8"))


def _member_key(
    row: dict[str, Any],
    *,
    fold: int,
    subject: int,
) -> tuple[int, int, str, int]:
    """Bind compact cache references to their verified parent shard.

    The frozen crosswalk intentionally keeps fold and subject on the
    enclosing shard rather than repeating them in every cache record.
    Any optional duplicate identity inside a member must agree with
    the enclosing shard or validation fails closed.
    """
    for field, expected in (
        ("fold", fold),
        ("subject", subject),
    ):
        if (
            field in row
            and int(row[field]) != int(expected)
        ):
            raise RuntimeError(
                PREFLIGHT_ABORT + ": contradictory " + field
            )

    return (
        int(fold),
        int(subject),
        str(row["model_variant"]),
        int(row["checkpoint_seed"]),
    )


class FrozenCSCPreForwardPlan:
    """Prepare all frozen shards without executing a CSC model forward."""

    def __init__(self) -> None:
        for relative, expected in PINS.items():
            if file_sha256(ROOT / relative) != expected:
                raise RuntimeError(PREFLIGHT_ABORT + ": source binding")

        if not (
            EXECUTION_AUTHORIZED is False
            and MODEL_FORWARD_AUTHORIZED is False
            and REAL_CSC_BODY_RELEASED is False
            and runtime.EXECUTION_AUTHORIZED is False
            and phase6l_v2.EXECUTION_AUTHORIZED is False
        ):
            raise RuntimeError(PREFLIGHT_ABORT + ": execution state")

        numeric = _load_bound(NUMERIC_PATH)
        crosswalk = _load_bound(CROSSWALK_PATH)
        phase6k = _load_bound(PHASE6K_FREEZE_PATH)
        phase6l = _load_bound(PHASE6L_FREEZE_PATH)

        if not (
            numeric["status"] == "PHASE6K_NUMERIC_WORKLOAD_AUDIT_PASS"
            and numeric["execution_authorized"] is False
            and numeric["original_phase6e_digests_reproduced"] is False
            and numeric["phase6k_independent_inventory_sha256"]
            == "3d94b6123524896fbd8220a8cf2de46c82149f028cb3dc95adad666bdae842d2"
            and crosswalk["status"] == "METADATA_OWNERSHIP_AUDIT_PASS"
            and crosswalk["execution_authorized"] is False
            and crosswalk[
                "candidate_cache_references_not_all_eligible_pair_members"
            ] is True
            and phase6k["freeze_scope"] == "EVIDENCE_ONLY"
            and phase6k["execution_authorized"] is False
            and phase6l["freeze_scope"] == "EVIDENCE_ONLY"
            and phase6l["execution_authorized"] is False
        ):
            raise RuntimeError(PREFLIGHT_ABORT + ": frozen evidence")

        self._validator = (
            phase6l_v2.FrozenProducerAwareCacheValidator(
                phase5_output_root="/unused/preflight/metadata"
            )
        )

        self._cache_by_member = {}
        self._cache_by_subject = defaultdict(list)

        for key in sorted(
            self._validator._inventory["plan_by_member"]
        ):
            selected = self._validator.select(
                fold=key[0],
                subject=key[1],
                model_variant=key[2],
                checkpoint_seed=key[3],
            )

            if (
                selected["execution_authorized"] is not False
                or key in self._cache_by_member
            ):
                raise RuntimeError(PREFLIGHT_ABORT + ": cache member")

            self._cache_by_member[key] = selected
            self._cache_by_subject[key[1]].append(selected)

        self._shards = {}
        self._ordered_shard_ids = []
        self._totals = Counter()

        numeric_rows = numeric["inventory"]["rows"]
        crosswalk_rows = crosswalk["shard_cache_ownership"]

        if len(numeric_rows) != 732 or len(crosswalk_rows) != 732:
            raise RuntimeError(PREFLIGHT_ABORT + ": shard count")

        crosswalk_by_id = {
            item["runtime_shard_id"]: item
            for item in crosswalk_rows
        }

        if len(crosswalk_by_id) != 732:
            raise RuntimeError(PREFLIGHT_ABORT + ": crosswalk duplicates")

        family_census = Counter()
        subject_census = Counter()
        parent_census = Counter()

        for row in sorted(
            numeric_rows,
            key=lambda item: (
                int(item["fold"]),
                int(item["subject"]),
                str(item["sensor_family"]),
            ),
        ):
            shard_id = str(row["runtime_shard_id"])
            fold = int(row["fold"])
            subject = int(row["subject"])
            family = str(row["sensor_family"])

            if shard_id in self._shards:
                raise RuntimeError(PREFLIGHT_ABORT + ": duplicate shard")

            if runtime.shard_id(
                fold=fold,
                subject=subject,
                sensor_family=family,
            ) != shard_id:
                raise RuntimeError(PREFLIGHT_ABORT + ": runtime shard ID")

            cross = crosswalk_by_id.get(shard_id)

            if cross is None:
                raise RuntimeError(PREFLIGHT_ABORT + ": absent crosswalk")

            for field in (
                "fold",
                "subject",
                "sensor_family",
                "pair_count",
                "pair_member_count",
            ):
                if cross[field] != row[field]:
                    raise RuntimeError(
                        PREFLIGHT_ABORT + ": crosswalk disagreement"
                    )

            caches = self._cache_by_subject[subject]

            if len(caches) != 6:
                raise RuntimeError(PREFLIGHT_ABORT + ": cache ownership")

            if {cache["fold"] for cache in caches} != {fold}:
                raise RuntimeError(PREFLIGHT_ABORT + ": fold mismatch")

            owned = {
                cache["clean_cache_id"]
                for cache in caches
            }

            declared = {
                cache["clean_cache_id"]
                for cache in cross["candidate_clean_caches"]
            }

            if (
                len(owned) != 6
                or len(declared) != 6
                or declared != owned
            ):
                raise RuntimeError(PREFLIGHT_ABORT + ": cache IDs")

            for member in cross["candidate_clean_caches"]:
                key = _member_key(member, fold=fold, subject=subject)
                selected = self._cache_by_member.get(key)

                if not selected or (
                    selected["clean_cache_id"]
                    != member["clean_cache_id"]
                    or selected["producer_kind"]
                    != member["producer_kind"]
                    or selected["expected_executor_sha256"]
                    != member["expected_executor_sha256"]
                ):
                    raise RuntimeError(
                        PREFLIGHT_ABORT + ": producer binding"
                    )

            for field in EXPECTED_TOTALS:
                self._totals[field] += int(row[field])

            sensor_count = int(
                row["sensor_reference_member_windows"]
            )
            compute_count = int(
                row["compute_faulted_member_windows"]
            )
            overlap_count = int(
                row["simultaneous_overlap_member_windows"]
            )
            union_count = int(
                row["union_member_windows"]
            )

            if union_count != (
                sensor_count + compute_count - overlap_count
            ):
                raise RuntimeError(PREFLIGHT_ABORT + ": temporal union")

            descriptor = {
                "runtime_shard_id": shard_id,
                "fold": fold,
                "subject": subject,
                "sensor_family": family,
                "sensor_parent_kind": row["sensor_parent_kind"],
                "pair_count": int(row["pair_count"]),
                "pair_member_count": int(row["pair_member_count"]),
                "zero_overlap_pair_count": int(
                    row["zero_overlap_pair_count"]
                ),
                "source_structural_omissions": int(
                    row["structural_omission_count"]
                ),
                "prospective_candidate_cache_count": 6,
                "prospective_candidate_clean_cache_ids": sorted(owned),
                "pair_member_eligibility_resolved": False,
                "execution_authorized": False,
            }

            self._shards[shard_id] = descriptor
            self._ordered_shard_ids.append(shard_id)

            family_census[family] += 1
            subject_census[subject] += 1
            parent_census[row["sensor_parent_kind"]] += 1

        if not (
            len(self._shards) == 732
            and len(self._cache_by_member) == 366
            and len(subject_census) == 61
            and set(subject_census.values()) == {12}
            and len(family_census) == 12
            and set(family_census.values()) == {61}
            and parent_census
            == {"stored_window": 305, "source_trial": 427}
            and dict(self._totals) == EXPECTED_TOTALS
        ):
            raise RuntimeError(PREFLIGHT_ABORT + ": global census")

        for subject, caches in self._cache_by_subject.items():
            if {
                (item["model_variant"], item["checkpoint_seed"])
                for item in caches
            } != EXPECTED_MODEL_MEMBERS:
                raise RuntimeError(
                    PREFLIGHT_ABORT + ": six-member surface"
                )

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def clean_cache_count(self) -> int:
        return len(self._cache_by_member)

    def shard(self, shard_id_value: str) -> dict[str, Any]:
        if not isinstance(shard_id_value, str):
            raise RuntimeError(UNKNOWN_SHARD)

        record = self._shards.get(shard_id_value)

        if record is None:
            raise RuntimeError(UNKNOWN_SHARD)

        return {
            **record,
            "prospective_candidate_clean_cache_ids": list(
                record["prospective_candidate_clean_cache_ids"]
            ),
        }

    def ordered_shards(self) -> list[dict[str, Any]]:
        return [
            self.shard(shard_id)
            for shard_id in self._ordered_shard_ids
        ]

    def validate_all_existing_clean_caches(
        self,
        *,
        phase5_output_root: str | Path,
        progress: bool = False,
    ) -> dict[str, Any]:
        """Rehash every real C0 artifact exactly once, without row parsing."""
        validator = (
            phase6l_v2.FrozenProducerAwareCacheValidator(
                phase5_output_root=phase5_output_root
            )
        )

        observed = []
        census = Counter()
        windows = 0

        for index, key in enumerate(
            sorted(self._cache_by_member),
            start=1,
        ):
            expected = self._cache_by_member[key]

            result = validator.validate(
                fold=key[0],
                subject=key[1],
                model_variant=key[2],
                checkpoint_seed=key[3],
                requested_cache_id=expected["clean_cache_id"],
            )

            if not (
                result["clean_cache_id"]
                == expected["clean_cache_id"]
                and result["expected_executor_sha256"]
                == expected["expected_executor_sha256"]
                and result["existing_cache_hash_validator_passed"] is True
                and result["execution_authorized"] is False
            ):
                raise RuntimeError(
                    PREFLIGHT_ABORT + ": real cache validator"
                )

            census[result["producer_kind"]] += 1
            windows += int(result["window_count"])

            observed.append({
                "clean_cache_id": result["clean_cache_id"],
                "fold": key[0],
                "subject": key[1],
                "model_variant": key[2],
                "checkpoint_seed": key[3],
                "producer_kind": result["producer_kind"],
                "producer_sha256":
                    result["expected_executor_sha256"],
                "window_count": int(result["window_count"]),
                "existing_cache_hash_validation_passed": True,
            })

            if progress and index % 61 == 0:
                print(
                    f"REAL_CLEAN_CACHES_VALIDATED={index}/366",
                    flush=True,
                )

        if not (
            len(observed) == 366
            and census == {"canary": 1, "fleet": 365}
            and windows == 1642980
        ):
            raise RuntimeError(
                PREFLIGHT_ABORT + ": real cache census"
            )

        return {
            "real_clean_cache_validation": "PASS",
            "validated_clean_cache_count": 366,
            "validated_output_file_hash_count": 732,
            "clean_cache_window_cardinality": windows,
            "producer_counts": dict(census),
            "rows": observed,
            "execution_authorized": False,
        }

    def numerical_totals(self) -> dict[str, int]:
        return dict(self._totals)


def execute_shard(
    *,
    shard_id_value: str,
    dataset_root: str | Path,
    phase5_output_root: str | Path,
    output_root: str | Path,
    gate_path: str | Path | None = None,
) -> None:
    """Deliberately unavailable real execution entrypoint.

    Even a caller-supplied authorization-like gate cannot cause data
    loading or a model forward in this prospective Phase6M stage.
    """
    raise RuntimeError(EXECUTION_BLOCK)
