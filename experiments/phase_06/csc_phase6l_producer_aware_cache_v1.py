"""Prospective Phase6L producer-aware Phase5 C0 cache resolution.

Metadata and existing-cache validation only. This module never authorizes
CSC execution, regenerates clean caches, or loads model checkpoints.
Frozen Phase6J code and Phase6E/Phase6K evidence are not modified.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False

PLAN_REL = "manifests/phase_5e_compute_fi_outer_execution_plan_v1.json"
BINDING_REL = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_mixed_producer_cache_binding_audit_v1.json"
)
FREEZE_REL = "manifests/phase_6k_csc_numeric_evidence_freeze_v1.json"
PHASE6J_REL = "experiments/phase_06/csc_execution_runtime_v1.py"

PLAN_SHA = "95aecd14b4aa8dce70492f0c4d6d50a8bf5a8dafb2ec962aff9c033b8472bb54"
BINDING_SHA = "7b0e3f280b3b549794f6e76eaaddf8f8753a7391544ef09e6d98bd76f261d309"
FREEZE_SHA = "b486264d2caeafc9c899bbd9cb0bf6e758ff952582466cffaacb7c8f511b0bfe"
PHASE6J_SHA = "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438"

PRODUCER_SHA = {
    "canary": "aebc8e6d9d89ee44eb31e8b4bb39afa1efe46f597b0dacdfd1d9d0efc9a1a645",
    "fleet": "82424cf132a7272c9dc8a7ffd2271472b19990a209490f9dbc94bec7a7af9188",
}
PRODUCER_SOURCE = {
    "canary": "experiments/phase_05/compute_fi_outer_canary_executor_v1.py",
    "fleet": "experiments/phase_05/compute_fi_outer_fleet_executor_v1.py",
}

CANARY_MEMBER = (5, 9, "fp32", 42)
CANARY_ID = "p5e-c0-f5-s009-fp32-seed42-e19b32428112a04b"
SEEDS = (42, 123, 2025)
VARIANTS = ("fp32", "ptq_v7")

UNKNOWN_CACHE = "PHASE6L_UNKNOWN_OR_MISMATCHED_FROZEN_CLEAN_CACHE"
INVALID_INVENTORY = "PHASE6L_FROZEN_CACHE_INVENTORY_INVALID"
INVALID_RESOLUTION = "PHASE6L_FROZEN_CACHE_RESOLUTION_INVALID"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(root: Path, relative: str, expected: str) -> None:
    if sha256_file(root / relative) != expected:
        raise RuntimeError(INVALID_INVENTORY + ": " + relative)


def _member(row: Mapping[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["fold"]),
        int(row["subject"]),
        str(row["model_variant"]),
        int(row["checkpoint_seed"]),
    )


def load_frozen_inventory(root: Path = ROOT) -> dict[str, Any]:
    """Bind all 366 identities to verified Phase5 and Phase6K evidence."""
    root = Path(root)

    for relative, expected in (
        (PLAN_REL, PLAN_SHA),
        (BINDING_REL, BINDING_SHA),
        (FREEZE_REL, FREEZE_SHA),
        (PHASE6J_REL, PHASE6J_SHA),
    ):
        _require_sha(root, relative, expected)

    for kind, relative in PRODUCER_SOURCE.items():
        _require_sha(root, relative, PRODUCER_SHA[kind])

    plan = json.loads((root / PLAN_REL).read_text(encoding="utf-8"))
    binding = json.loads((root / BINDING_REL).read_text(encoding="utf-8"))
    freeze = json.loads((root / FREEZE_REL).read_text(encoding="utf-8"))

    if not (
        freeze["freeze_scope"] == "EVIDENCE_ONLY"
        and freeze["execution_authorized"] is False
        and freeze["historical_phase6e_digests_reproduced"] is False
        and freeze["phase6j_producer_aware_runtime_qualified"] is False
        and binding["execution_authorized"] is False
        and binding["verified_cache_count"] == 366
        and binding["all_output_hashes_revalidated"] is True
        and binding["wrong_producer_negative_tests_passed"] is True
        and binding["phase5_plan_sha256"] == PLAN_SHA
    ):
        raise RuntimeError(INVALID_INVENTORY + ": frozen qualification")

    plan_rows = plan["clean_caches"]
    binding_rows = binding["rows"]

    if len(plan_rows) != 366 or len(binding_rows) != 366:
        raise RuntimeError(INVALID_INVENTORY + ": cardinality")

    plan_by_key = {}
    binding_by_key = {}
    plan_ids = set()
    binding_ids = set()

    for row in plan_rows:
        key = _member(row)
        cache_id = str(row["clean_cache_id"])
        if key in plan_by_key or cache_id in plan_ids:
            raise RuntimeError(INVALID_INVENTORY + ": duplicate plan entry")
        plan_by_key[key] = row
        plan_ids.add(cache_id)

    for row in binding_rows:
        key = _member(row)
        cache_id = str(row["clean_cache_id"])
        if key in binding_by_key or cache_id in binding_ids:
            raise RuntimeError(INVALID_INVENTORY + ": duplicate binding entry")
        binding_by_key[key] = row
        binding_ids.add(cache_id)

    if set(plan_by_key) != set(binding_by_key) or plan_ids != binding_ids:
        raise RuntimeError(INVALID_INVENTORY + ": plan/binding mismatch")

    census = Counter()
    by_subject = defaultdict(set)
    total_windows = 0

    for key, plan_row in plan_by_key.items():
        fold, subject, variant, seed = key
        bound = binding_by_key[key]
        cache_id = str(plan_row["clean_cache_id"])

        kind = "canary" if key == CANARY_MEMBER else "fleet"
        expected_sha = PRODUCER_SHA[kind]

        if not (
            variant in VARIANTS
            and seed in SEEDS
            and str(bound["clean_cache_id"]) == cache_id
            and str(bound["producer_kind"]) == kind
            and str(bound["expected_executor_sha256"]) == expected_sha
            and int(bound["window_count"]) == int(plan_row["window_count"])
            and bound["success_marker_validated"] is True
            and bound["output_hashes_validated"] is True
            and bound["metadata_identity_validated"] is True
        ):
            raise RuntimeError(INVALID_INVENTORY + ": bound cache mismatch")

        if (cache_id == CANARY_ID) != (key == CANARY_MEMBER):
            raise RuntimeError(INVALID_INVENTORY + ": canary identity")

        if key == CANARY_MEMBER and cache_id != CANARY_ID:
            raise RuntimeError(INVALID_INVENTORY + ": canary ID")

        by_subject[subject].add((fold, variant, seed))
        total_windows += int(plan_row["window_count"])
        census[kind] += 1

    expected_members = {
        (variant, seed) for variant in VARIANTS for seed in SEEDS
    }

    if not (
        census == {"canary": 1, "fleet": 365}
        and len(by_subject) == 61
        and all(
            {(variant, seed) for _, variant, seed in members}
            == expected_members
            and len({fold for fold, _, _ in members}) == 1
            for members in by_subject.values()
        )
        and total_windows == 1642980
    ):
        raise RuntimeError(INVALID_INVENTORY + ": ownership census")

    return {
        "phase5_plan": plan,
        "plan_by_member": plan_by_key,
        "binding_by_member": binding_by_key,
        "cache_count": 366,
        "execution_authorized": False,
    }


def select_frozen_cache(
    inventory: Mapping[str, Any],
    *,
    fold: int,
    subject: int,
    model_variant: str,
    checkpoint_seed: int,
    requested_cache_id: str | None = None,
) -> dict[str, Any]:
    """Select a producer using frozen identity, never marker self-reporting."""
    key = (int(fold), int(subject), str(model_variant), int(checkpoint_seed))

    plan_row = inventory["plan_by_member"].get(key)
    bound = inventory["binding_by_member"].get(key)

    if plan_row is None or bound is None:
        raise RuntimeError(UNKNOWN_CACHE)

    cache_id = str(plan_row["clean_cache_id"])
    kind = "canary" if key == CANARY_MEMBER else "fleet"
    producer = PRODUCER_SHA[kind]

    if (
        (requested_cache_id is not None and requested_cache_id != cache_id)
        or bound["clean_cache_id"] != cache_id
        or bound["producer_kind"] != kind
        or bound["expected_executor_sha256"] != producer
        or (key == CANARY_MEMBER and cache_id != CANARY_ID)
        or (key != CANARY_MEMBER and cache_id == CANARY_ID)
    ):
        raise RuntimeError(UNKNOWN_CACHE)

    return {
        "clean_cache_id": cache_id,
        "fold": key[0],
        "subject": key[1],
        "model_variant": key[2],
        "checkpoint_seed": key[3],
        "window_count": int(plan_row["window_count"]),
        "producer_kind": kind,
        "expected_executor_sha256": producer,
        "expected_plan_sha256": PLAN_SHA,
        "execution_authorized": False,
    }


def validate_existing_cache(
    inventory: Mapping[str, Any],
    *,
    fold: int,
    subject: int,
    model_variant: str,
    checkpoint_seed: int,
    phase5_output_root: str | Path,
    phase5_outer_core: Any,
    frozen_phase6j_resolver: Callable[..., Mapping[str, Any]],
    requested_cache_id: str | None = None,
) -> dict[str, Any]:
    """Delegate read-only hash validation; never execute or regenerate C0.

    The caller must supply the separately SHA-pinned frozen Phase6J
    resolver and Phase5 core. This function is NOT an execution entrypoint.
    """
    selected = select_frozen_cache(
        inventory,
        fold=fold,
        subject=subject,
        model_variant=model_variant,
        checkpoint_seed=checkpoint_seed,
        requested_cache_id=requested_cache_id,
    )

    if not callable(frozen_phase6j_resolver) or phase5_outer_core is None:
        raise RuntimeError(INVALID_RESOLUTION + ": missing validator")

    result = frozen_phase6j_resolver(
        phase5_plan=inventory["phase5_plan"],
        fold=selected["fold"],
        subject=selected["subject"],
        model_variant=selected["model_variant"],
        checkpoint_seed=selected["checkpoint_seed"],
        phase5_output_root=phase5_output_root,
        expected_plan_sha256=PLAN_SHA,
        expected_executor_sha256=selected["expected_executor_sha256"],
        phase5_outer_core=phase5_outer_core,
    )

    if not isinstance(result, Mapping):
        raise RuntimeError(INVALID_RESOLUTION)

    success = result.get("success")
    if not isinstance(success, Mapping):
        raise RuntimeError(INVALID_RESOLUTION + ": missing success marker")

    coverage = success.get("coverage")
    if not isinstance(coverage, Mapping):
        raise RuntimeError(INVALID_RESOLUTION + ": missing coverage")

    if not (
        result.get("clean_cache_id") == selected["clean_cache_id"]
        and success.get("status") == "PASS"
        and success.get("artifact_id") == selected["clean_cache_id"]
        and success.get("artifact_kind") == "clean_cache"
        and success.get("plan_sha256") == PLAN_SHA
        and success.get("executor_sha256")
        == selected["expected_executor_sha256"]
        and coverage.get("window_count") == selected["window_count"]
        and coverage.get("clean_model_window_evaluations")
        == selected["window_count"]
        and coverage.get("labels_read") is False
        and coverage.get("onfield_read") is False
    ):
        raise RuntimeError(INVALID_RESOLUTION + ": inconsistent result")

    return {
        **selected,
        "existing_cache_hash_validator_passed": True,
        "execution_authorized": False,
    }
