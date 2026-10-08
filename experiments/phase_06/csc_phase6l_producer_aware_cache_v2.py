"""Phase6L v2: provenance-pinned, metadata-only Phase5 cache validation.

This is a prospective qualification interface, not a CSC execution
adapter or authorization gate.

Unlike the Phase6L v1 prototype, this interface does not accept an
injected resolver or Phase5 validator module from its caller.

Frozen sources and in-memory function origins are checked before each
cache validation. Clean outputs are delegated to the existing frozen
Phase6J and Phase5 hash validators; prediction rows are never parsed.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import csc_phase6l_producer_aware_cache_v1 as legacy

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False

PROVENANCE_ABORT = "PHASE6L_V2_VALIDATOR_PROVENANCE_ABORT"

SOURCE_PINS = {
    "csc_phase6l_producer_aware_cache_v1": (
        "experiments/phase_06/csc_phase6l_producer_aware_cache_v1.py",
        "bfb034367519521b93a8ae043dd181e5d185526815728b9b21150960f1c06a14",
        (
            "load_frozen_inventory",
            "select_frozen_cache",
            "validate_existing_cache",
        ),
    ),
    "csc_execution_runtime_v1": (
        "experiments/phase_06/csc_execution_runtime_v1.py",
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
        ("resolve_phase5_clean_cache",),
    ),
    "compute_fi_outer_executor_v1": (
        "experiments/phase_05/compute_fi_outer_executor_v1.py",
        "77ee323f71a5dbf4c54cc454a5911849aa45c5d4bdcc6d63bb908d8cece363d0",
        ("validate_success_marker",),
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verified_module(name: str) -> ModuleType:
    relative, expected_sha, functions = SOURCE_PINS[name]
    canonical_path = (ROOT / relative).resolve(strict=True)

    module = importlib.import_module(name)

    if not isinstance(module, ModuleType):
        raise RuntimeError(PROVENANCE_ABORT)

    if module.__name__ != name:
        raise RuntimeError(PROVENANCE_ABORT + ": module name")

    try:
        module_path = Path(module.__file__).resolve(strict=True)
    except (AttributeError, TypeError, OSError) as exc:
        raise RuntimeError(
            PROVENANCE_ABORT + ": missing module source"
        ) from exc

    if module_path != canonical_path:
        raise RuntimeError(PROVENANCE_ABORT + ": module path")

    if _sha256_file(canonical_path) != expected_sha:
        raise RuntimeError(PROVENANCE_ABORT + ": source SHA256")

    for function_name in functions:
        function = getattr(module, function_name, None)

        if not isinstance(function, FunctionType):
            raise RuntimeError(
                PROVENANCE_ABORT + ": function type"
            )

        if function.__module__ != name:
            raise RuntimeError(
                PROVENANCE_ABORT + ": function module"
            )

        if function.__globals__ is not vars(module):
            raise RuntimeError(
                PROVENANCE_ABORT + ": function globals"
            )

        try:
            code_path = Path(
                function.__code__.co_filename
            ).resolve(strict=True)
        except (TypeError, OSError) as exc:
            raise RuntimeError(
                PROVENANCE_ABORT + ": function code source"
            ) from exc

        if code_path != canonical_path:
            raise RuntimeError(
                PROVENANCE_ABORT + ": function code origin"
            )

    return module


def verified_production_validators() -> tuple[Any, ModuleType]:
    """Return only canonical, SHA-pinned frozen validator objects."""
    checked_legacy = _verified_module(
        "csc_phase6l_producer_aware_cache_v1"
    )

    if checked_legacy is not legacy:
        raise RuntimeError(
            PROVENANCE_ABORT + ": inconsistent legacy module"
        )

    runtime = _verified_module("csc_execution_runtime_v1")
    core = _verified_module("compute_fi_outer_executor_v1")

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and legacy.EXECUTION_AUTHORIZED is False
        and legacy.MODEL_FORWARD_AUTHORIZED is False
        and runtime.EXECUTION_AUTHORIZED is False
    ):
        raise RuntimeError(
            PROVENANCE_ABORT + ": execution enablement"
        )

    return runtime.resolve_phase5_clean_cache, core


class FrozenProducerAwareCacheValidator:
    """Metadata/cache-hash validator with no injectable resolver API."""

    def __init__(self, *, phase5_output_root: str | Path) -> None:
        verified_production_validators()

        self._phase5_output_root = Path(phase5_output_root)
        self._inventory = legacy.load_frozen_inventory()

        if not (
            self._inventory["execution_authorized"] is False
            and self._inventory["cache_count"] == 366
        ):
            raise RuntimeError(
                PROVENANCE_ABORT + ": frozen inventory"
            )

    def select(
        self,
        *,
        fold: int,
        subject: int,
        model_variant: str,
        checkpoint_seed: int,
        requested_cache_id: str | None = None,
    ) -> dict[str, Any]:
        verified_production_validators()

        return legacy.select_frozen_cache(
            self._inventory,
            fold=fold,
            subject=subject,
            model_variant=model_variant,
            checkpoint_seed=checkpoint_seed,
            requested_cache_id=requested_cache_id,
        )

    def validate(
        self,
        *,
        fold: int,
        subject: int,
        model_variant: str,
        checkpoint_seed: int,
        requested_cache_id: str | None = None,
    ) -> dict[str, Any]:
        # No caller-supplied resolver, core, cache plan or producer SHA.
        resolver, core = verified_production_validators()

        result = legacy.validate_existing_cache(
            self._inventory,
            fold=fold,
            subject=subject,
            model_variant=model_variant,
            checkpoint_seed=checkpoint_seed,
            requested_cache_id=requested_cache_id,
            phase5_output_root=self._phase5_output_root,
            phase5_outer_core=core,
            frozen_phase6j_resolver=resolver,
        )

        if not (
            result["existing_cache_hash_validator_passed"] is True
            and result["execution_authorized"] is False
        ):
            raise RuntimeError(
                PROVENANCE_ABORT + ": validation outcome"
            )

        return result
