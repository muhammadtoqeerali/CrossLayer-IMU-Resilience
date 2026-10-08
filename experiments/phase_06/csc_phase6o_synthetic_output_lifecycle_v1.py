"""Phase6O synthetic-only frozen Phase6H output lifecycle qualification.

The actual frozen Phase6J prepare/commit/validate helpers are exercised
inside disposable TemporaryDirectory fixtures.

All four outputs contain explicit synthetic qualification markers.
No real CSC prediction, signal, label, checkpoint, gate, shard result
or durable Phase6H success marker is produced.

The fake gate and runtime hashes are intentionally derived from
strings that cannot represent approved production authorization.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import csc_execution_runtime_v1 as runtime

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

OUTPUT_ABORT = "PHASE6O_SYNTHETIC_OUTPUT_ABORT"
EXECUTION_BLOCK = "PHASE6O_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

PINS = {
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_phase6o_synthetic_interface_bridge_v1.py":
        "083b3e6d63b763fcdde0475ad5a77810a48c99f50bd4b84fc2f9788276bf341a",
    "manifests/phase_6o_interface_evidence_v1/phase6o_prospective_csc_execution_contract_inventory_v1.json":
        "c31352b05e88fb7c11417660081b041c3e33a68f84e0390c18f1d5a804ed77bf",
    "manifests/phase_6o_interface_evidence_v1/phase6o_synthetic_interface_bridge_qualification_v1.json":
        "62dd32064441f8e4ff1c314e34619d559898b793c5316bd89aec2d93eee87c61",
    "manifests/phase_6n_canonical_member_evidence_freeze_v1.json":
        "3d7585fdf355b7ced8e692a5076e6b90a26eea25ef43060968eaaea2b598dc8e",
}

SHARD = "phase6o_synthetic_fixture_not_a_real_csc_shard"

GATE_SHA = hashlib.sha256(
    b"PHASE6O_SYNTHETIC_GATE_NOT_AUTHORIZED_V1"
).hexdigest()

RUNTIME_SHA = hashlib.sha256(
    b"PHASE6O_SYNTHETIC_RUNTIME_NOT_PRODUCTION_V1"
).hexdigest()


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
                OUTPUT_ABORT + ": binding mismatch: " + relative
            )

    if Path(runtime.__file__).resolve() != (
        ROOT / "experiments/phase_06/csc_execution_runtime_v1.py"
    ).resolve():
        raise RuntimeError(
            OUTPUT_ABORT + ": runtime module provenance"
        )

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and PRODUCTION_BODY_RELEASED is False
        and runtime.EXECUTION_AUTHORIZED is False
    ):
        raise RuntimeError(
            OUTPUT_ABORT + ": execution state"
        )


def synthetic_output_bytes(name: str) -> bytes:
    """Produce explicitly non-scientific fixture bytes."""

    if name not in runtime.OUTPUT_FILES:
        raise ValueError("not a frozen output filename")

    row = {
        "schema_version": "phase6o_synthetic_output_fixture_v1",
        "output_name": name,
        "qualification_only": True,
        "synthetic_fixture": True,
        "scientific_predictions_present": False,
        "execution_authorized": False,
    }

    return (
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_synthetic_outputs(
    partial: Path,
) -> dict[str, str]:
    """Write only explicitly synthetic fixture content."""

    if not partial.is_dir():
        raise RuntimeError(
            OUTPUT_ABORT + ": partial fixture missing"
        )

    hashes = {}

    for name in runtime.OUTPUT_FILES:
        content = synthetic_output_bytes(name)

        with (partial / name).open("xb") as stream:
            stream.write(content)

        hashes[name] = runtime.sha256_file(partial / name)

        if hashes[name] != hashlib.sha256(content).hexdigest():
            raise RuntimeError(
                OUTPUT_ABORT + ": fixture output digest mismatch"
            )

    return hashes


def _prepare(root: Path, *, recompute: bool = False):
    return runtime.prepare_phase6h_artifact(
        output_root=root,
        shard_id_value=SHARD,
        gate_sha256=GATE_SHA,
        runtime_sha256=RUNTIME_SHA,
        recompute_partial=recompute,
    )


def _commit(
    partial: Path,
    final: Path,
    hashes: Mapping[str, str],
):
    return runtime.commit_phase6h_artifact(
        partial_dir=partial,
        final_dir=final,
        shard_id_value=SHARD,
        gate_sha256=GATE_SHA,
        runtime_sha256=RUNTIME_SHA,
        output_hashes=hashes,
        coverage={
            "synthetic_fixture_only": True,
            "scientific_pair_members_executed": 0,
            "real_model_forwards": 0,
            "execution_authorized": False,
        },
    )


def qualify_synthetic_output_lifecycle() -> dict[str, Any]:
    """Exercise all frozen artifact transitions in disposable storage."""

    check_pins()

    if len(runtime.OUTPUT_FILES) != 4:
        raise RuntimeError(
            OUTPUT_ABORT + ": frozen output file count changed"
        )

    counts = {
        "synthetic_success_commits": 0,
        "valid_success_reuse_checks": 0,
        "marker_before_rename_checks": 0,
        "tamper_rejection_checks": 0,
        "wrong_gate_rejection_checks": 0,
        "recompute_required_checks": 0,
        "explicit_recompute_checks": 0,
        "incorrect_file_hash_rejections": 0,
        "incorrect_output_set_rejections": 0,
        "partial_interruption_rejections": 0,
        "partial_recovery_checks": 0,
    }

    with tempfile.TemporaryDirectory(
        prefix="phase6o-synthetic-output-only-"
    ) as directory:
        fixture_root = Path(directory)
        output_root = fixture_root / "output"

        action, partial, reused = _prepare(output_root)

        if not (
            action == "compute"
            and partial is not None
            and reused is None
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": new fixture preparation"
            )

        if (partial / runtime.SUCCESS_MARKER).exists():
            raise RuntimeError(
                OUTPUT_ABORT + ": premature success marker"
            )

        hashes = write_synthetic_outputs(partial)
        final = output_root / "shards" / SHARD

        real_replace = runtime.os.replace

        def observe_atomic_rename(source, destination):
            marker = Path(source) / runtime.SUCCESS_MARKER

            if not marker.is_file():
                raise RuntimeError(
                    OUTPUT_ABORT + ": marker missing before rename"
                )

            if Path(destination) != final:
                raise RuntimeError(
                    OUTPUT_ABORT + ": unexpected final fixture path"
                )

            counts["marker_before_rename_checks"] += 1

            return real_replace(source, destination)

        with patch.object(runtime.os, "replace", observe_atomic_rename):
            success = _commit(partial, final, hashes)

        counts["synthetic_success_commits"] += 1

        if not (
            success["status"] == "PASS"
            and success["shard_id"] == SHARD
            and success["gate_sha256"] == GATE_SHA
            and success["runtime_sha256"] == RUNTIME_SHA
            and success["coverage"]["synthetic_fixture_only"] is True
            and success["coverage"]["real_model_forwards"] == 0
            and set(success["output_hashes"]) == set(runtime.OUTPUT_FILES)
            and final.is_dir()
            and not partial.exists()
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": atomic success contract"
            )

        validated = runtime.validate_phase6h_success_marker(
            final_dir=final,
            expected_shard_id=SHARD,
            expected_gate_sha256=GATE_SHA,
            expected_runtime_sha256=RUNTIME_SHA,
        )

        if validated != success:
            raise RuntimeError(
                OUTPUT_ABORT + ": marker validation"
            )

        action, other_partial, reused = _prepare(output_root)

        if not (
            action == "reuse"
            and other_partial is None
            and reused == success
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": valid final reuse"
            )

        counts["valid_success_reuse_checks"] += 1

        wrong_gate = runtime.validate_phase6h_success_marker(
            final_dir=final,
            expected_shard_id=SHARD,
            expected_gate_sha256="0" * 64,
            expected_runtime_sha256=RUNTIME_SHA,
        )

        if wrong_gate is not None:
            raise RuntimeError(
                OUTPUT_ABORT + ": fake gate accepted"
            )

        counts["wrong_gate_rejection_checks"] += 1

        # Damage a fixture output. The frozen marker validator must
        # refuse reuse despite the existing success marker.
        tampered_file = final / runtime.PAIR_MEMBERS_JSONL

        with tampered_file.open("ab") as stream:
            stream.write(b"SYNTHETIC_TAMPER\n")

        invalid = runtime.validate_phase6h_success_marker(
            final_dir=final,
            expected_shard_id=SHARD,
            expected_gate_sha256=GATE_SHA,
            expected_runtime_sha256=RUNTIME_SHA,
        )

        if invalid is not None:
            raise RuntimeError(
                OUTPUT_ABORT + ": corrupted output accepted"
            )

        counts["tamper_rejection_checks"] += 1

        try:
            _prepare(output_root, recompute=False)
        except ValueError as exc:
            if "recompute_partial required" not in str(exc):
                raise
            counts["recompute_required_checks"] += 1
        else:
            raise RuntimeError(
                OUTPUT_ABORT + ": invalid final silently reused"
            )

        action, regenerated, reused = _prepare(
            output_root,
            recompute=True,
        )

        if not (
            action == "compute"
            and regenerated is not None
            and reused is None
            and not final.exists()
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": explicit recompute failed"
            )

        counts["explicit_recompute_checks"] += 1

        regenerated_hashes = write_synthetic_outputs(regenerated)

        try:
            _commit(
                regenerated,
                final,
                {
                    **regenerated_hashes,
                    runtime.PAIR_MEMBERS_JSONL: "0" * 64,
                },
            )
        except ValueError as exc:
            if "hash mismatch" not in str(exc):
                raise
            counts["incorrect_file_hash_rejections"] += 1
        else:
            raise RuntimeError(
                OUTPUT_ABORT + ": wrong hash accepted"
            )

        if final.exists() or (
            regenerated / runtime.SUCCESS_MARKER
        ).exists():
            raise RuntimeError(
                OUTPUT_ABORT + ": invalid commit created final"
            )

        try:
            _commit(
                regenerated,
                final,
                {
                    key: value
                    for key, value in regenerated_hashes.items()
                    if key != runtime.PAIR_MEMBERS_JSONL
                },
            )
        except ValueError as exc:
            if "output hash set" not in str(exc):
                raise
            counts["incorrect_output_set_rejections"] += 1
        else:
            raise RuntimeError(
                OUTPUT_ABORT + ": incomplete output set accepted"
            )

        _commit(regenerated, final, regenerated_hashes)
        counts["synthetic_success_commits"] += 1

        # Separately verify failure/recovery from an interrupted
        # partial directory, not a published final directory.
        interrupted_root = fixture_root / "interrupted"

        action, interrupted, reused = _prepare(interrupted_root)

        if not (
            action == "compute"
            and interrupted is not None
            and reused is None
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": interrupted fixture preparation"
            )

        with (interrupted / "SYNTHETIC_INCOMPLETE").open(
            "xb"
        ) as stream:
            stream.write(b"PHASE6O_SYNTHETIC_INTERRUPTION\n")

        try:
            _prepare(interrupted_root, recompute=False)
        except ValueError as exc:
            if "recompute_partial required" not in str(exc):
                raise
            counts["partial_interruption_rejections"] += 1
        else:
            raise RuntimeError(
                OUTPUT_ABORT + ": abandoned partial accepted"
            )

        action, repaired, reused = _prepare(
            interrupted_root,
            recompute=True,
        )

        if not (
            action == "compute"
            and repaired == interrupted
            and reused is None
            and not (repaired / "SYNTHETIC_INCOMPLETE").exists()
            and not (repaired / runtime.SUCCESS_MARKER).exists()
        ):
            raise RuntimeError(
                OUTPUT_ABORT + ": interrupted partial recovery"
            )

        counts["partial_recovery_checks"] += 1

        fixture_hashes = {
            name: hashlib.sha256(
                synthetic_output_bytes(name)
            ).hexdigest()
            for name in runtime.OUTPUT_FILES
        }

        if fixture_hashes != hashes:
            raise RuntimeError(
                OUTPUT_ABORT + ": nondeterministic fixture bytes"
            )

        temp_path = fixture_root

    if temp_path.exists():
        raise RuntimeError(
            OUTPUT_ABORT + ": temporary fixture cleanup failed"
        )

    if not (
        counts["synthetic_success_commits"] == 2
        and all(
            number >= 1
            for name, number in counts.items()
            if name != "synthetic_success_commits"
        )
    ):
        raise RuntimeError(
            OUTPUT_ABORT + ": incomplete transition census"
        )

    return {
        "qualification_status":
            "FROZEN_PHASE6H_SYNTHETIC_OUTPUT_LIFECYCLE_PASS",
        "qualification_scope":
            "DISPOSABLE_SYNTHETIC_FIXTURES_ONLY",
        "frozen_output_file_count": len(runtime.OUTPUT_FILES),
        "frozen_output_file_names": list(runtime.OUTPUT_FILES),
        "synthetic_output_sha256": fixture_hashes,
        "transition_counters": counts,
        "temporary_fixture_cleanup_verified": True,
        "synthetic_success_markers_created_then_deleted": 2,
        "permanent_phase6h_success_marker_created": False,
        "scientific_output_schema_validated": False,
        "real_csc_artifacts_created": False,
        "real_csc_execution_performed": False,
        "model_forward_performed": False,
        "ptq_restoration_qualified": False,
        "historical_phase6e_digests_reproduced": False,
        "execution_authorized": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
