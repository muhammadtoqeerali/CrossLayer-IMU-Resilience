"""Phase6O evidence-only frozen CSC shard cardinality guard.

Independently reconciles all 732 frozen Phase6K subject-family shard
workloads with published Phase6N member-stream and shard inventories.

Provides a fail-closed checker for declared shard coverage against
those frozen counts.

Separately tests that the frozen Phase6H success-marker validator is
insufficient by itself to verify actual JSONL row counts. A synthetic
fixture validator adds a physical line census and checks the marker's
declared counts.

No real CSC output is read or validated by this qualification.
No real CSC success marker is created or reused.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import csc_execution_runtime_v1 as runtime

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

ABORT = "PHASE6O_FROZEN_CARDINALITY_ABORT"
EXECUTION_BLOCK = "PHASE6O_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

NUMERIC = (
    "manifests/phase_6k_csc_numeric_evidence_v1/"
    "phase6k_numeric_workload_report_v1.json"
)

FLEET = (
    "manifests/phase_6n_canonical_evidence_v1/"
    "phase6n_full_fleet_canonical_member_index_audit_v1.json"
)

ARCH = (
    "configs/evaluation/"
    "phase6h_csc_execution_architecture_clarification_v1.json"
)

PINS = {
    NUMERIC:
        "e9b977b40dfdeb2a25a9ea041e5c3f22041a58251cfba79b8d9b055df2c725c2",
    FLEET:
        "abc7f60308a09022b2afcd915b4247f78a3a485e88dc30cc724155bf03608c45",
    ARCH:
        "7cb6234876506fe8f127ee823550d235a04063643013eb3348d81583fdb88a26",
    "manifests/phase_6n_canonical_member_evidence_freeze_v1.json":
        "3d7585fdf355b7ced8e692a5076e6b90a26eea25ef43060968eaaea2b598dc8e",
    "manifests/phase_6m_csc_pre_forward_evidence_freeze_v1.json":
        "a77b437cf669894b4423caa65f9f094f27074b58e862261376c169d04b1a9607",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_phase6o_synthetic_record_contract_v1.py":
        "3c0b9a0a6b81ca3a78922f6226f9991b63c4a2d460e1d9b26ca0580f78a4f168",
    "manifests/phase_6o_interface_evidence_v1/phase6o_synthetic_record_contract_qualification_v1.json":
        "cc0b096a48e46518c4aa74de7e19159af572f3ca5f55d8365f939acb8df847b3",
}

COUNT_FIELDS = (
    "pair_member_records",
    "sensor_reference_records",
    "csc_fault_records",
    "simultaneous_overlap_records",
)

OUTPUT_TO_COUNT = {
    runtime.PAIR_MEMBERS_JSONL: "pair_member_records",
    runtime.SENSOR_REFERENCE_JSONL: "sensor_reference_records",
    runtime.CSC_FAULT_JSONL: "csc_fault_records",
}

FIXTURE_SHARD = "phase6o_synthetic_cardinality_fixture_not_real_shard"

FIXTURE_GATE_SHA = hashlib.sha256(
    b"PHASE6O_SYNTHETIC_CARDINALITY_GATE_NOT_AUTHORIZED"
).hexdigest()

FIXTURE_RUNTIME_SHA = hashlib.sha256(
    b"PHASE6O_SYNTHETIC_CARDINALITY_RUNTIME_NOT_PRODUCTION"
).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _abort(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(ABORT + ": " + reason)


def _load(relative: str) -> dict[str, Any]:
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8")
    )


def validate_count_claim(
    coverage: Mapping[str, Any],
    expected: Mapping[str, int],
) -> None:
    """Reject missing, noninteger or incorrect declared counts."""

    _abort(
        isinstance(coverage, Mapping)
        and isinstance(expected, Mapping),
        "coverage mapping required",
    )

    for field in COUNT_FIELDS:
        _abort(field in expected, "expected field missing: " + field)

        _abort(
            field in coverage
            and type(coverage[field]) is int
            and coverage[field] >= 0
            and coverage[field] == expected[field],
            "invalid coverage: " + field,
        )


class FrozenShardCardinalityPlan:
    def __init__(self) -> None:
        for relative, expected in PINS.items():
            _abort(
                sha(ROOT / relative) == expected,
                "SHA256 binding " + relative,
            )

        _abort(
            Path(runtime.__file__).resolve()
            == (
                ROOT
                / "experiments/phase_06/csc_execution_runtime_v1.py"
            ).resolve(),
            "frozen runtime origin",
        )

        _abort(
            EXECUTION_AUTHORIZED is False
            and MODEL_FORWARD_AUTHORIZED is False
            and PRODUCTION_BODY_RELEASED is False
            and runtime.EXECUTION_AUTHORIZED is False,
            "execution authorization",
        )

        numeric = _load(NUMERIC)
        fleet = _load(FLEET)
        arch = _load(ARCH)

        _abort(
            numeric["execution_authorized"] is False
            and fleet["execution_authorized"] is False
            and arch["execution_authorized"] is False
            and fleet["qualification_status"]
                == "FULL_61_SUBJECT_CANONICAL_MEMBER_INDEX_PASS"
            and fleet["subject_count"] == 61
            and fleet["shard_count"] == 732
            and fleet["model_member_stream_count"] == 4392
            and fleet["model_independent_pair_count"] == 4237835
            and fleet["canonical_pair_member_count"] == 21793038,
            "historical evidence status",
        )

        frozen = {
            row["runtime_shard_id"]: row
            for row in numeric["inventory"]["rows"]
        }

        qualified = {
            row["runtime_shard_id"]: row
            for row in fleet["shards"]
        }

        _abort(
            len(frozen) == len(qualified) == 732
            and set(frozen) == set(qualified),
            "independent shard identities",
        )

        streams_by_shard = defaultdict(list)

        for row in fleet["streams"]:
            streams_by_shard[
                row["runtime_shard_id"]
            ].append(row)

        _abort(
            len(fleet["streams"]) == 4392
            and set(streams_by_shard) == set(frozen),
            "complete six-member stream surface",
        )

        self.expected_by_shard = {}
        globals_total = Counter()
        all_subjects = set()

        for shard_id, frozen_row in frozen.items():
            qualified_row = qualified[shard_id]
            streams = streams_by_shard[shard_id]

            for original, derived in (
                ("fold", "fold"),
                ("subject", "subject"),
                ("sensor_family", "sensor_family"),
                ("sensor_parent_kind", "sensor_parent_kind"),
                ("pair_count", "model_independent_pair_count"),
                ("pair_member_count", "pair_member_count"),
                (
                    "new_phase6k_descriptor_sha256",
                    "frozen_phase6k_descriptor_sha256",
                ),
                ("zero_overlap_pair_count", "zero_overlap_pair_count"),
                ("structural_omission_count", "structural_omission_count"),
            ):
                _abort(
                    frozen_row[original] == qualified_row[derived],
                    "shard identity/workload: " + shard_id + ":" + original,
                )

            _abort(
                len(streams) == 6
                and qualified_row["model_member_stream_count"] == 6,
                "six streams: " + shard_id,
            )

            stream_identities = {
                (
                    row["model_variant"],
                    int(row["checkpoint_seed"]),
                )
                for row in streams
            }

            _abort(
                stream_identities == {
                    (variant, seed)
                    for variant in ("fp32", "ptq_v7")
                    for seed in (42, 123, 2025)
                },
                "variant/seed identities: " + shard_id,
            )

            for frozen_field, stream_field in (
                ("pair_member_count", "pair_member_count"),
                (
                    "sensor_reference_member_windows",
                    "sensor_reference_member_windows",
                ),
                (
                    "compute_faulted_member_windows",
                    "compute_faulted_member_windows",
                ),
                (
                    "simultaneous_overlap_member_windows",
                    "simultaneous_overlap_member_windows",
                ),
                ("union_member_windows", "union_member_windows"),
            ):
                _abort(
                    int(frozen_row[frozen_field])
                    == sum(int(row[stream_field]) for row in streams),
                    "independent member sums: "
                    + shard_id + ":" + frozen_field,
                )

            for stream in streams:
                _abort(
                    stream["execution_authorized"] is False
                    and stream["runtime_shard_id"] == shard_id
                    and stream["fold"] == frozen_row["fold"]
                    and stream["subject"] == frozen_row["subject"]
                    and stream["sensor_family"]
                        == frozen_row["sensor_family"],
                    "stream provenance: " + shard_id,
                )

            counts = {
                "pair_member_records":
                    int(frozen_row["pair_member_count"]),
                "sensor_reference_records":
                    int(frozen_row["sensor_reference_member_windows"]),
                "csc_fault_records":
                    int(frozen_row["compute_faulted_member_windows"]),
                "simultaneous_overlap_records":
                    int(frozen_row["simultaneous_overlap_member_windows"]),
            }

            _abort(
                all(value > 0 for value in counts.values())
                or (
                    counts["pair_member_records"] > 0
                    and counts["sensor_reference_records"] > 0
                    and counts["csc_fault_records"] > 0
                    and counts["simultaneous_overlap_records"] == 0
                ),
                "nonpositive frozen workload: " + shard_id,
            )

            self.expected_by_shard[shard_id] = counts

            globals_total.update(counts)
            all_subjects.add(int(frozen_row["subject"]))

        expected_global = {
            "pair_member_records": 21793038,
            "sensor_reference_records": 35167107,
            "csc_fault_records": 411540372,
            "simultaneous_overlap_records": 19926021,
        }

        _abort(
            globals_total == expected_global
            and len(all_subjects) == 61
            and len(self.expected_by_shard) == 732,
            "global frozen workload counts",
        )

        architecture_counts = arch["cardinality_contract"]

        _abort(
            architecture_counts["global_pair_member_records"]
                == expected_global["pair_member_records"]
            and architecture_counts["global_sensor_reference_records"]
                == expected_global["sensor_reference_records"]
            and architecture_counts["global_csc_fault_window_records"]
                == expected_global["csc_fault_records"]
            and arch["pre_execution_gate_required"]["required"] is True,
            "Phase6H architecture cardinality",
        )

        self.global_counts = dict(globals_total)

    def expected_counts(self, shard_id: str) -> dict[str, int]:
        counts = self.expected_by_shard.get(shard_id)

        if counts is None:
            raise RuntimeError(ABORT + ": unknown frozen shard")

        return dict(counts)

    def check_frozen_claim(
        self,
        shard_id: str,
        coverage: Mapping[str, Any],
    ) -> None:
        _abort(
            coverage.get("synthetic_fixture_only") is not True,
            "synthetic fixture presented as frozen shard",
        )

        validate_count_claim(
            coverage,
            self.expected_counts(shard_id),
        )


def count_physical_jsonl_lines(path: Path) -> int:
    """Count physical canonical-looking lines without parsing predictions."""

    _abort(path.is_file(), "missing output file")

    count = 0

    with path.open("rb") as handle:
        for line in handle:
            _abort(
                line.startswith(b"{")
                and line.endswith(b"}\n")
                and b"\r" not in line,
                "malformed or non-LF JSONL framing",
            )

            count += 1

    return count


def validate_synthetic_fixture(
    final_dir: Path,
    expected: Mapping[str, int],
) -> dict[str, int]:
    """Verify hash-valid marker, declared coverage and real line counts.

    This deliberately accepts only a synthetic sentinel shard identity,
    never a frozen real CSC shard.
    """

    _abort(
        final_dir.name == FIXTURE_SHARD,
        "not an explicitly synthetic fixture shard",
    )

    marker = runtime.validate_phase6h_success_marker(
        final_dir=final_dir,
        expected_shard_id=FIXTURE_SHARD,
        expected_gate_sha256=FIXTURE_GATE_SHA,
        expected_runtime_sha256=FIXTURE_RUNTIME_SHA,
    )

    _abort(marker is not None, "invalid frozen success marker")
    _abort(
        marker["coverage"].get("synthetic_fixture_only") is True
        and marker["coverage"].get("execution_authorized") is False,
        "not a synthetic coverage marker",
    )

    validate_count_claim(marker["coverage"], expected)

    physical = {}

    for filename, count_field in OUTPUT_TO_COUNT.items():
        physical[count_field] = count_physical_jsonl_lines(
            final_dir / filename
        )

        _abort(
            physical[count_field] == expected[count_field],
            "physical JSONL count: " + filename,
        )

    return physical


def _write_fixture_outputs(partial: Path) -> dict[str, str]:
    for filename, count in (
        (runtime.PAIR_MEMBERS_JSONL, 2),
        (runtime.SENSOR_REFERENCE_JSONL, 3),
        (runtime.CSC_FAULT_JSONL, 4),
    ):
        runtime.write_jsonl(
            partial / filename,
            [
                {
                    "synthetic_fixture": True,
                    "record_number": index,
                    "scientific_prediction": False,
                }
                for index in range(count)
            ],
        )

    runtime.write_json(
        partial / runtime.METADATA_JSON,
        {
            "synthetic_fixture": True,
            "execution_authorized": False,
            "real_csc_results": False,
        },
    )

    return {
        filename: runtime.sha256_file(partial / filename)
        for filename in runtime.OUTPUT_FILES
    }


def qualify_synthetic_fixture_attack_cases() -> dict[str, Any]:
    """Demonstrate why marker hashes alone are insufficient."""

    expected = {
        "pair_member_records": 2,
        "sensor_reference_records": 3,
        "csc_fault_records": 4,
        "simultaneous_overlap_records": 1,
    }

    checks = Counter()

    with tempfile.TemporaryDirectory(
        prefix="phase6o-cardinality-guard-fixture-"
    ) as directory:
        output_root = Path(directory) / "out"

        action, partial, reused = runtime.prepare_phase6h_artifact(
            output_root=output_root,
            shard_id_value=FIXTURE_SHARD,
            gate_sha256=FIXTURE_GATE_SHA,
            runtime_sha256=FIXTURE_RUNTIME_SHA,
            recompute_partial=False,
        )

        _abort(
            action == "compute"
            and partial is not None
            and reused is None,
            "fixture preparation",
        )

        output_hashes = _write_fixture_outputs(partial)

        final = output_root / "shards" / FIXTURE_SHARD

        coverage = {
            **expected,
            "synthetic_fixture_only": True,
            "execution_authorized": False,
            "scientific_pair_members_executed": 0,
        }

        runtime.commit_phase6h_artifact(
            partial_dir=partial,
            final_dir=final,
            shard_id_value=FIXTURE_SHARD,
            gate_sha256=FIXTURE_GATE_SHA,
            runtime_sha256=FIXTURE_RUNTIME_SHA,
            output_hashes=output_hashes,
            coverage=coverage,
        )

        verified = validate_synthetic_fixture(final, expected)

        _abort(
            verified == {
                "pair_member_records": 2,
                "sensor_reference_records": 3,
                "csc_fault_records": 4,
            },
            "valid fixture coverage",
        )
        checks["valid_fixture_pass"] += 1

        marker_path = final / runtime.SUCCESS_MARKER
        original_marker = marker_path.read_bytes()
        success = json.loads(original_marker)

        # Attack 1: coverage can be forged without changing any of
        # the output hashes. The frozen marker validator still accepts
        # the file identity/hash set, but our guard must reject it.
        forged = dict(success)
        forged["coverage"] = {
            **coverage,
            "pair_member_records": 1,
        }

        runtime.write_json(marker_path, forged)

        frozen_accepts = runtime.validate_phase6h_success_marker(
            final_dir=final,
            expected_shard_id=FIXTURE_SHARD,
            expected_gate_sha256=FIXTURE_GATE_SHA,
            expected_runtime_sha256=FIXTURE_RUNTIME_SHA,
        )

        _abort(
            frozen_accepts is not None,
            "unexpected frozen marker behavior",
        )
        checks["frozen_hash_only_marker_accepts_wrong_coverage"] += 1

        try:
            validate_synthetic_fixture(final, expected)
        except RuntimeError as exc:
            _abort(ABORT in str(exc), "coverage rejection error")
            checks["forged_coverage_rejected"] += 1
        else:
            raise RuntimeError(
                ABORT + ": forged coverage incorrectly accepted"
            )

        marker_path.write_bytes(original_marker)

        # Attack 2: a reduced JSONL file can be accompanied by its
        # new correct SHA in the marker. SHA validation still passes,
        # but the physical cardinality no longer matches.
        fault_path = final / runtime.CSC_FAULT_JSONL
        original_fault = fault_path.read_bytes()

        rows = original_fault.splitlines(keepends=True)
        _abort(len(rows) == 4, "fixture fault row count")

        fault_path.write_bytes(b"".join(rows[:3]))

        rehashed = json.loads(original_marker)
        rehashed["output_hashes"][runtime.CSC_FAULT_JSONL] = (
            runtime.sha256_file(fault_path)
        )
        runtime.write_json(marker_path, rehashed)

        _abort(
            runtime.validate_phase6h_success_marker(
                final_dir=final,
                expected_shard_id=FIXTURE_SHARD,
                expected_gate_sha256=FIXTURE_GATE_SHA,
                expected_runtime_sha256=FIXTURE_RUNTIME_SHA,
            ) is not None,
            "rehashed fixture should pass frozen hash check",
        )

        checks["frozen_marker_accepts_rehashed_undercount"] += 1

        try:
            validate_synthetic_fixture(final, expected)
        except RuntimeError as exc:
            _abort(ABORT in str(exc), "physical count rejection error")
            checks["physical_undercount_rejected"] += 1
        else:
            raise RuntimeError(
                ABORT + ": rehashed undercount incorrectly accepted"
            )

        fault_path.write_bytes(original_fault)
        marker_path.write_bytes(original_marker)

        validate_synthetic_fixture(final, expected)
        checks["restored_fixture_revalidated"] += 1

        # Attack 3: a structurally malformed JSONL file with the
        # correct line count and refreshed hash is still rejected.
        malformed = original_fault.replace(b"\n", b"\r\n")
        fault_path.write_bytes(malformed)

        changed = json.loads(original_marker)
        changed["output_hashes"][runtime.CSC_FAULT_JSONL] = (
            runtime.sha256_file(fault_path)
        )
        runtime.write_json(marker_path, changed)

        try:
            validate_synthetic_fixture(final, expected)
        except RuntimeError as exc:
            _abort(ABORT in str(exc), "framing rejection error")
            checks["noncanonical_line_endings_rejected"] += 1
        else:
            raise RuntimeError(
                ABORT + ": noncanonical framing incorrectly accepted"
            )

        temporary_root = Path(directory)

    _abort(
        not temporary_root.exists(),
        "temporary synthetic success marker not cleaned up",
    )

    _abort(
        len(checks) == 7
        and all(value == 1 for value in checks.values()),
        "synthetic negative test census",
    )

    return {
        "synthetic_transition_checks": dict(checks),
        "temporary_fixture_cleanup_verified": True,
        "permanent_success_marker_created": False,
        "real_frozen_shard_output_read": False,
        "execution_authorized": False,
    }


def qualify_cardinality_guard() -> dict[str, Any]:
    plan = FrozenShardCardinalityPlan()

    _abort(
        len(plan.expected_by_shard) == 732,
        "frozen shard census",
    )

    for shard_id, expected in plan.expected_by_shard.items():
        plan.check_frozen_claim(shard_id, expected)

        bad = {
            **expected,
            "pair_member_records":
                expected["pair_member_records"] - 1,
        }

        try:
            plan.check_frozen_claim(shard_id, bad)
        except RuntimeError as exc:
            _abort(
                ABORT in str(exc),
                "negative frozen claim error",
            )
        else:
            raise RuntimeError(
                ABORT + ": undercount accepted: " + shard_id
            )

    negative_fixture = qualify_synthetic_fixture_attack_cases()

    return {
        "qualification_status":
            "PHASE6O_FROZEN_732_SHARD_CARDINALITY_GUARD_PASS",
        "qualification_scope":
            "FROZEN_METADATA_EXPECTATIONS_AND_SYNTHETIC_OUTPUT_ATTACK_FIXTURES",
        "frozen_shard_count": 732,
        "correct_frozen_claims_validated": 732,
        "incorrect_frozen_claims_rejected": 732,
        "global_expected_counts": dict(plan.global_counts),
        "synthetic_fixture_result": negative_fixture,
        "per_shard_expected_counts": [
            {
                "runtime_shard_id": shard_id,
                **counts,
            }
            for shard_id, counts in sorted(
                plan.expected_by_shard.items()
            )
        ],
        "actual_frozen_shard_outputs_validated": False,
        "full_scientific_row_payload_validation_qualified": False,
        "per_shard_physical_overlap_membership_validated": False,
        "prospective_production_guard_wired": False,
        "historical_phase6e_digests_reproduced": False,
        "model_forward_performed": False,
        "real_csc_execution_performed": False,
        "execution_authorized": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
