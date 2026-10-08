"""Phase6O frozen Phase6H record-structure qualification, synthetic only.

Reconstructs 189 previously qualified frozen Phase6G request identities.
Uses Phase6J's actual synthetic record constructors and validates
required schema fields, inter-record identities, window membership,
temporal overlap, reference routing and per-request cardinality.

The three JSONL record types are written to a disposable directory
and read back in frozen canonical serialization.

This is NOT scientific output-value validation. Hash-like fields in
the injected summaries are synthetic tokens, not real model hashes.

No real Phase6H shard success marker is created.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import csc_execution_adapter_v1 as adapter
import csc_execution_runtime_v1 as runtime
import csc_phase6k_metadata_fleet_auditor_v1 as auditor
import csc_phase6m_canonical_request_witness_v1 as witness
import csc_phase6m_pre_forward_orchestrator_v1 as preflight

ROOT = Path(__file__).resolve().parents[2]

EXECUTION_AUTHORIZED = False
MODEL_FORWARD_AUTHORIZED = False
PRODUCTION_BODY_RELEASED = False

ABORT = "PHASE6O_SYNTHETIC_RECORD_CONTRACT_ABORT"
EXECUTION_BLOCK = "PHASE6O_REAL_CSC_EXECUTION_NOT_AUTHORIZED"

ARCH = (
    "configs/evaluation/"
    "phase6h_csc_execution_architecture_clarification_v1.json"
)

PINS = {
    ARCH:
        "7cb6234876506fe8f127ee823550d235a04063643013eb3348d81583fdb88a26",
    "experiments/phase_06/csc_execution_runtime_v1.py":
        "e68f1e8ee7e1dbe5dab6801669d8e7ffafc8efddfb20c78871a4422ffc9b5438",
    "experiments/phase_06/csc_execution_adapter_v1.py":
        "2d8d218b0d2214e099dab28b769eeb9fb90d6fa6420b50351b21e54cc4bd1fb5",
    "experiments/phase_06/csc_phase6m_canonical_request_witness_v1.py":
        "6b3d8028b31ad52e10a0fb4e4060e0b7bf8866177cced7630c825adeb91fd774",
    "manifests/phase_6m_pre_forward_evidence_v1/phase6m_canonical_request_synthetic_witness_v1.json":
        "7ad0376d72161702cd36e60117c12fa2a85bf4405e5e032681525f40ce023d46",
    "manifests/phase_6o_interface_evidence_v1/phase6o_synthetic_output_lifecycle_qualification_v1.json":
        "fe4a4b0f1c83a43bb00c1a231702c4ec8c282ecbb47d8b4e8f6279d5b33d2a71",
    "manifests/phase_6n_canonical_member_evidence_freeze_v1.json":
        "3d7585fdf355b7ced8e692a5076e6b90a26eea25ef43060968eaaea2b598dc8e",
}

SCHEMAS = {
    "pair_member": (
        "pair_member_record_schema",
        "PAIR_MEMBER_REQUIRED",
        "phase6h_csc_pair_member_record_v1",
    ),
    "sensor_reference": (
        "sensor_reference_record_schema",
        "SENSOR_REFERENCE_REQUIRED",
        "phase6h_csc_sensor_reference_record_v1",
    ),
    "csc_fault": (
        "csc_fault_record_schema",
        "CSC_FAULT_REQUIRED",
        "phase6h_csc_fault_window_record_v1",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pins() -> dict[str, Any]:
    for relative, expected in PINS.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(ABORT + ": source binding: " + relative)

    if Path(runtime.__file__).resolve() != (
        ROOT / "experiments/phase_06/csc_execution_runtime_v1.py"
    ).resolve():
        raise RuntimeError(ABORT + ": runtime module origin")

    if not (
        EXECUTION_AUTHORIZED is False
        and MODEL_FORWARD_AUTHORIZED is False
        and PRODUCTION_BODY_RELEASED is False
        and runtime.EXECUTION_AUTHORIZED is False
    ):
        raise RuntimeError(ABORT + ": execution state")

    architecture = json.loads(
        (ROOT / ARCH).read_text(encoding="utf-8")
    )

    if architecture["execution_authorized"] is not False:
        raise RuntimeError(ABORT + ": architecture authorization")

    for kind, (section, required_name, version) in SCHEMAS.items():
        schema = architecture[section]
        actual_required = getattr(runtime, required_name)

        if (
            set(schema["required_fields"]) != set(actual_required)
            or schema["schema_version"] != version
            or len(schema["required_fields"])
                != len(set(schema["required_fields"]))
        ):
            raise RuntimeError(ABORT + ": frozen schema " + kind)

    if set(runtime.OUTPUT_FILES) != {
        "pair_members.jsonl",
        "sensor_reference.jsonl",
        "csc_fault.jsonl",
        "metadata.json",
    }:
        raise RuntimeError(ABORT + ": output filename set")

    return architecture


def validate_record_shape(
    kind: str,
    row: dict[str, Any],
) -> None:
    if kind not in SCHEMAS or not isinstance(row, dict):
        raise RuntimeError(ABORT + ": record kind or type")

    _, required_name, version = SCHEMAS[kind]

    required = getattr(runtime, required_name)
    missing = set(required) - set(row)

    if missing or row.get("schema_version") != version:
        raise RuntimeError(ABORT + ": record schema " + kind)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(ABORT + ": " + reason)


def validate_bundle(
    result: dict[str, Any],
    request: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, int]:
    """Check exact relational identities and geometry for one member."""

    adapter.validate_execution_request(request)

    pair = result["pair_member"]
    references = result["sensor_reference_rows"]
    faults = result["csc_fault_rows"]

    validate_record_shape("pair_member", pair)

    exposed = list(request["sensor_exposed_window_indices"])
    compute = list(request["compute_execution_window_indices"])
    overlap = list(request["simultaneous_overlap_window_indices"])

    _require(
        len(references) == len(exposed)
        and len(faults) == len(compute),
        "row cardinalities",
    )

    for key, value in common.items():
        _require(pair.get(key) == value, "pair identity " + key)

    for key in (
        "target_name",
        "target_role",
        "representation_class",
        "persistence",
        "compute_sampling_instance_id",
        "element_index",
        "bit_position",
        "onset_or_inference_index",
    ):
        _require(
            pair.get(key) == request.get(key),
            "pair compute coordinate " + key,
        )

    _require(
        pair["sensor_exposed_window_indices"] == exposed
        and pair["compute_execution_window_indices"] == compute
        and pair["simultaneous_overlap_window_indices"] == overlap
        and pair["temporal_overlap_window_count"] == len(overlap)
        and pair["zero_temporal_overlap"] is (not bool(overlap))
        and pair["sensor_reference_record_count"] == len(references)
        and pair["csc_fault_record_count"] == len(faults)
        and pair["execution_status"] == "COMPLETE"
        and pair["metrics_generated"] is False,
        "pair temporal/count/status",
    )

    _require(
        overlap == sorted(set(exposed) & set(compute)),
        "temporal overlap identity",
    )

    for expected_index, row in zip(exposed, references):
        validate_record_shape("sensor_reference", row)

        for key, value in common.items():
            if key != "phase5_clean_cache_id":
                _require(
                    row.get(key) == value,
                    "sensor reference identity " + key,
                )

        _require(
            row["reference_window_index"] == expected_index
            and type(row["output_nonfinite"]) is bool
            and isinstance(row["output_values"], list)
            and isinstance(row["output_float32_hex"], list)
            and isinstance(row["softmax_values"], list),
            "sensor reference payload shape",
        )

    for expected_index, row in zip(compute, faults):
        validate_record_shape("csc_fault", row)

        for key, value in common.items():
            _require(
                row.get(key) == value,
                "CSC fault identity " + key,
            )

        active = expected_index in set(exposed)
        kind = (
            "phase6_sensor_reference"
            if active else "phase5_clean_cache"
        )

        _require(
            row["execution_window_index"] == expected_index
            and row["sensor_active_at_execution_window"] is active
            and row["simultaneous_sensor_compute_active"] is active
            and row["reference_kind"] == kind
            and type(row["faulted_output_nonfinite"]) is bool
            and isinstance(row["faulted_output_values"], list)
            and isinstance(row["faulted_output_float32_hex"], list)
            and isinstance(row["faulted_softmax_values"], list)
            and isinstance(row["mutation"], dict),
            "CSC reference route/payload shape",
        )

    return {
        "pair_members": 1,
        "sensor_references": len(references),
        "csc_fault_windows": len(faults),
        "simultaneous_overlap": len(overlap),
        "zero_overlap_members": int(not bool(overlap)),
    }


def validate_written_fixture(
    directory: Path,
    jobs: dict[str, dict[str, Any]],
    expected_counts: dict[str, int],
) -> dict[str, int]:
    """Validate canonical JSONL and exact per-request record coverage."""

    filenames = {
        "pair_members": runtime.PAIR_MEMBERS_JSONL,
        "sensor_references": runtime.SENSOR_REFERENCE_JSONL,
        "csc_fault_windows": runtime.CSC_FAULT_JSONL,
    }

    parsed: dict[str, list[dict[str, Any]]] = {}

    for category, filename in filenames.items():
        payload = (directory / filename).read_text(encoding="utf-8")
        lines = payload.splitlines(keepends=True)

        rows = []

        for line in lines:
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    ABORT + ": invalid JSONL"
                ) from exc

            _require(
                line == runtime.canonical_json(row) + "\n",
                "noncanonical JSONL",
            )
            rows.append(row)

        _require(
            len(rows) == expected_counts[category],
            "file cardinality " + category,
        )

        parsed[category] = rows

    metadata = json.loads(
        (directory / runtime.METADATA_JSON).read_text(
            encoding="utf-8"
        )
    )

    _require(
        metadata.get("synthetic_qualification_only") is True
        and metadata.get("scientific_outputs_present") is False
        and metadata.get("execution_authorized") is False
        and metadata.get("expected_counts") == expected_counts,
        "fixture metadata coverage",
    )

    pair_rows = parsed["pair_members"]

    _require(
        len({
            row["execution_request_id"] for row in pair_rows
        }) == len(jobs),
        "unique request identities",
    )

    by_reference = defaultdict(list)
    by_fault = defaultdict(list)

    for row in parsed["sensor_references"]:
        by_reference[row["execution_request_id"]].append(row)

    for row in parsed["csc_fault_windows"]:
        by_fault[row["execution_request_id"]].append(row)

    totals = Counter()

    for pair in pair_rows:
        request_id = pair["execution_request_id"]
        job = jobs.get(request_id)

        _require(
            job is not None,
            "unknown request in output",
        )

        result = {
            "pair_member": pair,
            "sensor_reference_rows": by_reference.pop(
                request_id, []
            ),
            "csc_fault_rows": by_fault.pop(request_id, []),
        }

        totals.update(
            validate_bundle(
                result,
                job["request"],
                job["common"],
            )
        )

    _require(
        not by_reference and not by_fault,
        "orphan output record",
    )

    for key in filenames:
        _require(
            totals[key] == expected_counts[key],
            "aggregated file count " + key,
        )

    return dict(totals)


def qualify_synthetic_record_contract() -> dict[str, Any]:
    check_pins()

    previous = json.loads(
        (
            ROOT
            / "manifests/phase_6m_pre_forward_evidence_v1/"
            "phase6m_canonical_request_synthetic_witness_v1.json"
        ).read_text(encoding="utf-8")
    )

    _require(
        previous["qualification_status"]
        == "FROZEN_SUBJECT_REQUEST_WITNESSES_PASS"
        and previous["distinct_pair_witness_count"] == 37
        and previous["request_counts"]["requests"] == 189
        and previous["execution_authorized"] is False,
        "previous witness",
    )

    previous_ids = {
        row["execution_request_id"]
        for row in previous["requests"]
    }

    plan = preflight.FrozenCSCPreForwardPlan()
    context = auditor.FrozenContext()
    capture = witness.CapturingSubjectAudit(context, 9)
    audit = capture.run()

    _require(
        len(audit["rows"]) == 12 and capture.fold == 5,
        "frozen subject metadata",
    )

    examples = {
        item["sensor_fault_id"]: item
        for item in capture.examples_by_tag.values()
    }

    _require(len(examples) == 37, "pair sample census")

    bindings = adapter.validate_pinned_bindings()

    all_pairs = []
    all_refs = []
    all_faults = []
    jobs = {}
    totals = Counter()
    negatives_checked = 0

    for example in sorted(
        examples.values(),
        key=lambda row: (
            row["sensor_family"],
            row["task"],
            row["trial"],
            row["sensor_fault_id"],
        ),
    ):
        pair = example["pair_metadata"]
        exposed = list(example["sensor_exposed_window_indices"])

        for variant in pair["eligible_model_variants"]:
            for seed in pair["checkpoint_seeds"]:
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

                clean = plan._cache_by_member[
                    (5, 9, variant, int(seed))
                ]

                shard = runtime.shard_id(
                    fold=5,
                    subject=9,
                    sensor_family=example["sensor_family"],
                )

                _require(
                    clean["clean_cache_id"] in plan.shard(shard)[
                        "prospective_candidate_clean_cache_ids"
                    ],
                    "clean cache ownership",
                )

                reference_id = runtime.sensor_reference_cache_id(
                    fold=5,
                    subject=9,
                    task=example["task"],
                    trial=example["trial"],
                    sensor_parent_kind=example["sensor_parent_kind"],
                    parent_local_index=example["parent_local_index"],
                    sensor_fault_id=example["sensor_fault_id"],
                    sensor_replay_id=example["sensor_replay_id"],
                    model_variant=variant,
                    checkpoint_seed=int(seed),
                )

                common = {
                    "shard_id": shard,
                    "execution_request_id":
                        request["execution_request_id"],
                    "fold": 5,
                    "subject": 9,
                    "task": int(example["task"]),
                    "trial": int(example["trial"]),
                    "sensor_parent_kind":
                        example["sensor_parent_kind"],
                    "parent_local_index":
                        int(example["parent_local_index"]),
                    "sensor_fault_id": example["sensor_fault_id"],
                    "sensor_replay_id": example["sensor_replay_id"],
                    "model_variant": variant,
                    "checkpoint_seed": int(seed),
                    "sensor_reference_cache_id": reference_id,
                    "phase5_clean_cache_id":
                        clean["clean_cache_id"],
                }

                indices = (
                    set(exposed)
                    | set(request["compute_execution_window_indices"])
                )

                result = runtime.synthetic_execute_pair_member(
                    request=request,
                    shard_id_value=shard,
                    fold=5,
                    subject=9,
                    task=int(example["task"]),
                    trial=int(example["trial"]),
                    parent_local_index=int(
                        example["parent_local_index"]
                    ),
                    phase5_clean_cache_id=clean["clean_cache_id"],
                    sensor_reference_cache_id_value=reference_id,
                    windows_by_index={
                        index: f"SYNTHETIC_WINDOW_{index}"
                        for index in indices
                    },
                    hooks=runtime.SyntheticHooks(
                        reference_forward=witness._reference_summary,
                        fault_sequence=witness._fault_summaries,
                    ),
                )

                observed = validate_bundle(result, request, common)
                totals.update(observed)

                request_id = request["execution_request_id"]
                _require(
                    request_id not in jobs,
                    "duplicate execution request",
                )

                jobs[request_id] = {
                    "request": request,
                    "common": common,
                }

                all_pairs.append(result["pair_member"])
                all_refs.extend(result["sensor_reference_rows"])
                all_faults.extend(result["csc_fault_rows"])

                # A structural mutation must be rejected independently
                # of output-file SHA-256 checks.
                if negatives_checked == 0:
                    bad = copy.deepcopy(result)
                    bad["pair_member"][
                        "sensor_reference_record_count"
                    ] += 1

                    try:
                        validate_bundle(bad, request, common)
                    except RuntimeError as exc:
                        _require(
                            ABORT in str(exc),
                            "negative test exception",
                        )
                        negatives_checked += 1
                    else:
                        raise RuntimeError(
                            ABORT + ": wrong coverage accepted"
                        )

    _require(
        set(jobs) == previous_ids,
        "previous Phase6M request IDs",
    )

    expected_counts = {
        "pair_members": 189,
        "sensor_references": 5310,
        "csc_fault_windows": 13989,
    }

    for key, expected in expected_counts.items():
        _require(
            totals[key] == expected,
            "frozen witness record count " + key,
        )

    with tempfile.TemporaryDirectory(
        prefix="phase6o-record-contract-synthetic-"
    ) as directory:
        temporary = Path(directory)

        runtime.write_jsonl(
            temporary / runtime.PAIR_MEMBERS_JSONL,
            all_pairs,
        )
        runtime.write_jsonl(
            temporary / runtime.SENSOR_REFERENCE_JSONL,
            all_refs,
        )
        runtime.write_jsonl(
            temporary / runtime.CSC_FAULT_JSONL,
            all_faults,
        )
        runtime.write_json(
            temporary / runtime.METADATA_JSON,
            {
                "synthetic_qualification_only": True,
                "scientific_outputs_present": False,
                "execution_authorized": False,
                "expected_counts": expected_counts,
            },
        )

        written = validate_written_fixture(
            temporary,
            jobs,
            expected_counts,
        )

        _require(
            written["pair_members"] == 189
            and written["sensor_references"] == 5310
            and written["csc_fault_windows"] == 13989,
            "round-trip record census",
        )

        output_hashes = {
            name: sha(temporary / name)
            for name in runtime.OUTPUT_FILES
        }

        _require(
            len(output_hashes) == 4,
            "four-file fixture set",
        )

        _require(
            not (temporary / runtime.SUCCESS_MARKER).exists(),
            "unexpected success marker",
        )

        temporary_path = temporary

    _require(
        not temporary_path.exists(),
        "temporary fixture cleanup",
    )

    return {
        "qualification_status":
            "PHASE6O_FROZEN_SYNTHETIC_RECORD_CONTRACT_PASS",
        "scope": "189_FROZEN_REQUESTS_WITH_SYNTHETIC_OUTPUT_VALUES",
        "validated_request_count": len(jobs),
        "validated_record_counts": expected_counts,
        "synthetic_overlap_records": totals["simultaneous_overlap"],
        "zero_overlap_pair_member_count":
            totals["zero_overlap_members"],
        "negative_coverage_mutation_rejected": True,
        "canonical_jsonl_round_trip_passed": True,
        "all_frozen_required_fields_verified": True,
        "per_request_relational_identities_verified": True,
        "per_request_temporal_reference_routing_verified": True,
        "four_synthetic_fixture_file_sha256": output_hashes,
        "temporary_fixture_cleanup_verified": True,
        "permanent_phase6h_success_marker_created": False,
        "full_shard_frozen_cardinality_verified_by_this_stage": False,
        "production_coverage_marker_validator_complete": False,
        "scientific_prediction_values_validated": False,
        "real_sensor_inputs_read": False,
        "real_label_arrays_read": False,
        "real_model_loaded": False,
        "real_model_forward_performed": False,
        "real_csc_execution_performed": False,
        "historical_phase6e_digests_reproduced": False,
        "execution_authorized": False,
    }


def execute_shard(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(EXECUTION_BLOCK)
