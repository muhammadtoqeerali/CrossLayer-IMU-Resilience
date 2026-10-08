"""Phase6K independent prospective CSC metadata workload auditor.

No sensor arrays, label arrays, probabilities, checkpoints, fault
operators, or model forwards are accessed. Raw source CSV records are
counted without parsing their fields.

The Phase6E historical stream digests are NOT replaced or claimed to
be reproduced. Phase6K creates independent, explicitly versioned
descriptor digests and checks frozen numerical workload invariants.

This module does not authorize CSC execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import csc_outer_executor_v1 as csc
import csc_execution_runtime_v1 as runtime
import csc_source_trial_exposure_v1 as exposure
import sensor_fi_outer_executor_v1 as p4


ROOT = Path(__file__).resolve().parents[2]

PLAN_SHA256 = (
    "888d90f720682d19cfed6ea211d84b12b4ea8fb1408aaef36cffdb7301975bea"
)

SUBJECT10_REFERENCE_SHA256 = (
    "4480cf647fc427f5c796e5c065ec3301e7668780274be71d24877edf93754f13"
)

EXPECTED_TOTALS = {
    "shards": 732,
    "stored_pairs": 4107450,
    "source_groups": 132489,
    "source_retained": 130385,
    "source_omitted": 2104,
    "pairs": 4237835,
    "members": 21793038,
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(relative):
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8")
    )


def canonical_line(payload):
    return (
        csc.canonical_json(payload) + "\n"
    ).encode("utf-8")


def level_of(instance):
    if "severity_level" in instance:
        return str(instance["severity_level"])
    return str(instance["severity"]["level"])


def pair_contribution(pair):
    """Derive numeric workload from a frozen metadata-only pair."""
    timing = pair["temporal_accounting"]

    sensor = int(timing["sensor_exposed_window_count"])
    compute = int(timing["compute_active_window_count"])
    overlap = int(timing["temporal_overlap_window_count"])
    union = int(timing["sensor_compute_union_window_count"])

    multiplier = int(pair["pair_member_multiplier"])

    if sensor < 1 or compute < 1:
        raise ValueError("empty retained pair exposure/workload")

    if not 0 <= overlap <= min(sensor, compute):
        raise ValueError("invalid temporal overlap")

    if union != sensor + compute - overlap:
        raise ValueError("temporal union identity failed")

    if multiplier not in (3, 6):
        raise ValueError("invalid checkpoint/variant expansion")

    persistence = pair["persistence"]

    if persistence == "transient_one_inference":
        if compute != 1 or overlap != 1:
            raise ValueError("invalid transient overlap")
    elif persistence == "persistent_from_onset_until_trial_end":
        pass
    else:
        raise ValueError("unknown persistence")

    if overlap == 0 and persistence != (
        "persistent_from_onset_until_trial_end"
    ):
        raise ValueError("zero-overlap transient pair")

    return {
        "pair_count": 1,
        "pair_member_count": multiplier,
        "sensor_exposed_window_sum": sensor,
        "compute_active_window_sum": compute,
        "simultaneous_overlap_window_sum": overlap,
        "sensor_compute_union_window_sum": union,
        "member_sensor_reference_window_sum": multiplier * sensor,
        "member_compute_faulted_window_sum": multiplier * compute,
        "member_overlap_window_sum": multiplier * overlap,
        "member_union_window_sum": multiplier * union,
    }


def add_counter(target, contribution):
    for key, value in contribution.items():
        target[key] += int(value)


def assert_fields_equal(actual, frozen, fields, label):
    for field in fields:
        observed = int(actual.get(field, 0))
        expected = int(frozen[field])

        if observed != expected:
            raise ValueError(
                f"{label}.{field}: observed={observed}, "
                f"frozen={expected}"
            )


WORKLOAD_FIELDS = (
    "pair_count",
    "pair_member_count",
    "sensor_exposed_window_sum",
    "compute_active_window_sum",
    "simultaneous_overlap_window_sum",
    "sensor_compute_union_window_sum",
    "member_compute_faulted_window_sum",
    "member_overlap_window_sum",
)


class FrozenContext:
    def __init__(self):
        if csc.EXECUTION_ENABLED is not False:
            raise RuntimeError("metadata executor unexpectedly enabled")

        if runtime.EXECUTION_AUTHORIZED is not False:
            raise RuntimeError("CSC execution unexpectedly authorized")

        if exposure.validate_pinned_exposure_contract()[
            "status"
        ] != "PINNED_EXPOSURE_CONTRACT_VALID":
            raise RuntimeError("exposure contract invalid")

        plan_path = (
            ROOT
            / "configs/evaluation/phase6e_csc_execution_plan_v1.json"
        )

        if sha256_file(plan_path) != PLAN_SHA256:
            raise RuntimeError("Phase6E frozen plan hash mismatch")

        self.plan = load(
            "configs/evaluation/phase6e_csc_execution_plan_v1.json"
        )

        for dependency in self.plan["frozen_dependencies"].values():
            if not isinstance(dependency, dict):
                continue
            if "path" not in dependency:
                continue

            path = Path(dependency["path"])

            if not path.is_absolute():
                path = ROOT / path

            if not path.is_file():
                raise RuntimeError(
                    "missing frozen dependency: " + str(path)
                )

            if sha256_file(path) != dependency["sha256"]:
                raise RuntimeError(
                    "frozen dependency hash mismatch: " + str(path)
                )

        self.phase6d = load(
            "configs/evaluation/phase6d_csc_pairing_protocol_v1.json"
        )
        self.r1 = load(
            "configs/evaluation/phase6d_r1_csc_pairing_clarification_v1.json"
        )
        self.phase5d = load(
            "configs/evaluation/phase5d_compute_fi_outer_protocol_v1.json"
        )
        self.p4cfg = load(
            "configs/evaluation/phase4h_sensor_fi_outer_executor_v1.json"
        )
        self.severity = load(
            "configs/faults/phase4h_sensor_fi_severity_protocol_v1.json"
        )
        self.p3k = load(
            "manifests/phase_3k_event_annotation_trial_mapping_v1.json"
        )
        self.p5e = load(
            "manifests/phase_5e_compute_fi_outer_execution_plan_v1.json"
        )

        self.validated = {
            "plan": self.plan,
            "phase6d": self.phase6d,
            "phase6d_r1": self.r1,
            "phase5d": self.phase5d,
        }

        self.runner, _, self.sampling = (
            p4.load_runtime_modules(self.p4cfg)
        )

        binding = self.p4cfg[
            "frozen_dependencies"
        ]["risk_index"]

        risk_path = Path(binding["path"])

        if not risk_path.is_absolute():
            risk_path = ROOT / risk_path

        if sha256_file(risk_path) != binding["sha256"]:
            raise RuntimeError("frozen risk-index hash mismatch")

        _, self.risk_index = self.runner.load_risk_index(
            risk_path
        )

        self.univr_root = Path(
            self.p4cfg["execution_locations"]["univr_source_root"]
        )
        self.kfall_root = Path(
            self.p4cfg["execution_locations"]["kfall_source_root"]
        )

        if not self.univr_root.is_dir():
            raise RuntimeError("UniVR raw source root missing")

        if not self.kfall_root.is_dir():
            raise RuntimeError("KFall raw source root missing")

        self.p3 = {
            (
                int(row["storage_subject"]),
                int(row["task"]),
                int(row["trial"]),
            ): row
            for row in self.p3k[
                "processed_primary_trials"
            ]["records"]
        }

        self.p5 = {
            (
                int(row["subject"]),
                int(row["task"]),
                int(row["trial"]),
            ): row
            for row in self.p5e["trial_inventory"]
        }

        if len(self.p3) != 6309 or len(self.p5) != 6309:
            raise RuntimeError("trial inventory count changed")

        if set(self.p3) != set(self.p5):
            raise RuntimeError("Phase3K/Phase5E trial identities differ")

        self.subject_folds = defaultdict(set)
        self.subject_windows = Counter()
        self.trials_by_subject = defaultdict(list)

        activity = 0
        falling = 0

        for key, row in self.p5.items():
            subject = key[0]
            windows = int(row["window_count"])
            p3row = self.p3[key]

            counts = p3row["label_counts"]
            a = int(counts.get("Activity", 0))
            f = int(counts.get("Falling", 0))

            if a + f != windows:
                raise RuntimeError(
                    "trial window/class count mismatch: " + repr(key)
                )

            activity += a
            falling += f

            self.subject_folds[subject].add(int(row["fold"]))
            self.subject_windows[subject] += windows
            self.trials_by_subject[subject].append(key)

        if len(self.subject_folds) != 61:
            raise RuntimeError("outer subject count changed")

        if any(
            len(folds) != 1
            for folds in self.subject_folds.values()
        ):
            raise RuntimeError("subject belongs to multiple folds")

        if (activity, falling) != (264024, 9806):
            raise RuntimeError("frozen class totals changed")

        for keys in self.trials_by_subject.values():
            keys.sort()

        self.stored_families = (
            "axis_loss",
            "bias",
            "clipping_saturation",
            "noise",
            "scale_factor",
        )

        self.source_families = (
            "delay",
            "drift",
            "dropout",
            "frame_loss",
            "jitter",
            "orientation",
            "stuck_channel",
        )

        if (
            len(self.stored_families)
            + len(self.source_families)
        ) != 12:
            raise RuntimeError("sensor family census changed")

    def source_length(self, key):
        subject, task, trial = key

        path = self.runner.source_trial_path(
            subject=subject,
            task=task,
            trial=trial,
            univr_root=self.univr_root,
            kfall_root=self.kfall_root,
        )

        if not path.is_file():
            raise RuntimeError(
                "missing raw source trial: " + str(path)
            )

        # Metadata-only line count: never parse source CSV values.
        with path.open("rb") as handle:
            length = sum(1 for _ in handle) - 1

        if length < 30:
            raise RuntimeError("invalid source length: " + str(path))

        return length

    def historical_ends(self, key):
        subject, task, trial = key

        counts = self.p3[key]["label_counts"]
        activity = int(counts.get("Activity", 0))
        falling = int(counts.get("Falling", 0))

        ends = [
            30 + 15 * index
            for index in range(activity)
        ]

        risk = p4.risk_row(
            self.risk_index,
            subject=subject,
            task=task,
            trial=trial,
        )

        if falling:
            if risk is None:
                raise RuntimeError(
                    "Falling trial without risk row: " + repr(key)
                )

            start = int(risk.fall_start_position)

            if start < 0:
                raise RuntimeError("negative fall start position")

            ends.extend(
                start + 30 + 15 * index
                for index in range(falling)
            )

        if len(ends) != int(self.p5[key]["window_count"]):
            raise RuntimeError("historical window count differs")

        return ends


class SubjectAudit:
    def __init__(self, context, subject):
        self.ctx = context
        self.subject = int(subject)

        if self.subject not in context.subject_folds:
            raise ValueError("unknown frozen outer subject")

        self.fold = next(iter(
            context.subject_folds[self.subject]
        ))

        self.keys = context.trials_by_subject[self.subject]

        self.by_family = defaultdict(Counter)
        self.descriptor_hashers = defaultdict(hashlib.sha256)

        self.by_stratum = defaultdict(Counter)
        self.by_parent_persistence = defaultdict(Counter)
        self.by_variant = defaultdict(Counter)

        self.totals = Counter()

        self.selected_fault_ids = set()
        self.selected_replay_ids = set()

        self.source_candidate_count = 0
        self.source_observable_candidate_count = 0
        self.source_group_count = 0
        self.source_omission_count = 0

    def add_pair(self, key, family, level, kind, instance, exposed):
        subject, task, trial = key

        fid = str(instance["fault_id"])
        rid = str(instance["replay_id"])

        if fid in self.selected_fault_ids:
            raise RuntimeError("duplicate selected fault identity")

        if rid in self.selected_replay_ids:
            raise RuntimeError("duplicate selected replay identity")

        self.selected_fault_ids.add(fid)
        self.selected_replay_ids.add(rid)

        pair = csc.derive_pair_metadata(
            sensor_parent_kind=kind,
            sensor_instance=instance,
            sensor_exposed_window_indices=exposed,
            fold=self.fold,
            subject=subject,
            task=task,
            trial=trial,
            trial_window_count=int(
                self.ctx.p5[key]["window_count"]
            ),
            validated=self.ctx.validated,
        )

        if pair["execution_performed"] is not False:
            raise RuntimeError("metadata-only boundary violated")

        idx = int(pair["compute_stratum_index"])

        if not 0 <= idx < 28:
            raise RuntimeError("invalid compute stratum")

        frozen_stratum = self.ctx.plan["compute_strata"][idx]

        if pair["target_name"] != frozen_stratum["target_name"]:
            raise RuntimeError("compute target mismatch")

        if pair["persistence"] != frozen_stratum["persistence"]:
            raise RuntimeError("persistence mismatch")

        if (
            pair["eligible_model_variants"]
            != frozen_stratum["eligible_model_variants"]
        ):
            raise RuntimeError("model eligibility mismatch")

        if pair["checkpoint_seeds"] != [42, 123, 2025]:
            raise RuntimeError("checkpoint seed mismatch")

        contribution = pair_contribution(pair)

        timing = pair["temporal_accounting"]

        sensor = int(timing["sensor_exposed_window_count"])
        compute = int(timing["compute_active_window_count"])
        overlap = int(timing["temporal_overlap_window_count"])
        union = int(timing["sensor_compute_union_window_count"])

        persistence = pair["persistence"]
        multiplier = int(pair["pair_member_multiplier"])

        add_counter(self.totals, contribution)
        add_counter(self.by_family[family], contribution)
        add_counter(self.by_stratum[idx], contribution)

        bucket = f"{kind}|{persistence}"

        add_counter(
            self.by_parent_persistence[bucket],
            contribution,
        )

        self.by_family[family]["pairs"] += 1

        for variant in pair["eligible_model_variants"]:
            v = self.by_variant[variant]
            v["pair_count"] += 1
            v["pair_member_count"] += 3
            v["compute_faulted_window_count"] += 3 * compute
            v["sensor_reference_window_count"] += 3 * sensor
            v["simultaneous_overlap_window_count"] += 3 * overlap
            v["sensor_compute_union_window_count"] += 3 * union

            if persistence == "transient_one_inference":
                v["transient_pair_count"] += 1
            else:
                v["persistent_pair_count"] += 1

        if overlap == 0:
            self.totals["zero_overlap_pairs"] += 1
            self.totals["zero_overlap_members"] += multiplier

            self.by_family[family]["zero_overlap_pairs"] += 1
            self.by_family[family]["zero_overlap_members"] += multiplier

            self.by_parent_persistence[bucket][
                "zero_overlap_pairs"
            ] += 1

            self.by_parent_persistence[bucket][
                "zero_overlap_members"
            ] += multiplier

            self.totals[
                f"zero_overlap_{kind}_pairs"
            ] += 1

            self.totals[
                f"zero_overlap_{kind}_members"
            ] += multiplier

        descriptor = {
            "schema_version":
                "phase6k_subject_metadata_audit_descriptor_v1",
            "fold": self.fold,
            "subject": subject,
            "task": task,
            "trial": trial,
            "sensor_family": family,
            "sensor_severity": level,
            "sensor_parent_kind": kind,
            "sensor_fault_id": fid,
            "sensor_replay_id": rid,
            "compute_stratum_index": idx,
            "compute_coordinate": pair["compute_coordinate"],
            "temporal_counts": {
                "sensor": sensor,
                "compute": compute,
                "overlap": overlap,
                "union": union,
            },
            "pair_member_multiplier": multiplier,
        }

        self.descriptor_hashers[family].update(
            canonical_line(descriptor)
        )

    def audit_stored(self):
        c = self.ctx

        for key in self.keys:
            subject, task, trial = key
            window_count = int(c.p5[key]["window_count"])

            for window_index in range(window_count):
                parent_id = (
                    f"subject={subject}"
                    f"|task={task}"
                    f"|trial={trial}"
                    f"|window={window_index}"
                )

                candidates = c.sampling.generate_window_instances(
                    fold=self.fold,
                    partition="outer_test",
                    parent_sequence_id=parent_id,
                    severity_protocol=c.severity,
                )

                if len(candidates) != 153:
                    raise RuntimeError("stored candidate count changed")

                grouped = defaultdict(list)

                for candidate in candidates:
                    grouped[
                        (
                            str(candidate["family"]),
                            level_of(candidate),
                        )
                    ].append(candidate)

                if len(grouped) != 15:
                    raise RuntimeError("stored family/severity groups changed")

                for (family, level), group in sorted(grouped.items()):
                    if family not in c.stored_families:
                        raise RuntimeError("unrecognized stored family")

                    chosen = csc.select_stored_window_candidate(
                        candidates=group,
                        fold=self.fold,
                        subject=subject,
                        family=family,
                        severity=level,
                        phase6d=c.phase6d,
                    )

                    self.add_pair(
                        key,
                        family,
                        level,
                        "stored_window",
                        chosen,
                        [window_index],
                    )

    def audit_source(self):
        c = self.ctx

        for key in self.keys:
            subject, task, trial = key
            length = c.source_length(key)
            ends = c.historical_ends(key)

            parent_id = (
                f"subject={subject}"
                f"|task={task}"
                f"|trial={trial}"
            )

            candidates = c.sampling.generate_sequence_instances(
                fold=self.fold,
                partition="outer_test",
                parent_sequence_id=parent_id,
                parent_length=length,
                severity_protocol=c.severity,
            )

            if len(candidates) != 138:
                raise RuntimeError("source candidate count changed")

            grouped = defaultdict(list)

            for candidate in candidates:
                grouped[
                    (
                        str(candidate["family"]),
                        level_of(candidate),
                    )
                ].append(candidate)

            if len(grouped) != 21:
                raise RuntimeError("source family/severity groups changed")

            for (family, level), group in sorted(grouped.items()):
                if family not in c.source_families:
                    raise RuntimeError("unrecognized source family")

                self.source_group_count += 1
                self.by_family[family]["original_groups"] += 1

                exposure_by_fault = {}

                for candidate in group:
                    fid = str(candidate["fault_id"])

                    if fid in exposure_by_fault:
                        raise RuntimeError("duplicate source candidate ID")

                    windows = (
                        exposure.source_trial_exposed_window_indices(
                            instance=candidate,
                            historical_window_ends=ends,
                            source_length=length,
                        )
                    )

                    exposure_by_fault[fid] = windows
                    self.source_candidate_count += 1

                    if windows:
                        self.source_observable_candidate_count += 1

                decision = csc.select_source_trial_candidate(
                    candidates=group,
                    exposure_count_by_fault_id={
                        fid: len(windows)
                        for fid, windows in exposure_by_fault.items()
                    },
                    fold=self.fold,
                    subject=subject,
                    family=family,
                    severity=level,
                    phase6d=c.phase6d,
                )

                if decision["status"] == (
                    "STRUCTURALLY_INELIGIBLE_NO_CSC_PAIR"
                ):
                    if decision["selected"] is not None:
                        raise RuntimeError("omission selected candidate")

                    self.source_omission_count += 1
                    self.by_family[family][
                        "structural_omissions"
                    ] += 1
                    continue

                if decision["status"] != "ELIGIBLE_SELECTED":
                    raise RuntimeError("unexpected R2 selection status")

                chosen = decision["selected"]

                selected_exposure = exposure_by_fault[
                    str(chosen["fault_id"])
                ]

                if not selected_exposure:
                    raise RuntimeError("unobservable selected source pair")

                self.add_pair(
                    key,
                    family,
                    level,
                    "source_trial",
                    chosen,
                    selected_exposure,
                )

    def run(self):
        self.audit_stored()
        self.audit_source()

        c = self.ctx
        families = sorted(
            c.stored_families + c.source_families
        )

        rows = []

        for family in families:
            kind = (
                "stored_window"
                if family in c.stored_families
                else "source_trial"
            )

            counts = self.by_family[family]

            if kind == "stored_window":
                expected = 3 * c.subject_windows[self.subject]

                if counts["pairs"] != expected:
                    raise RuntimeError("stored shard cardinality mismatch")

                if counts["structural_omissions"]:
                    raise RuntimeError("stored structural omission")
            else:
                expected = 3 * len(self.keys)

                if (
                    counts["pairs"]
                    + counts["structural_omissions"]
                    != expected
                ):
                    raise RuntimeError("source shard cardinality mismatch")

            if counts["pairs"] <= 0:
                raise RuntimeError("zero-pair shard")

            row = {
                "runtime_shard_id": runtime.shard_id(
                    fold=self.fold,
                    subject=self.subject,
                    sensor_family=family,
                ),
                "fold": self.fold,
                "subject": self.subject,
                "sensor_family": family,
                "sensor_parent_kind": kind,
                "pair_count": counts["pairs"],
                "pair_member_count": counts["pair_member_count"],
                "structural_omission_count":
                    counts["structural_omissions"],
                "zero_overlap_pair_count":
                    counts["zero_overlap_pairs"],
                "sensor_reference_member_windows":
                    counts["member_sensor_reference_window_sum"],
                "compute_faulted_member_windows":
                    counts["member_compute_faulted_window_sum"],
                "simultaneous_overlap_member_windows":
                    counts["member_overlap_window_sum"],
                "union_member_windows":
                    counts["member_union_window_sum"],
                "new_phase6k_descriptor_sha256":
                    self.descriptor_hashers[family].hexdigest(),
            }

            if row["union_member_windows"] != (
                row["sensor_reference_member_windows"]
                + row["compute_faulted_member_windows"]
                - row["simultaneous_overlap_member_windows"]
            ):
                raise RuntimeError("per-shard union identity mismatch")

            rows.append(row)

        if len(rows) != 12:
            raise RuntimeError("subject does not yield 12 shards")

        if sum(
            r["pair_count"] for r in rows
        ) != self.totals["pair_count"]:
            raise RuntimeError("subject pair count mismatch")

        if len(self.selected_fault_ids) != self.totals["pair_count"]:
            raise RuntimeError("selected fault identity mismatch")

        if len(self.selected_replay_ids) != self.totals["pair_count"]:
            raise RuntimeError("selected replay identity mismatch")

        if len(self.by_stratum) != 28:
            raise RuntimeError("not all compute strata represented")

        if self.totals["member_union_window_sum"] != (
            self.totals["member_sensor_reference_window_sum"]
            + self.totals["member_compute_faulted_window_sum"]
            - self.totals["member_overlap_window_sum"]
        ):
            raise RuntimeError("subject union identity mismatch")

        payload = {
            "schema_version":
                "phase6k_complete_subject_metadata_audit_v1",
            "subject": self.subject,
            "fold": self.fold,
            "trial_count": len(self.keys),
            "window_count": c.subject_windows[self.subject],
            "rows": rows,
        }

        return {
            "subject": self.subject,
            "fold": self.fold,
            "trial_count": len(self.keys),
            "stored_window_count": c.subject_windows[self.subject],
            "source_groups": self.source_group_count,
            "source_omissions": self.source_omission_count,
            "source_candidates": self.source_candidate_count,
            "source_observable_candidates":
                self.source_observable_candidate_count,
            "subject_inventory_sha256":
                hashlib.sha256(canonical_line(payload)).hexdigest(),
            "rows": rows,
            "totals": dict(self.totals),
            "strata": {
                str(k): dict(v)
                for k, v in sorted(self.by_stratum.items())
            },
            "parent_persistence": {
                key: dict(value)
                for key, value in sorted(
                    self.by_parent_persistence.items()
                )
            },
            "variant": {
                key: dict(value)
                for key, value in sorted(self.by_variant.items())
            },
        }


def verify_full_global(context, results):
    plan = context.plan
    totals = Counter()

    stratum_totals = defaultdict(Counter)
    parent_persistence_totals = defaultdict(Counter)
    variant_totals = defaultdict(Counter)

    rows = []
    source_groups = 0
    source_omissions = 0

    for result in results:
        rows.extend(result["rows"])

        source_groups += result["source_groups"]
        source_omissions += result["source_omissions"]

        add_counter(totals, result["totals"])

        for key, values in result["strata"].items():
            add_counter(stratum_totals[int(key)], values)

        for key, values in result["parent_persistence"].items():
            add_counter(parent_persistence_totals[key], values)

        for key, values in result["variant"].items():
            add_counter(variant_totals[key], values)

    if len(rows) != EXPECTED_TOTALS["shards"]:
        raise RuntimeError("global shard cardinality mismatch")

    ids = [row["runtime_shard_id"] for row in rows]

    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate runtime shard ID")

    stored = sum(
        row["pair_count"]
        for row in rows
        if row["sensor_parent_kind"] == "stored_window"
    )

    source = sum(
        row["pair_count"]
        for row in rows
        if row["sensor_parent_kind"] == "source_trial"
    )

    assert_fields_equal(
        {
            "stored_pairs": stored,
            "source_groups": source_groups,
            "source_retained": source,
            "source_omitted": source_omissions,
            "pairs": totals["pair_count"],
            "members": totals["pair_member_count"],
            "shards": len(rows),
        },
        EXPECTED_TOTALS,
        tuple(EXPECTED_TOTALS),
        "global_cardinality",
    )

    frozen_strata = plan["compute_strata"]

    if len(frozen_strata) != 28 or len(stratum_totals) != 28:
        raise RuntimeError("28-stratum census mismatch")

    for frozen in frozen_strata:
        index = int(frozen["index"])

        assert_fields_equal(
            stratum_totals[index],
            frozen,
            WORKLOAD_FIELDS,
            f"stratum[{index}]",
        )

    expected_parent = plan[
        "temporal_workload"
    ]["parent_persistence_workload"]

    if set(expected_parent) != set(parent_persistence_totals):
        raise RuntimeError("parent-persistence surface mismatch")

    for key, frozen in expected_parent.items():
        assert_fields_equal(
            parent_persistence_totals[key],
            frozen,
            WORKLOAD_FIELDS
            + (
                "member_sensor_reference_window_sum",
                "member_union_window_sum",
            ),
            "parent_persistence[" + key + "]",
        )

    frozen_variants = plan[
        "temporal_workload"
    ]["variant_workload"]

    for variant, frozen in frozen_variants.items():
        assert_fields_equal(
            variant_totals[variant],
            frozen,
            (
                "pair_member_count",
                "compute_faulted_window_count",
                "sensor_reference_window_count",
                "simultaneous_overlap_window_count",
                "sensor_compute_union_window_count",
            ),
            "variant[" + variant + "]",
        )

    for variant in ("fp32", "ptq_v7"):
        assert_fields_equal(
            variant_totals[variant],
            plan["model_variant_surface"][variant],
            (
                "pair_count",
                "pair_member_count",
            ) if False else ("pair_count",),
            "variant_surface[" + variant + "]",
        )

        v = variant_totals[variant]

        if v["pair_member_count"] != plan[
            "model_variant_surface"
        ][variant]["seed_expanded_pair_member_count"]:
            raise RuntimeError(
                "seed-expanded pair-member count mismatch"
            )

        for field in (
            "transient_pair_count",
            "persistent_pair_count",
        ):
            if v[field] != plan[
                "model_variant_surface"
            ][variant][field]:
                raise RuntimeError(
                    f"{variant}.{field} frozen mismatch"
                )

    persistence = plan["persistence_surface"]

    for kind in ("stored_window", "source_trial"):
        for short, long_name in (
            ("transient", "transient_one_inference"),
            ("persistent", "persistent_from_onset_until_trial_end"),
        ):
            key = kind + "|" + long_name
            observed = parent_persistence_totals[key]["pair_count"]
            expected = persistence[kind][short + "_pair_count"]

            if observed != expected:
                raise RuntimeError(
                    "parent persistence count mismatch: " + key
                )

    frozen_global = plan["temporal_workload"]

    global_pairs = {
        "sensor_reference_member_window_count":
            totals["member_sensor_reference_window_sum"],
        "compute_faulted_member_window_count":
            totals["member_compute_faulted_window_sum"],
        "simultaneous_csc_overlap_member_window_count":
            totals["member_overlap_window_sum"],
        "sensor_compute_union_member_window_count":
            totals["member_union_window_sum"],
        "zero_temporal_overlap_pair_count":
            totals["zero_overlap_pairs"],
        "zero_temporal_overlap_pair_member_count":
            totals["zero_overlap_members"],
    }

    assert_fields_equal(
        global_pairs,
        frozen_global,
        tuple(global_pairs),
        "global_workload",
    )

    for kind in ("stored_window", "source_trial"):
        observed = {
            "pair_count":
                totals[f"zero_overlap_{kind}_pairs"],
            "pair_member_count":
                totals[f"zero_overlap_{kind}_members"],
        }

        assert_fields_equal(
            observed,
            frozen_global["zero_overlap_by_parent_kind"][kind],
            ("pair_count", "pair_member_count"),
            "zero_overlap[" + kind + "]",
        )

    rows.sort(
        key=lambda row: (
            row["fold"],
            row["subject"],
            row["sensor_family"],
        )
    )

    payload = {
        "schema_version":
            "phase6k_fleet_metadata_workload_audit_v1",
        "execution_authorized": False,
        "phase6e_historical_digest_serializer_recovered": False,
        "phase6e_frozen_digests_replaced": False,
        "metadata_only": True,
        "shard_count": len(rows),
        "model_independent_pair_count": totals["pair_count"],
        "rows": rows,
    }

    return {
        "status": "PHASE6K_NUMERIC_WORKLOAD_AUDIT_PASS",
        "execution_authorized": False,
        "original_phase6e_digests_reproduced": False,
        "frozen_numeric_workload_reproduced": True,
        "subject_count": len(results),
        "shard_count": len(rows),
        "source_original_groups": source_groups,
        "source_structural_omissions": source_omissions,
        "totals": dict(totals),
        "phase6k_independent_inventory_sha256":
            hashlib.sha256(canonical_line(payload)).hexdigest(),
        "inventory": payload,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", type=int)
    group.add_argument("--full", action="store_true")

    parser.add_argument(
        "--expected-subject-sha256",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    context = FrozenContext()

    if args.subject is not None:
        result = SubjectAudit(context, args.subject).run()

        if args.expected_subject_sha256 is not None:
            if (
                result["subject_inventory_sha256"]
                != args.expected_subject_sha256
            ):
                raise RuntimeError(
                    "independent subject digest does not reproduce"
                )

        print(
            "PHASE6K_SUBJECT_AUDIT="
            + json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        print(
            "PHASE6K_SUBJECT_METADATA_AUDIT=PASS"
        )
        return

    results = []

    for subject in sorted(context.subject_folds):
        result = SubjectAudit(context, subject).run()
        results.append(result)

        print(
            "PHASE6K_FLEET_PROGRESS="
            + json.dumps(
                {
                    "completed_subjects": len(results),
                    "total_subjects": 61,
                    "subject": subject,
                    "subject_pairs":
                        result["totals"]["pair_count"],
                    "subject_digest":
                        result["subject_inventory_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    report = verify_full_global(context, results)

    print(
        "PHASE6K_FLEET_REPORT="
        + json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    print(
        "PHASE6K_FULL_NUMERIC_WORKLOAD_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
