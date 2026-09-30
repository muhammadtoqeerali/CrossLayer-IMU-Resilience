from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path.home() / "toqeer"

DIRTY = ROOT / "IMU_Reliability"
CLEAN = ROOT / "RC-RGD-IMU_publish"

OUT_JSON = (
    PROJECT
    / "manifests"
    / "phase_1b_baseline_provenance_audit_v1.json"
)

OUT_MD = (
    PROJECT
    / "docs"
    / "PHASE_1B_BASELINE_PROVENANCE.md"
)


TARGET_PATHS = [
    # Task-specific reconstructed baseline
    "src/imu_reliability/baseline/date2025_cnn400.py",
    "src/imu_reliability/baseline/historical_decision.py",
    "docs/protected_baseline_v1.md",

    # Baseline reconstruction / verification
    "experiments/00_baseline/finalize_historical_decision.py",
    "experiments/00_baseline/fingerprint_400ms_datasets.py",
    "experiments/00_baseline/lock_and_evaluate_baseline.py",
    "experiments/00_baseline/verify_400ms_candidate.py",

    # Generic clean baseline candidates
    "models/architectures/baseline_cnn.py",
    "models/architectures/ds_cnn.py",
    "models/architectures/tcn.py",

    # Generic baseline training
    "experiments/03_baselines/train_1dcnn_baseline.py",

    # Existing project configuration
    "configs/project.yaml",
]


KNOWN_DATE_CHECKPOINT_SHA256 = (
    "ee7c0079bfb8555bff45c3077cc24eaa"
    "4373c57729045d92a831a1d7a3ea9bb1"
)


MODEL_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".tflite",
    ".h5",
    ".pkl",
    ".pickle",
    ".bin",
}


SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_output(
    root: Path,
    *args: str,
) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )

        if proc.returncode != 0:
            return None

        return proc.stdout.strip()
    except Exception:
        return None


def git_blob_at_head(
    root: Path,
    relative: str,
) -> bytes | None:
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            capture_output=True,
            timeout=30,
        )

        if proc.returncode != 0:
            return None

        return proc.stdout
    except Exception:
        return None


def git_status_for_path(
    root: Path,
    relative: str,
) -> str:
    result = git_output(
        root,
        "status",
        "--porcelain",
        "--",
        relative,
    )
    return result or ""


def safe_text(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def analyze_python(text: str) -> dict[str, Any]:
    result = {
        "classes": [],
        "functions": [],
        "imports": [],
        "syntax_ok": False,
    }

    try:
        tree = ast.parse(text)
    except Exception:
        return result

    result["syntax_ok"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            result["functions"].append(node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result["imports"].append(module)

    result["classes"] = sorted(
        set(result["classes"])
    )

    result["functions"] = sorted(
        set(result["functions"])
    )

    result["imports"] = sorted(
        set(result["imports"])
    )

    return result


def extract_relevant_strings(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except Exception:
        return []

    values = []

    markers = (
        "data",
        "result",
        "model",
        "checkpoint",
        ".pt",
        ".pth",
        ".onnx",
        ".tflite",
        "univr",
        "fall",
        "400",
    )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        ):
            value = node.value.strip()

            if not value:
                continue

            lower = value.lower()

            if any(
                marker in lower
                for marker in markers
            ):
                if len(value) <= 300:
                    values.append(value)

    return sorted(set(values))


def compare_target(relative: str) -> dict[str, Any]:
    dirty_path = DIRTY / relative
    clean_path = CLEAN / relative

    record: dict[str, Any] = {
        "path": relative,
        "dirty_exists": dirty_path.is_file(),
        "clean_exists": clean_path.is_file(),
    }

    if dirty_path.is_file():
        dirty_bytes = dirty_path.read_bytes()

        record["dirty_worktree"] = {
            "sha256": sha256_bytes(dirty_bytes),
            "size_bytes": len(dirty_bytes),
            "git_status": git_status_for_path(
                DIRTY,
                relative,
            ),
        }

        if dirty_path.suffix == ".py":
            text = dirty_bytes.decode(
                "utf-8",
                errors="ignore",
            )

            record["dirty_worktree"][
                "python"
            ] = analyze_python(text)

            record["dirty_worktree"][
                "relevant_strings"
            ] = extract_relevant_strings(text)

    dirty_head = git_blob_at_head(
        DIRTY,
        relative,
    )

    if dirty_head is not None:
        record["dirty_head"] = {
            "sha256": sha256_bytes(dirty_head),
            "size_bytes": len(dirty_head),
        }

    if clean_path.is_file():
        clean_bytes = clean_path.read_bytes()

        record["clean_worktree"] = {
            "sha256": sha256_bytes(clean_bytes),
            "size_bytes": len(clean_bytes),
            "git_status": git_status_for_path(
                CLEAN,
                relative,
            ),
        }

        if clean_path.suffix == ".py":
            text = clean_bytes.decode(
                "utf-8",
                errors="ignore",
            )

            record["clean_worktree"][
                "python"
            ] = analyze_python(text)

            record["clean_worktree"][
                "relevant_strings"
            ] = extract_relevant_strings(text)

    clean_head = git_blob_at_head(
        CLEAN,
        relative,
    )

    if clean_head is not None:
        record["clean_head"] = {
            "sha256": sha256_bytes(clean_head),
            "size_bytes": len(clean_head),
        }

    dirty_sha = (
        record.get(
            "dirty_worktree",
            {},
        ).get("sha256")
    )

    dirty_head_sha = (
        record.get(
            "dirty_head",
            {},
        ).get("sha256")
    )

    clean_sha = (
        record.get(
            "clean_worktree",
            {},
        ).get("sha256")
    )

    if (
        dirty_sha is not None
        and clean_sha is not None
    ):
        record["dirty_equals_clean"] = (
            dirty_sha == clean_sha
        )

    if (
        dirty_sha is not None
        and dirty_head_sha is not None
    ):
        record["dirty_equals_own_head"] = (
            dirty_sha == dirty_head_sha
        )

    return record


def scan_model_artifacts(
    root: Path,
) -> list[dict[str, Any]]:
    artifacts = []

    for current, dirs, files in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS
        ]

        for filename in files:
            path = Path(current) / filename

            if path.suffix.lower() not in MODEL_SUFFIXES:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            # Prevent hashing unexpectedly huge archived models.
            if size > 250 * 1024 * 1024:
                continue

            try:
                digest = sha256_file(path)
            except Exception:
                continue

            artifacts.append(
                {
                    "path": str(
                        path.relative_to(root)
                    ),
                    "size_bytes": size,
                    "sha256": digest,
                    "matches_known_date_checkpoint": (
                        digest
                        == KNOWN_DATE_CHECKPOINT_SHA256
                    ),
                }
            )

    artifacts.sort(
        key=lambda x: (
            not x[
                "matches_known_date_checkpoint"
            ],
            x["path"],
        )
    )

    return artifacts


def search_first_party_files(
    root: Path,
    patterns: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = []

    for current, dirs, files in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS
            and d != "external"
            and d != "external_references"
        ]

        for filename in files:
            path = Path(current) / filename

            try:
                relative = str(
                    path.relative_to(root)
                ).replace("\\", "/")
            except Exception:
                continue

            lower = relative.lower()

            if not any(
                re.search(pattern, lower)
                for pattern in patterns
            ):
                continue

            if path.suffix.lower() not in {
                ".py",
                ".c",
                ".h",
                ".cpp",
                ".cc",
                ".hpp",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                ".txt",
                ".sh",
            }:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            if size > 2_000_000:
                continue

            records.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": sha256_file(path),
                }
            )

    records.sort(
        key=lambda x: x["path"]
    )

    return records


def repository_info(
    root: Path,
) -> dict[str, Any]:
    return {
        "path": str(root),
        "branch": git_output(
            root,
            "branch",
            "--show-current",
        ),
        "head": git_output(
            root,
            "rev-parse",
            "HEAD",
        ),
        "origin": git_output(
            root,
            "remote",
            "get-url",
            "origin",
        ),
        "dirty": bool(
            git_output(
                root,
                "status",
                "--porcelain",
            )
        ),
    }


def main() -> int:
    print(
        "CrossLayer Phase-1B baseline provenance audit"
    )
    print()

    comparisons = []

    for relative in TARGET_PATHS:
        record = compare_target(relative)
        comparisons.append(record)

        state = []

        if record.get("dirty_exists"):
            state.append("dirty-tree:yes")
        else:
            state.append("dirty-tree:no")

        if record.get("clean_exists"):
            state.append("clean-tree:yes")
        else:
            state.append("clean-tree:no")

        if "dirty_equals_clean" in record:
            state.append(
                "same"
                if record["dirty_equals_clean"]
                else "DIFFERENT"
            )

        print(
            f"[TARGET] {relative}"
        )
        print(
            "         "
            + " | ".join(state)
        )

    print()
    print(
        "Scanning local model/checkpoint artifacts..."
    )

    dirty_artifacts = scan_model_artifacts(
        DIRTY
    )

    clean_artifacts = scan_model_artifacts(
        CLEAN
    )

    matched = [
        {
            "source": "IMU_Reliability",
            **item,
        }
        for item in dirty_artifacts
        if item[
            "matches_known_date_checkpoint"
        ]
    ] + [
        {
            "source": "RC-RGD-IMU_publish",
            **item,
        }
        for item in clean_artifacts
        if item[
            "matches_known_date_checkpoint"
        ]
    ]

    print(
        "Known DATE checkpoint matches:",
        len(matched),
    )

    for item in matched:
        print(
            "  MATCH",
            item["source"],
            item["path"],
            item["sha256"],
        )

    print()
    print(
        "Scanning first-party quantization/export infrastructure..."
    )

    quant_dirty = search_first_party_files(
        DIRTY,
        (
            r"quant",
            r"onnx",
            r"tflite",
            r"export",
            r"int8",
        ),
    )

    quant_clean = search_first_party_files(
        CLEAN,
        (
            r"quant",
            r"onnx",
            r"tflite",
            r"export",
            r"int8",
        ),
    )

    print(
        "Quant/export candidates:",
        f"dirty={len(quant_dirty)}",
        f"clean={len(quant_clean)}",
    )

    print()
    print(
        "Scanning first-party embedded/runtime infrastructure..."
    )

    embedded_dirty = search_first_party_files(
        DIRTY,
        (
            r"embedded",
            r"runtime",
            r"stm32",
            r"cmsis",
            r"firmware",
        ),
    )

    embedded_clean = search_first_party_files(
        CLEAN,
        (
            r"embedded",
            r"runtime",
            r"stm32",
            r"cmsis",
            r"firmware",
        ),
    )

    print(
        "Embedded/runtime candidates:",
        f"dirty={len(embedded_dirty)}",
        f"clean={len(embedded_clean)}",
    )

    manifest = {
        "schema": (
            "crosslayer_phase1b_"
            "baseline_provenance_audit_v1"
        ),
        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "audit_mode": "read_only",
        "crosslayer_commit": git_output(
            PROJECT,
            "rev-parse",
            "HEAD",
        ),
        "sources": {
            "development": repository_info(
                DIRTY
            ),
            "published_clean": repository_info(
                CLEAN
            ),
        },
        "known_date_checkpoint_sha256": (
            KNOWN_DATE_CHECKPOINT_SHA256
        ),
        "target_comparisons": comparisons,
        "model_artifact_inventory": {
            "development": dirty_artifacts,
            "published_clean": clean_artifacts,
            "known_checkpoint_matches": matched,
        },
        "quantization_export_candidates": {
            "development": quant_dirty,
            "published_clean": quant_clean,
        },
        "embedded_runtime_candidates": {
            "development": embedded_dirty,
            "published_clean": embedded_clean,
        },
        "working_baseline_decision": {
            "primary_candidate": (
                "DATE2025_CNN_400MS_RECONSTRUCTED"
            ),
            "status": (
                "candidate_pending_phase1_verification"
            ),
            "comparison_candidates": [
                "generic_compact_1d_cnn",
                "ds_cnn",
                "compact_tcn",
            ],
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------------

    lines: list[str] = []

    lines.append(
        "# Phase 1B Baseline Provenance Audit"
    )
    lines.append("")
    lines.append(
        "Status: evidence collected. "
        "Baseline remains unfrozen."
    )
    lines.append("")

    lines.append(
        "## Source repositories"
    )
    lines.append("")

    for name, info in manifest[
        "sources"
    ].items():
        lines.append(
            f"### {name}"
        )
        lines.append("")
        lines.append(
            f"- Path: `{info['path']}`"
        )
        lines.append(
            f"- Branch: `{info['branch']}`"
        )
        lines.append(
            f"- HEAD: `{info['head']}`"
        )
        lines.append(
            f"- Origin: `{info['origin']}`"
        )
        lines.append(
            f"- Dirty: `{info['dirty']}`"
        )
        lines.append("")

    lines.append(
        "## Exact candidate-file comparison"
    )
    lines.append("")

    lines.append(
        "| File | Development exists | "
        "Clean exists | Dev = clean | "
        "Dev modified vs own HEAD |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for item in comparisons:
        dirty_vs_head = None

        if (
            "dirty_equals_own_head"
            in item
        ):
            dirty_vs_head = not item[
                "dirty_equals_own_head"
            ]

        lines.append(
            f"| `{item['path']}` | "
            f"{item.get('dirty_exists', False)} | "
            f"{item.get('clean_exists', False)} | "
            f"{item.get('dirty_equals_clean', '-')} | "
            f"{dirty_vs_head if dirty_vs_head is not None else '-'} |"
        )

    lines.append("")
    lines.append(
        "## DATE-2025 checkpoint verification"
    )
    lines.append("")
    lines.append(
        f"Expected SHA-256: "
        f"`{KNOWN_DATE_CHECKPOINT_SHA256}`"
    )
    lines.append("")

    if matched:
        lines.append(
            "Matching local artifact(s) were found:"
        )
        lines.append("")

        for item in matched:
            lines.append(
                f"- `{item['source']}/{item['path']}`"
            )
    else:
        lines.append(
            "No local artifact with the expected "
            "checkpoint hash was found by this scan."
        )

    lines.append("")
    lines.append(
        "Checkpoint absence would not invalidate the "
        "architecture as a research candidate, but it "
        "would change the reproducibility path and must "
        "be handled explicitly."
    )

    lines.append("")
    lines.append(
        "## Quantization/export candidates"
    )
    lines.append("")

    for label, records in (
        (
            "Development tree",
            quant_dirty,
        ),
        (
            "Clean published tree",
            quant_clean,
        ),
    ):
        lines.append(
            f"### {label}"
        )
        lines.append("")

        for item in records[:80]:
            lines.append(
                f"- `{item['path']}` "
                f"`{item['sha256'][:12]}`"
            )

        if not records:
            lines.append(
                "- None detected"
            )

        lines.append("")

    lines.append(
        "## Embedded/runtime candidates"
    )
    lines.append("")

    for label, records in (
        (
            "Development tree",
            embedded_dirty,
        ),
        (
            "Clean published tree",
            embedded_clean,
        ),
    ):
        lines.append(
            f"### {label}"
        )
        lines.append("")

        for item in records[:100]:
            lines.append(
                f"- `{item['path']}` "
                f"`{item['sha256'][:12]}`"
            )

        if not records:
            lines.append(
                "- None detected"
            )

        lines.append("")

    lines.append(
        "## Current baseline decision"
    )
    lines.append("")
    lines.append(
        "`DATE2025_CNN_400MS_RECONSTRUCTED` is the "
        "current primary candidate because it directly "
        "matches the pre-impact fall-detection task."
    )
    lines.append("")
    lines.append(
        "It is **not frozen**."
    )
    lines.append("")
    lines.append(
        "The generic compact 1D CNN, DS-CNN and TCN "
        "remain comparison/generalization candidates."
    )

    OUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"TARGET_FILES={len(comparisons)}"
    )
    print(
        f"DIRTY_MODEL_ARTIFACTS="
        f"{len(dirty_artifacts)}"
    )
    print(
        f"CLEAN_MODEL_ARTIFACTS="
        f"{len(clean_artifacts)}"
    )
    print(
        f"KNOWN_CHECKPOINT_MATCHES="
        f"{len(matched)}"
    )
    print(
        f"DIRTY_QUANT_EXPORT="
        f"{len(quant_dirty)}"
    )
    print(
        f"CLEAN_QUANT_EXPORT="
        f"{len(quant_clean)}"
    )
    print(
        f"DIRTY_EMBEDDED_RUNTIME="
        f"{len(embedded_dirty)}"
    )
    print(
        f"CLEAN_EMBEDDED_RUNTIME="
        f"{len(embedded_clean)}"
    )

    print()
    print(
        "PRIMARY_BASELINE_CANDIDATE="
        "DATE2025_CNN_400MS_RECONSTRUCTED"
    )
    print(
        "BASELINE_FREEZE_STATUS=PENDING"
    )
    print(
        "PHASE_1B_PROVENANCE_AUDIT=PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
