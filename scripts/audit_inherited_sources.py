from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_ROOT = Path.home() / "toqeer"

OUTPUT_JSON = (
    PROJECT_ROOT
    / "manifests"
    / "phase_1_source_audit_v1.json"
)

OUTPUT_MD = (
    PROJECT_ROOT
    / "docs"
    / "PHASE_1_SOURCE_AUDIT.md"
)


# ---------------------------------------------------------------------
# Candidate workstation projects
# ---------------------------------------------------------------------

CANDIDATE_PROJECTS = [
    ("IMU_Reliability", HOME_ROOT / "IMU_Reliability"),
    ("RC-RGD-IMU_publish", HOME_ROOT / "RC-RGD-IMU_publish"),
    ("Protechto-master", HOME_ROOT / "Protechto-master"),
    ("Protechto_master", HOME_ROOT / "Protechto_master"),
    ("Protechto-repo", HOME_ROOT / "Protechto-repo"),
    ("Protechto", HOME_ROOT / "Protechto"),
    ("fall_project_code", HOME_ROOT / "fall_project_code"),
    ("HR_LR_Fallings", HOME_ROOT / "HR_LR_Fallings"),
    ("TRUST_ROBOT", HOME_ROOT / "TRUST_ROBOT"),
    ("TRUST_ROBOT_RUNTIME", HOME_ROOT / "TRUST_ROBOT_RUNTIME"),
]


SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "Debug",
    "Release",
    "results",
    "outputs",
    "runs",
    "logs",
    "checkpoints",
    "artifacts",
    "exports",
    "cache",
    ".cache",
    "wandb",
    "tensorboard",
    "figures",
    "images",
    "plots",
    "analysis_dual_axis_300ms_50ov",
}


SKIP_PATH_FRAGMENTS = {
    "/data/",
    "/datasets/",
    "/raw/",
    "/processed/",
    "/generated/",
    "/saved_models/",
    "/weights/",
}


TEXT_EXTENSIONS = {
    ".py",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".ino",
    ".sh",
    ".bash",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
}


ROOT_FILES_ALWAYS_INCLUDE = {
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "Pipfile",
    "Makefile",
    "CMakeLists.txt",
    ".gitmodules",
}


PATH_CATEGORY_PATTERNS = {
    "baseline_model": [
        r"baseline",
        r"model",
        r"architect",
        r"cnn",
        r"tcn",
        r"network",
    ],
    "training_pipeline": [
        r"train",
        r"trainer",
        r"fit",
        r"loss",
        r"optimizer",
    ],
    "data_pipeline": [
        r"dataset",
        r"dataloader",
        r"loader",
        r"preprocess",
        r"window",
        r"segment",
        r"split",
        r"univr",
        r"imu",
    ],
    "quantization_export": [
        r"quant",
        r"tflite",
        r"onnx",
        r"torchscript",
        r"export",
        r"int8",
        r"convert",
    ],
    "embedded_runtime": [
        r"embedded",
        r"stm32",
        r"cmsis",
        r"microcontroller",
        r"firmware",
        r"runtime",
        r"tflm",
        r"tinyml",
    ],
    "fault_injection": [
        r"fault",
        r"inject",
        r"corrupt",
        r"bitflip",
        r"bit_flip",
        r"noise",
        r"dropout",
        r"stuck",
        r"freeze",
        r"drift",
    ],
    "integrity_runtime": [
        r"integrity",
        r"reliability",
        r"trust",
        r"decision",
        r"ood",
        r"monitor",
        r"supervisor",
        r"recovery",
    ],
    "evaluation": [
        r"eval",
        r"evaluate",
        r"benchmark",
        r"metric",
        r"test",
        r"experiment",
        r"analysis",
    ],
}


CONTENT_KEYWORDS = {
    "pytorch": [
        "import torch",
        "from torch",
        "torch.nn",
    ],
    "tensorflow": [
        "import tensorflow",
        "from tensorflow",
        "keras",
        "tf.lite",
    ],
    "onnx": [
        "onnx",
    ],
    "tflite": [
        "tflite",
        "tf.lite",
    ],
    "stm32": [
        "stm32",
        "STM32",
    ],
    "cmsis": [
        "CMSIS",
        "cmsis",
    ],
    "conv1d": [
        "Conv1d",
        "conv1d",
        "Conv1D",
    ],
    "depthwise": [
        "depthwise",
        "groups=",
        "Depthwise",
    ],
    "lstm": [
        "LSTM",
        "lstm",
    ],
    "transformer": [
        "Transformer",
        "transformer",
        "MultiheadAttention",
    ],
    "quantization": [
        "quantiz",
        "int8",
        "INT8",
        "qconfig",
    ],
}


BASELINE_PATTERNS = [
    (
        "compact_1d_cnn",
        re.compile(
            r"(baseline[_\-]?cnn|lightweight[_\-]?cnn|1d[_\-]?cnn|cnn)",
            re.IGNORECASE,
        ),
    ),
    (
        "ds_cnn",
        re.compile(
            r"(ds[_\-]?cnn|depthwise|separable)",
            re.IGNORECASE,
        ),
    ),
    (
        "compact_tcn",
        re.compile(
            r"(^|[/_.\-])tcn([/_.\-]|$)|temporal[_\-]?conv",
            re.IGNORECASE,
        ),
    ),
]


def run(
    args: list[str],
    cwd: Path,
) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=20,
        )
        output = (proc.stdout or "").strip()
        if not output:
            output = (proc.stderr or "").strip()
        return proc.returncode, output
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def git_value(
    root: Path,
    args: list[str],
) -> str | None:
    code, output = run(
        ["git", *args],
        root,
    )
    if code != 0:
        return None
    return output or None


def git_info(root: Path) -> dict[str, Any]:
    code, _ = run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        root,
    )

    if code != 0:
        return {
            "is_git_repository": False,
        }

    return {
        "is_git_repository": True,
        "root": git_value(
            root,
            ["rev-parse", "--show-toplevel"],
        ),
        "branch": git_value(
            root,
            ["branch", "--show-current"],
        ),
        "head": git_value(
            root,
            ["rev-parse", "HEAD"],
        ),
        "remote_origin": git_value(
            root,
            ["remote", "get-url", "origin"],
        ),
        "status_porcelain": git_value(
            root,
            ["status", "--porcelain"],
        )
        or "",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safely_read_text(
    path: Path,
    max_bytes: int = 1_000_000,
) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def classify_path(
    relative: str,
) -> list[str]:
    lower = relative.lower()
    found: list[str] = []

    for category, patterns in PATH_CATEGORY_PATTERNS.items():
        if any(
            re.search(pattern, lower)
            for pattern in patterns
        ):
            found.append(category)

    return sorted(set(found))


def content_flags(
    text: str,
) -> list[str]:
    flags = []

    for name, needles in CONTENT_KEYWORDS.items():
        if any(
            needle in text
            for needle in needles
        ):
            flags.append(name)

    return sorted(flags)


def is_interesting(
    relative: str,
    filename: str,
) -> bool:
    if filename in ROOT_FILES_ALWAYS_INCLUDE:
        return True

    categories = classify_path(relative)

    return bool(categories)


def family_candidates(
    relative: str,
    text: str,
) -> list[str]:
    search_space = f"{relative}\n{text[:100000]}"
    families = []

    for family, pattern in BASELINE_PATTERNS:
        if pattern.search(search_space):
            families.append(family)

    return sorted(set(families))


def walk_project(
    root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for current, dirs, files in os.walk(root):
        current_path = Path(current)

        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIR_NAMES
            and not d.startswith(".tox")
        ]

        current_string = "/" + str(
            current_path.relative_to(root)
        ).replace("\\", "/") + "/"

        if any(
            fragment in current_string.lower()
            for fragment in SKIP_PATH_FRAGMENTS
        ):
            dirs[:] = []
            continue

        for filename in files:
            path = current_path / filename

            if path.suffix.lower() not in TEXT_EXTENSIONS:
                if filename not in ROOT_FILES_ALWAYS_INCLUDE:
                    continue

            try:
                relative = str(
                    path.relative_to(root)
                ).replace("\\", "/")
            except Exception:
                continue

            if not is_interesting(
                relative,
                filename,
            ):
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            # Avoid generated giant text/code files in the audit.
            if size > 2_000_000:
                continue

            text = safely_read_text(path)

            record = {
                "path": relative,
                "size_bytes": size,
                "sha256": sha256(path),
                "categories": classify_path(relative),
                "content_flags": content_flags(text),
                "baseline_families": family_candidates(
                    relative,
                    text,
                ),
            }

            records.append(record)

    records.sort(
        key=lambda item: item["path"].lower()
    )

    return records


def candidate_score(
    project_name: str,
    record: dict[str, Any],
    family: str,
) -> int:
    path = record["path"].lower()
    score = 0

    if project_name in {
        "IMU_Reliability",
        "RC-RGD-IMU_publish",
    }:
        score += 10

    if "architect" in path:
        score += 6

    if "/models/" in f"/{path}":
        score += 5

    if "baseline" in path:
        score += 5

    if path.endswith(".py"):
        score += 3

    if family == "compact_1d_cnn":
        if "baseline_cnn" in path:
            score += 10
        if "lightweight" in path:
            score += 7

    if family == "ds_cnn":
        if "ds_cnn" in path:
            score += 10
        if "depthwise" in path:
            score += 5

    if family == "compact_tcn":
        if path.endswith("/tcn.py") or path == "tcn.py":
            score += 10
        if "tcn" in Path(path).stem:
            score += 7

    # Penalize proposed/reliability-aware models as protected baseline.
    if any(
        marker in path
        for marker in (
            "reliability",
            "trust",
            "fault_aware",
            "fault-aware",
            "gated",
        )
    ):
        score -= 20

    return score


def main() -> int:
    available_projects: list[dict[str, Any]] = []

    missing_projects: list[dict[str, str]] = []

    hash_locations: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    baseline_candidates: list[dict[str, Any]] = []

    print(
        "CrossLayer-IMU-Resilience Phase-1 inherited source audit"
    )
    print(f"search_root={HOME_ROOT}")
    print()

    for name, root in CANDIDATE_PROJECTS:
        if not root.exists():
            missing_projects.append(
                {
                    "name": name,
                    "path": str(root),
                }
            )
            print(
                f"[MISSING] {name}: {root}"
            )
            continue

        if not root.is_dir():
            missing_projects.append(
                {
                    "name": name,
                    "path": str(root),
                    "reason": "not_directory",
                }
            )
            print(
                f"[SKIP] {name}: not a directory"
            )
            continue

        print(
            f"[SCAN] {name}: {root}"
        )

        repo_info = git_info(root)

        files = walk_project(root)

        category_counts: dict[str, int] = defaultdict(int)

        for record in files:
            for category in record["categories"]:
                category_counts[category] += 1

            hash_locations[
                record["sha256"]
            ].append(
                {
                    "project": name,
                    "path": record["path"],
                }
            )

            for family in record["baseline_families"]:
                baseline_candidates.append(
                    {
                        "project": name,
                        "project_path": str(root),
                        "path": record["path"],
                        "sha256": record["sha256"],
                        "size_bytes": record["size_bytes"],
                        "family": family,
                        "content_flags": record["content_flags"],
                        "score": candidate_score(
                            name,
                            record,
                            family,
                        ),
                    }
                )

        available_projects.append(
            {
                "name": name,
                "path": str(root),
                "git": repo_info,
                "relevant_file_count": len(files),
                "category_counts": dict(
                    sorted(
                        category_counts.items()
                    )
                ),
                "files": files,
            }
        )

        print(
            "       "
            f"relevant_files={len(files)} "
            f"git={repo_info.get('is_git_repository', False)}"
        )

    duplicate_groups = []

    for digest, locations in hash_locations.items():
        project_names = {
            loc["project"]
            for loc in locations
        }

        if len(project_names) < 2:
            continue

        duplicate_groups.append(
            {
                "sha256": digest,
                "locations": locations,
            }
        )

    duplicate_groups.sort(
        key=lambda item: (
            -len(item["locations"]),
            item["sha256"],
        )
    )

    baseline_candidates.sort(
        key=lambda item: (
            item["family"],
            -item["score"],
            item["project"],
            item["path"],
        )
    )

    # Deduplicate same physical/code candidate by family + SHA.
    deduped_candidates = []
    seen_candidate_keys = set()

    for item in baseline_candidates:
        key = (
            item["family"],
            item["sha256"],
        )

        if key in seen_candidate_keys:
            continue

        seen_candidate_keys.add(key)
        deduped_candidates.append(item)

    baseline_candidates = deduped_candidates

    manifest = {
        "schema": "crosslayer_phase1_source_audit_v1",
        "generated_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "crosslayer_project_commit": git_value(
            PROJECT_ROOT,
            ["rev-parse", "HEAD"],
        ),
        "audit_mode": "read_only",
        "source_projects": available_projects,
        "missing_candidate_projects": missing_projects,
        "duplicate_file_groups": duplicate_groups,
        "baseline_candidates": baseline_candidates,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = []

    lines.append(
        "# Phase 1 Source Audit"
    )
    lines.append("")
    lines.append(
        "Status: automated evidence collection complete. "
        "Scientific reuse decisions remain pending."
    )
    lines.append("")
    lines.append(
        "The audit is read-only with respect to inherited projects."
    )
    lines.append("")
    lines.append(
        "## Source projects"
    )
    lines.append("")
    lines.append(
        "| Project | Exists | Git | Branch | HEAD | Relevant files |"
    )
    lines.append(
        "|---|---:|---:|---|---|---:|"
    )

    available_names = {
        item["name"]
        for item in available_projects
    }

    for name, root in CANDIDATE_PROJECTS:
        project = next(
            (
                item
                for item in available_projects
                if item["name"] == name
            ),
            None,
        )

        if project is None:
            lines.append(
                f"| `{name}` | no | - | - | - | - |"
            )
            continue

        info = project["git"]

        head = info.get("head") or "-"
        if head != "-":
            head = head[:12]

        lines.append(
            f"| `{name}` | yes | "
            f"{'yes' if info.get('is_git_repository') else 'no'} | "
            f"`{info.get('branch') or '-'}` | "
            f"`{head}` | "
            f"{project['relevant_file_count']} |"
        )

    lines.append("")
    lines.append(
        "## Relevant-file category counts"
    )
    lines.append("")

    categories = sorted(
        {
            category
            for project in available_projects
            for category in project[
                "category_counts"
            ]
        }
    )

    header = "| Project | " + " | ".join(
        categories
    ) + " |"

    separator = "|---|" + "|".join(
        "---:"
        for _ in categories
    ) + "|"

    lines.append(header)
    lines.append(separator)

    for project in available_projects:
        cells = [
            str(
                project["category_counts"].get(
                    category,
                    0,
                )
            )
            for category in categories
        ]

        lines.append(
            f"| `{project['name']}` | "
            + " | ".join(cells)
            + " |"
        )

    lines.append("")
    lines.append(
        "## Detected clean-baseline candidates"
    )
    lines.append("")
    lines.append(
        "Automated scores are navigation aids only. "
        "They are not scientific model rankings."
    )
    lines.append("")
    lines.append(
        "| Family | Source | File | SHA-256 | Audit score | Signals |"
    )
    lines.append(
        "|---|---|---|---|---:|---|"
    )

    for item in baseline_candidates[:60]:
        flags = ", ".join(
            item["content_flags"]
        ) or "-"

        lines.append(
            f"| `{item['family']}` | "
            f"`{item['project']}` | "
            f"`{item['path']}` | "
            f"`{item['sha256'][:12]}` | "
            f"{item['score']} | "
            f"{flags} |"
        )

    lines.append("")
    lines.append(
        "## Cross-project duplicate source groups"
    )
    lines.append("")
    lines.append(
        f"Detected duplicate groups across distinct candidate projects: "
        f"**{len(duplicate_groups)}**."
    )
    lines.append("")

    for index, group in enumerate(
        duplicate_groups[:30],
        start=1,
    ):
        lines.append(
            f"### Duplicate group {index}"
        )
        lines.append("")
        lines.append(
            f"`{group['sha256']}`"
        )
        lines.append("")

        for loc in group["locations"]:
            lines.append(
                f"- `{loc['project']}/{loc['path']}`"
            )

        lines.append("")

    lines.append(
        "## Phase-1 decision state"
    )
    lines.append("")
    lines.append(
        "No component is yet classified as REUSE, ADAPT, "
        "REFERENCE_ONLY or REJECT."
    )
    lines.append("")
    lines.append(
        "The next step is expert review of the detected candidates "
        "and exact source-file inspection."
    )

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"AVAILABLE_PROJECTS={len(available_projects)}"
    )
    print(
        f"MISSING_PROJECTS={len(missing_projects)}"
    )
    print(
        f"BASELINE_CANDIDATES={len(baseline_candidates)}"
    )
    print(
        f"CROSS_PROJECT_DUPLICATE_GROUPS={len(duplicate_groups)}"
    )

    print()
    print("Top baseline candidates by family:")

    for family in (
        "compact_1d_cnn",
        "ds_cnn",
        "compact_tcn",
    ):
        items = [
            item
            for item in baseline_candidates
            if item["family"] == family
        ]

        print()
        print(f"[{family}]")

        for item in sorted(
            items,
            key=lambda x: (
                -x["score"],
                x["project"],
                x["path"],
            ),
        )[:12]:
            print(
                "  "
                f"score={item['score']:>3} "
                f"{item['project']}/"
                f"{item['path']} "
                f"sha={item['sha256'][:12]}"
            )

    print()
    print(
        f"JSON={OUTPUT_JSON.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"REPORT={OUTPUT_MD.relative_to(PROJECT_ROOT)}"
    )
    print()
    print("PHASE_1A_SOURCE_AUDIT=PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
