from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "MASTER_PROJECT_CONTEXT.md",
    "AGENTS.md",
    "configs/project.yaml",
    "configs/datasets/local_paths.example.yaml",
    "docs/RESEARCH_PLAN.md",
    "docs/EXPERIMENT_CONTRACT.md",
    "docs/BASELINE_SELECTION_PROTOCOL.md",
    "docs/CLAIM_LEDGER.md",
    "docs/NOVELTY_BOUNDARY.md",
    "docs/FAULT_TAXONOMY_CANDIDATE.md",
    "docs/PHASE_LOG.md",
    "src/crosslayer_resilience/__init__.py",
    "src/crosslayer_resilience/contracts.py",
)

FORBIDDEN_TRACKED_PREFIXES = (
    "data/",
    "datasets/",
    "results/",
    "outputs/",
    "runs/",
    "logs/",
    "checkpoints/",
    "artifacts/",
    "exports/",
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def main() -> int:
    print("CrossLayer-IMU-Resilience project doctor")
    print(f"root={ROOT}")
    print()

    if not (ROOT / ".git").is_dir():
        fail(".git directory missing")

    ok("Git repository detected")

    missing = [
        path
        for path in REQUIRED_FILES
        if not (ROOT / path).is_file()
    ]

    if missing:
        fail(
            "required files missing: "
            + ", ".join(missing)
        )

    ok(
        f"{len(REQUIRED_FILES)} required project files present"
    )

    master = (
        ROOT / "MASTER_PROJECT_CONTEXT.md"
    ).read_text(encoding="utf-8")

    required_terms = (
        "SENSOR",
        "COMPUTE",
        "COMBINED",
        "final held-out",
        "Phase 6",
        "silent",
    )

    missing_terms = [
        term
        for term in required_terms
        if term not in master
    ]

    if missing_terms:
        fail(
            "master context missing required terms: "
            + ", ".join(missing_terms)
        )

    ok("Master context contains cross-layer safeguards")

    agents = (
        ROOT / "AGENTS.md"
    ).read_text(encoding="utf-8")

    agent_markers = (
        "Do not use final test data",
        "Split parent data before stochastic fault generation",
        "Do not claim physical hardware evidence from software injection",
    )

    missing_rules = [
        rule
        for rule in agent_markers
        if rule not in agents
    ]

    if missing_rules:
        fail(
            "AGENTS.md missing safeguards: "
            + " | ".join(missing_rules)
        )

    ok("Agent safeguards present")

    config = (
        ROOT / "configs/project.yaml"
    ).read_text(encoding="utf-8")

    config_markers = (
        "name: CrossLayer-IMU-Resilience",
        "split_before_fault_injection: true",
        "test_data_for_tuning: false",
        "reliability_aware_model_as_primary_baseline: false",
        "target_status: candidate_not_frozen",
    )

    missing_config = [
        marker
        for marker in config_markers
        if marker not in config
    ]

    if missing_config:
        fail(
            "project config missing markers: "
            + ", ".join(missing_config)
        )

    ok("Project configuration remains conservative")

    violations = [
        path
        for path in tracked_files()
        if any(
            path.startswith(prefix)
            for prefix in FORBIDDEN_TRACKED_PREFIXES
        )
    ]

    if violations:
        fail(
            "forbidden generated/data paths tracked: "
            + ", ".join(violations)
        )

    ok("No forbidden generated/data paths are tracked")

    print()
    print("PROJECT_DOCTOR_STATUS=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
