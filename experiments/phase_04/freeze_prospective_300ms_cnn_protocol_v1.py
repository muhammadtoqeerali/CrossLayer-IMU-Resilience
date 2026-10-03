from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path.cwd()

FOLD_MANIFEST = ROOT / "manifests/phase_3j_primary_300ms_fivefold_v1.json"

SOURCE_CANDIDATES = [
    Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori"),
    Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path: Path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_source_files(root: Path):
    wanted = {
        "train.py",
        "CNN.py",
        "KFoldDataloader.py",
        "IMUNormalizer.py",
    }

    records = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.name not in wanted:
            continue

        try:
            records.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        except OSError:
            pass

    return sorted(records, key=lambda x: x["path"])


folds = json.loads(FOLD_MANIFEST.read_text())

all_text = json.dumps(folds)

# The manifest must remain the authority for subject membership.
assert FOLD_MANIFEST.exists()
assert "300" in all_text
assert "50" in all_text

source_records = []

for root in SOURCE_CANDIDATES:
    if root.exists():
        source_records.append(
            {
                "root": str(root),
                "git_commit": git_commit(root),
                "files": collect_source_files(root),
            }
        )

protocol = {
    "status": "FROZEN_CANDIDATE",
    "phase": "4B",
    "protocol_name": "prospective_300ms_cnn_fivefold_baseline",
    "dataset_protocol": {
        "window_ms": 300,
        "overlap_percent": 50,
        "sampling_hz": 100,
        "samples_per_window": 30,
        "stride_samples": 15,
        "stride_ms": 150,
        "primary_subject_count": 61,
        "univr_subject_count": 29,
        "kfall_subject_count": 32,
        "onfield_role": "external_validation_only",
        "onfield_retained_ids": [
            "1001",
            "1002",
            "1003",
            "1004",
            "1005",
            "1006",
            "1007",
            "1008",
            "1009",
            "1010",
        ],
        "permanently_excluded_ids": [
            "999",
            "1000",
        ],
    },
    "model": {
        "family": "CNN",
        "checkpoint_reuse": False,
        "source_checkpoint_selection": "none",
    },
    "fold_authority": {
        "manifest": str(FOLD_MANIFEST),
        "sha256": sha256(FOLD_MANIFEST),
        "regenerate_folds": False,
    },
    "selection_boundary": {
        "outer_test_open_during_training": False,
        "outer_test_used_for_checkpoint_selection": False,
        "onfield_open_during_training": False,
        "onfield_used_for_checkpoint_selection": False,
        "historical_checkpoint_accuracy_used_for_selection": False,
    },
    "quantization_boundary": {
        "int8_calibration_before_baseline_freeze": False,
        "calibration_source": "training_only",
    },
    "fault_injection_boundary": {
        "faults_before_baseline_freeze": False,
    },
    "source_records": source_records,
}

out = Path(
    "configs/baseline/"
    "prospective_300ms_cnn_training_candidate_v1.json"
)

out.write_text(
    json.dumps(protocol, indent=2, sort_keys=True) + "\n"
)

print("[PASS] Created prospective training protocol candidate")
print("Protocol:", out)
print("Fold manifest SHA256:", protocol["fold_authority"]["sha256"])

for record in source_records:
    print("Source root:", record["root"])
    print("Git commit:", record["git_commit"])
    print("Source files:", len(record["files"]))
