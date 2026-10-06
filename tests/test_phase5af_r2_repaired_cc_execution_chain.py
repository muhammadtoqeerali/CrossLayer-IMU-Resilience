from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

PHASE05 = (
    ROOT
    / "experiments/phase_05"
)

sys.path.insert(
    0,
    str(
        PHASE05
    ),
)

import cc_outcome_analyzer_v2 as analyzer  # noqa: E402
import cc_outcome_io_runner_v2 as io  # noqa: E402
import cc_outcome_executor_v2 as executor  # noqa: E402


RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5af_r2_repaired_cc_execution_chain_qualification_v1/qualification.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5af_r2_repaired_cc_execution_chain_qualification_v1.json"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def test_qualification_result():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "QUALIFIED_POST_BOUNDARY_REPAIRED_CC_EXECUTION_CHAIN"
    )

    assert x[
        "qualification_check_count"
    ] == 7

    assert len(
        x[
            "qualification_checks"
        ]
    ) == 7

    assert all(
        x[
            "qualification_checks"
        ].values()
    )


def test_exact_named_checks():
    x = load(
        RESULT
    )

    assert set(
        x[
            "qualification_checks"
        ]
    ) == {
        "all_three_frozen_operating_points",
        "dict_nan_persistent_end_to_end",
        "dict_nan_transient_end_to_end",
        "exact_repaired_module_hash_binding",
        "finite_reconstruction_semantics_preserved",
        "nonfinite_fault_accounting",
        "paired_clean_and_faulted_metrics_constructed",
    }


def test_exact_v2_hash_binding():
    x = load(
        RESULT
    )

    assert (
        x[
            "v2_analyzer_sha256"
        ]
        == sha(
            analyzer.__file__
        )
    )

    assert (
        x[
            "v2_io_runner_sha256"
        ]
        == sha(
            io.__file__
        )
    )

    assert (
        x[
            "v2_executor_sha256"
        ]
        == sha(
            executor.__file__
        )
    )


def test_io_runner_is_bound_only_to_v2_analyzer():
    source = Path(
        io.__file__
    ).read_text()

    assert (
        "from cc_outcome_analyzer_v2 import"
        in source
    )

    assert (
        "from cc_outcome_analyzer_v1 import"
        not in source
    )


def test_executor_is_bound_to_v2_chain():
    source = Path(
        executor.__file__
    ).read_text()

    assert (
        "import cc_outcome_analyzer_v2 as analyzer"
        in source
    )

    assert (
        "import cc_outcome_io_runner_v2 as io"
        in source
    )

    assert (
        "repair_binding_path"
        in source
    )


def test_scientific_contract_is_unchanged():
    x = load(
        RESULT
    )

    assert x[
        "scientific_protocol_changed"
    ] is False

    assert x[
        "phase5af_resume_authorized"
    ] is False


def test_qualification_is_synthetic_only():
    x = load(
        RESULT
    )

    boundary = x[
        "qualification_boundary"
    ]

    assert boundary[
        "synthetic_estate_only"
    ] is True

    for key, value in boundary.items():
        if key == "synthetic_estate_only":
            continue

        assert value is False


def test_nan_token_is_supported_end_to_end():
    value = analyzer.faulted_falling_probability({
        "faulted_softmax_values": [
            {
                "nonfinite":
                    "nan",
            },
            {
                "nonfinite":
                    "nan",
            },
        ],
    })

    assert value != value


def test_repair_binding_rejects_wrong_v2_executor_hash(tmp_path):
    binding = load(
        CONFIG
    )

    binding[
        "implementation_lineage"
    ][
        "v2_executor"
    ][
        "sha256"
    ] = "0" * 64

    bad = (
        tmp_path
        / "bad.json"
    )

    bad.write_text(
        json.dumps(
            binding
        )
    )

    with pytest.raises(
        executor.OutcomeExecutorError,
        match="V2 executor hash mismatch",
    ):
        executor.validate_lineage(
            protocol_path=binding[
                "phase5z_protocol"
            ][
                "path"
            ],
            gate_path=binding[
                "base_phase5ac_gate"
            ][
                "path"
            ],
            plan_path=binding[
                "phase5z_protocol"
            ][
                "path"
            ],
            outer_root="/synthetic/not/accepted",
            dataset_root="/synthetic/not/dataset",
            qualification_mode=True,
            repair_binding_path=bad,
        )
