from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
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

import cc_outcome_analyzer_v2 as v2  # noqa: E402
import cc_outcome_analyzer_v3 as v3  # noqa: E402
import cc_outcome_io_runner_v3 as io  # noqa: E402
import cc_outcome_executor_v3 as executor  # noqa: E402


RESULT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/results/phase5af_r4_canonical_persistence_execution_chain_qualification_v1/qualification.json"
)

CONFIG = (
    ROOT
    / "configs/evaluation/"
    "phase5af_r4_canonical_persistence_execution_chain_qualification_v1.json"
)

CANONICAL = (
    "persistent_from_onset_until_trial_end"
)

NONCANONICAL = (
    "persistent_from_onset_to_trial_end"
)


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def persistent_rows(
    outer_id="x",
):
    return [
        {
            "outer_instance_id":
                outer_id,

            "execution_window_index":
                1,

            "onset_or_inference_index":
                1,

            "faulted_softmax_values":
                [0.2, 0.8],
        },
        {
            "outer_instance_id":
                outer_id,

            "execution_window_index":
                2,

            "onset_or_inference_index":
                1,

            "faulted_softmax_values":
                [0.2, 0.8],
        },
    ]


def test_qualification_result():
    x = load(
        RESULT
    )

    assert (
        x["status"]
        == "QUALIFIED_POST_BOUNDARY_CANONICAL_PERSISTENCE_V3_CHAIN"
    )

    assert x[
        "qualification_check_count"
    ] == 10

    assert all(
        x[
            "qualification_checks"
        ].values()
    )


def test_exact_v3_hashes():
    x = load(
        RESULT
    )

    assert (
        x["v3_analyzer_sha256"]
        == sha(v3.__file__)
    )

    assert (
        x["v3_io_runner_sha256"]
        == sha(io.__file__)
    )

    assert (
        x["v3_executor_sha256"]
        == sha(executor.__file__)
    )


def test_canonical_persistent_value_is_accepted():
    output = v3.reconstruct_fault_scenario(
        [0.1, 0.2, 0.3],
        persistent_rows(),
        outer_instance_id="x",
        persistence=CANONICAL,
    )

    assert np.allclose(
        output,
        [0.1, 0.8, 0.8],
    )


def test_noncanonical_alias_is_rejected():
    with pytest.raises(
        v3.AnalysisContractError,
        match="unknown persistence mode",
    ):
        v3.reconstruct_fault_scenario(
            [0.1, 0.2, 0.3],
            persistent_rows(),
            outer_instance_id="x",
            persistence=NONCANONICAL,
        )


def test_v2_reproduces_original_canonical_failure():
    with pytest.raises(
        v2.AnalysisContractError,
        match="unknown persistence mode",
    ):
        v2.reconstruct_fault_scenario(
            [0.1, 0.2, 0.3],
            persistent_rows(),
            outer_instance_id="x",
            persistence=CANONICAL,
        )


def test_transient_v2_v3_equivalence():
    records = [
        {
            "outer_instance_id":
                "t",

            "parent": {
                "window_index":
                    1,
            },

            "faulted_softmax_values":
                [0.25, 0.75],
        },
    ]

    a = v2.reconstruct_fault_scenario(
        [0.1, 0.2],
        records,
        outer_instance_id="t",
        persistence="transient_one_inference",
    )

    b = v3.reconstruct_fault_scenario(
        [0.1, 0.2],
        records,
        outer_instance_id="t",
        persistence="transient_one_inference",
    )

    assert np.array_equal(
        a,
        b,
    )


def test_v3_preserves_nan_token_repair():
    rows = persistent_rows()

    for row in rows:
        row[
            "faulted_softmax_values"
        ] = [
            {
                "nonfinite":
                    "nan",
            },
            {
                "nonfinite":
                    "nan",
            },
        ]

    output = v3.reconstruct_fault_scenario(
        [0.1, 0.2, 0.3],
        rows,
        outer_instance_id="x",
        persistence=CANONICAL,
    )

    assert np.isnan(
        output[1]
    )

    assert np.isnan(
        output[2]
    )


def test_v3_io_is_bound_only_to_v3_analyzer():
    source = Path(
        io.__file__
    ).read_text()

    assert (
        "from cc_outcome_analyzer_v3 import"
        in source
    )

    assert (
        "from cc_outcome_analyzer_v2 import"
        not in source
    )


def test_v3_executor_is_bound_to_v3_chain():
    source = Path(
        executor.__file__
    ).read_text()

    assert (
        "import cc_outcome_analyzer_v3 as analyzer"
        in source
    )

    assert (
        "import cc_outcome_io_runner_v3 as io"
        in source
    )

    assert '"v3_analyzer"' in source
    assert '"v3_io_runner"' in source
    assert '"v3_executor"' in source


def test_r4_does_not_authorize_resume():
    x = load(
        RESULT
    )

    assert x[
        "scientific_protocol_changed"
    ] is False

    assert x[
        "phase5af_resume_authorized_by_r4"
    ] is False

    boundary = x[
        "qualification_boundary"
    ]

    assert boundary[
        "synthetic_only"
    ] is True

    assert boundary[
        "phase5af_production_resumed"
    ] is False

    assert boundary[
        "preserved_completed_shard_deleted"
    ] is False

    assert boundary[
        "preserved_completed_shard_rerun"
    ] is False
