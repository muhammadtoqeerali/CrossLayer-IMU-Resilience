import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CFG = (
    ROOT
    / "configs/datasets/"
    "phase4i_onfield_external_300ms_root_binding_v1.json"
)

MANIFEST = (
    ROOT
    / "manifests/"
    "phase_4i_onfield_external_300ms_root_binding_v1.json"
)


def load(path):
    return json.loads(
        path.read_text()
    )


def test_canonical_root_and_retained_cohort():
    x = load(CFG)

    assert x[
        "canonical_processed_root"
    ] == (
        "/mnt/hdd16T/protechto/data/OnField/"
        "segments/300ms_50ov_npseg_filt_binary"
    )

    assert x[
        "retained_storage_ids"
    ] == [
        str(i)
        for i in range(
            1001,
            1011,
        )
    ]

    assert x[
        "retained_subject_count"
    ] == 10

    assert x[
        "retained_trial_count"
    ] == 16

    assert x[
        "retained_window_count"
    ] == 1023337

    assert x[
        "retained_label_counts"
    ] == {
        "Activity": 1023337,
        "Falling": 0,
    }


def test_rejected_cases_remain_excluded():
    x = load(CFG)

    assert x[
        "rejected_storage_ids"
    ] == [
        "999",
        "1000",
    ]


def test_namespace_aliases_are_byte_identical_but_unused():
    x = load(CFG)
    m = load(MANIFEST)

    aliases = x[
        "workstation_namespace_alias_evidence"
    ]

    assert aliases[
        "1109"
    ][
        "canonical_storage_id"
    ] == "1009"

    assert aliases[
        "1110"
    ][
        "canonical_storage_id"
    ] == "1010"

    assert aliases[
        "1109"
    ][
        "byte_identical"
    ] is True

    assert aliases[
        "1110"
    ][
        "byte_identical"
    ] is True

    assert m[
        "namespace_resolution"
    ][
        "evaluation_will_use_alias_mapping"
    ] is False

    assert m[
        "namespace_resolution"
    ][
        "workstation_tree_renamed_or_modified"
    ] is False


def test_no_onfield_selection_or_tuning():
    x = load(CFG)

    assert all(
        value is False
        for value in x[
            "leakage_policy"
        ].values()
    )


def test_activity_only_claim_boundary():
    x = load(CFG)

    assert (
        "activity_specificity"
        in x["allowed_metrics"]
    )

    assert (
        "fall_recall"
        in x["prohibited_claims"]
    )

    assert (
        "fall_detection_lead_time"
        in x["prohibited_claims"]
    )
