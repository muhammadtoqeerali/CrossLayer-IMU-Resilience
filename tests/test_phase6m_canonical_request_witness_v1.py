"""Phase6M metadata-only request witness safety tests."""

import pytest

import csc_phase6m_canonical_request_witness_v1 as witness


def test_execution_remains_disabled():
    assert witness.EXECUTION_AUTHORIZED is False
    assert witness.MODEL_FORWARD_AUTHORIZED is False


def test_coverage_tag_generation():
    tags = witness._sample_tags(
        family="drift",
        kind="source_trial",
        persistence="persistent_from_onset_until_trial_end",
        stratum=5,
        overlap=0,
        eligible_variants=("fp32", "ptq_v7"),
    )

    assert ("stratum", 5) in tags
    assert ("family", "drift") in tags
    assert ("zero_overlap", "source_trial") in tags
    assert ("eligible_variant", "fp32") in tags
    assert ("eligible_variant", "ptq_v7") in tags


def test_nonzero_overlap_omits_zero_overlap_tag():
    tags = witness._sample_tags(
        family="bias",
        kind="stored_window",
        persistence="transient_one_inference",
        stratum=1,
        overlap=1,
        eligible_variants=("fp32",),
    )

    assert ("zero_overlap", "stored_window") not in tags


def test_unknown_subject_cannot_change_witness_scope():
    with pytest.raises(ValueError):
        witness.qualify_subject(subject=999)


def test_even_fake_authorization_does_not_execute():
    with pytest.raises(
        RuntimeError,
        match=witness.EXECUTION_BLOCK,
    ):
        witness.execute_shard(
            shard_id_value="synthetic",
            gate_path="fake-gate",
            dataset_root="/unused",
        )


def test_no_real_execution_call_in_entrypoint():
    import inspect

    source = inspect.getsource(witness.execute_shard)

    assert "raise RuntimeError(EXECUTION_BLOCK)" in source
    assert "load_model_bundle" not in source
    assert "execute_fault_sequence" not in source
