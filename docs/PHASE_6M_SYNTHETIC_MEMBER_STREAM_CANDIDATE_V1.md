# Phase6M synthetic model-member stream candidate v1

## Frozen evidence

The previously qualified 732-shard / 366-cache Phase6M preflight report
and subject-9 canonical-request witness report are preserved unchanged.

The witness includes 37 representative selected CSC pairs expanded into
189 actual Phase6G pre-forward requests covering all 28 compute strata
and all 12 sensor families. The previously qualified Phase6J synthetic
pair execution produced 5,310 reference rows and 13,989 synthetic
compute-fault rows.

## New qualification

This additive module groups the already-validated request identities
by frozen subject-family shard, model variant and checkpoint seed.

Each group is submitted to the frozen Phase6J
run_model_member_stream helper, using synthetic hooks only.

The synthetic qualification verifies:

- Stream variant/seed isolation
- Exactly one synthetic bundle load per stream
- Exactly one synthetic release per stream
- One synthetic pair hook invocation per witness request
- Rejection of cross-stream variant/seed mismatch
- Cleanup after an injected synthetic pair-hook exception
- Consistent frozen clean-cache and producer ownership
- The same 189 previously validated request IDs

Each submitted job carries a synthetic stream-identity proxy for a
request already qualified in the archived Phase6M witness. This stage
does not reconstruct or execute its full Phase6G payload.

## Explicit limitations

The witness sample is representative, not a full 21,793,038-member
canonical execution stream.

Stream grouping does not yet prove full frozen pair ordering,
sensor conditioning, actual model loading, PTQ weight reset,
fault mutation, real forward inference or output-file committing.

The single synthetic release callback is not evidence of actual
CUDA/torch resource cleanup or PTQ model restoration.

The original historical Phase6E cryptographic stream digests remain
unreproduced. No historical digest is modified or replaced.

Phase6J and the Phase6K/Phase6L freezes remain unchanged.

REAL_CSC_EXECUTION_AUTHORIZED=FALSE
