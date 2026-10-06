# Phase 5B — Paired Compute-FI Execution Harness Qualification v1

**Status:** QUALIFIED_PAIRED_SYNTHETIC_EXECUTION_HARNESS  
**Evidence tier:** P0  
**Date:** 2026-10-05

## Purpose

This stage qualifies the execution layer that pairs a clean inference with a
faulted inference while preserving the frozen Phase-5A fault identity and
payload semantics.

Qualification uses synthetic 30x9 model inputs only.

No training, calibration, validation, outer-test or OnField sample is used.

## FP32 execution

The frozen FP32 checkpoint is reconstructed directly from checkpoint
hyperparameters and loaded strictly.

FP32 activation and buffer faults are injected through temporary forward hooks
that return a mutated tensor copy. The source model parameters are not changed.

## PTQ execution

The qualified PTQ v7 comparator is reconstructed as an eager FX graph and the
frozen `v7_state_dict.pt` is strictly loaded.

The eager model is required to match the frozen TorchScript reference
bit-for-bit on clean synthetic inputs.

PTQ activation/buffer faults are injected at explicit FX graph nodes.

The exact frozen representations are preserved:

- persistent quantized weight: `torch.qint8` / `torch.int8`;
- quantized activation/buffer: `torch.quint8` / `torch.uint8`;
- floating activation/buffer: `torch.float32`.

## Qualified frozen fault families

The harness executes all five frozen Phase-5A families:

1. INT8 persistent-weight single-bit flip;
2. quantized-activation single-bit flip;
3. quantized-buffer single-bit flip;
4. FP32-activation single-bit flip;
5. FP32-buffer single-bit flip.

FP32 activation and buffer execution is qualified on both the FP32 model and
the mixed-precision PTQ path.

## Pairing guarantees

Each pair uses:

- the same synthetic input;
- the same checkpoint/model estate member;
- the same model preprocessing;
- one frozen fault identity.

Every active mutation records:

- representation class;
- target;
- element index;
- bit position;
- before payload;
- after payload;
- changed element indices;
- tensor dtype;
- integer payload dtype where applicable;
- quantization metadata preservation.

## Replay and temporal semantics

The same input and fault identity reproduce the same mutation record and the
same faulted output.

Transient faults occur only at their selected inference index.

Persistent weight faults apply from their onset through trial end.

A new trial/session resets to clean state.

## Source immutability

Checkpoint, PTQ state and TorchScript artifacts are hash-checked before and
after qualification and must remain unchanged.

## Scientific boundary

This is synthetic P0 software FI qualification.

It is not a CC task-performance result.

No real dataset is faulted at this stage.

No outer-test or OnField evidence is accessed.

No MCU, SEU-rate, register, cache, firmware, HIL or physical equivalence is
claimed.

The immutable outer compute-FI sampling plan remains unfrozen.

The next governed step is development-safe qualification on training/calibration
windows before freezing the outer CC sampling plan.
