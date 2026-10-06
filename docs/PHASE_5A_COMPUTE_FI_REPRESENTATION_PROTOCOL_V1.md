# Phase 5A — Compute-FI Representation Protocol v1

**Status:** FROZEN_PRE_OPERATOR_QUALIFICATION  
**Evidence tier:** P0 software fault injection  
**Date:** 2026-10-05

## Purpose

Phase 5 builds the Compute Fault Engine.

This protocol freezes the representation, identity, pairing, persistence and
deterministic-sampling semantics that must exist before any compute-fault
performance experiment is allowed.

It does **not** contain held-out results and it does not make a hardware claim.

## Frozen initial targets

The governed Phase-5 roadmap begins with:

1. INT8 weights;
2. activations;
3. intermediate buffers.

The qualified Phase-4G v7 deployment comparator is mixed precision rather than
fully INT8. Structural audit established that `conv_2.0.weight` is the only
genuine quantized persistent weight tensor in v7. Therefore the Phase-5A v1
INT8-weight family targets only that tensor.

FP32 parameter faults are not silently pooled with INT8 weight faults. They are
a separate possible future extension.

## Frozen v1 fault families

- `int8_weight_single_bit_flip`
- `quantized_activation_single_bit_flip`
- `quantized_buffer_single_bit_flip`
- `fp32_activation_single_bit_flip`
- `fp32_buffer_single_bit_flip`

Each v1 fault instance changes exactly one element and one bit.

The initial multiplicity is therefore frozen at one. Higher multiplicities are
not inferred from the v1 results and require a successor protocol.

## INT8 semantics

For a quantized payload:

1. select one valid tensor element;
2. obtain its signed 8-bit stored representation;
3. XOR exactly one bit in positions 0–7;
4. reinterpret the resulting two-complement payload;
5. preserve the frozen quantization scale/zero-point metadata.

Quantization metadata corruption is not part of v1.

## FP32 semantics

For an FP32 activation or intermediate buffer:

1. select exactly one float32 element;
2. reinterpret its 32-bit payload as an unsigned integer;
3. XOR exactly one selected bit;
4. reinterpret the result as float32.

Bits are represented exactly as:

- mantissa: 0–22;
- exponent: 23–30;
- sign: 31.

No NaN or infinity created by the bit flip may be silently clipped, repaired,
replaced or discarded. Such behavior must remain observable to the later
compute-FI runner.

## Persistence

Two semantics are frozen:

- `transient_one_inference`: the corruption exists only in the selected model
  invocation;
- `persistent_from_onset_until_trial_end`: the same logical target/bit remains
  corrupted from the selected inference index through the end of that trial.

No state may leak across trials.

## Deterministic replay

A fault instance is identified by a canonical SHA-256 identity containing:

- protocol;
- model variant;
- checkpoint seed;
- fold;
- fault family;
- representation class;
- target;
- target role;
- element index;
- bit position;
- inference index;
- persistence;
- multiplicity;
- replicate index.

Global process RNG state is not a valid identity mechanism.

Where deterministic target selection is required, SHA-256 namespace mapping is
used to map canonical identity material into the valid element range.

## Pairing

Every corrupted inference must be paired with its clean parent under:

- identical input;
- identical checkpoint;
- identical preprocessing;
- identical subject partition;
- identical operating point;
- identical trial order.

The only changed quantity is the frozen compute fault instance.

## Qualification boundary

Before any outer-test compute-FI execution, the implementation must prove:

- bit-exact mutation;
- exact clean restoration for transient faults;
- deterministic replay;
- persistent onset/end semantics;
- no cross-trial state leakage;
- quantized metadata preservation;
- correct handling of FP32 non-finite outcomes.

Qualification may use synthetic tensors and development-safe
training/calibration data only.

Outer-test and OnField data are prohibited for operator design, family
selection, target selection, bit selection, replicate-count selection or
protocol tuning.

## Claim boundary

This is P0 software fault injection.

It is not:

- a physical SEU rate model;
- an MCU SRAM/flash layout model;
- a register or cache fault study;
- firmware injection;
- HIL evidence;
- physical hardware evidence.

No CC or CSC task-performance result exists at this stage.
