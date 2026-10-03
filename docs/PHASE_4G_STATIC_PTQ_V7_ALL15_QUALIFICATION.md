# Phase 4G — v7 All-15 Training-Calibration Qualification

**Status:** QUALIFIED_TRAINING_CALIBRATION
**Date:** 2026-10-02
**Candidate:** `torch_fx_x86_fc1_fp32_v7`

## Result

The frozen v7 candidate passed the complete predeclared
training-calibration qualification:

- 15 / 15 overall passes;
- 15 / 15 structural passes;
- 15 / 15 numerical-fidelity passes;
- 15 / 15 serialization passes.

The qualification covers seeds 42, 123, and 2025 across folds 1–5.

Each checkpoint was calibrated and evaluated only on the corresponding
fold's frozen 4096 training-partition calibration identities.

## Numerical-fidelity summary

The frozen complete-population p99 absolute probability-error gate is
0.05.

Across the 15 frozen checkpoints:

- minimum p99: `0.017051076516509056`;
- mean p99: `0.026808109134435654`;
- maximum p99: `0.042779739946126938`;
- frozen limit: `0.050000000000000003`.

The worst member was seed
`42`, fold
`1`, with p99
`0.042779739946126938`.

The best member was seed
`123`, fold
`2`, with p99
`0.017051076516509056`.

## Precision contract

v7 remains **mixed-precision static PTQ**.

Under the frozen v7 contract:

- `conv_2.0` is the only quantized weighted operator;
- `fc.1` is FP32;
- there are no quantized Linear modules;
- the FP32 front end is preserved;
- `fc.4` remains FP32;
- the artifact must not be described as fully INT8.

## Per-class diagnostics

Activity and Falling class diagnostics were recorded for descriptive
inspection only.

They were not separate acceptance gates.

In particular, Falling-class p99 values may exceed 0.05 without
constituting protocol failure because the prospectively frozen p99
gate applies to the complete calibration population, not a per-class
subset.

## Scientific boundary

This qualification used no validation data, no outer-test data, no
OnField data, and no fault-injected data.

No quantizer, numerical gate, checkpoint, decision threshold, or
precision rule was changed after observing the all-15 result.

The pre-result v7 protocol remains unchanged and is preserved
separately.

This evidence supports training-calibration numerical fidelity and
serialization only. It does not establish fully INT8 execution,
MCU execution, latency, energy, held-out predictive performance, or
fault robustness.

## Evidence hashes

- pre-result protocol SHA256:
  `fce8535e9666db57eaf08b3fae9c0d1d56d60dfc9a91792a1f9647778aa148eb`
- v7 implementation SHA256:
  `38e34f3fa179c5b144d6db30494b2c1ace444156451685ddaa92268d635f14eb`
- all-15 execution manifest SHA256:
  `811481415964bb2dc964116600a0ab43af407b4cddef76422a0f0d1a85819316`
- all-15 summary CSV SHA256:
  `d209cb0568e67f2b4ed0835ebed58609672c7fdff0ca029ff10add79e34ee854`
- all-15 artifact manifest SHA256:
  `1a81556c274cc076c86dd5be5d16c4d4fcb0f8328f8a4ebe6d41748f94757274`
- repository qualification manifest SHA256:
  computed after this document is generated and reported by the
  freeze command.
