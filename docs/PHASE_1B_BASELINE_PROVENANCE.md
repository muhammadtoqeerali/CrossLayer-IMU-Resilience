# Phase 1B Baseline Provenance Audit

Status: evidence collected. Baseline remains unfrozen.

## Source repositories

### development

- Path: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/IMU_Reliability`
- Branch: `main`
- HEAD: `e2c82586fec6a04c55c059732ca0184b36f13b00`
- Origin: `git@github.com:muhammadtoqeerali/RC-RGD-IMU.git`
- Dirty: `True`

### published_clean

- Path: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/RC-RGD-IMU_publish`
- Branch: `main`
- HEAD: `e1db7880e17a5632bc4ce238125923f986e85519`
- Origin: `git@github.com:muhammadtoqeerali/RC-RGD-IMU.git`
- Dirty: `False`

## Exact candidate-file comparison

| File | Development exists | Clean exists | Dev = clean | Dev modified vs own HEAD |
|---|---:|---:|---:|---:|
| `src/imu_reliability/baseline/date2025_cnn400.py` | True | True | True | False |
| `src/imu_reliability/baseline/historical_decision.py` | True | True | True | False |
| `docs/protected_baseline_v1.md` | True | True | True | False |
| `experiments/00_baseline/finalize_historical_decision.py` | True | True | True | False |
| `experiments/00_baseline/fingerprint_400ms_datasets.py` | True | True | True | False |
| `experiments/00_baseline/lock_and_evaluate_baseline.py` | True | True | True | False |
| `experiments/00_baseline/verify_400ms_candidate.py` | True | True | True | False |
| `models/architectures/baseline_cnn.py` | True | True | True | - |
| `models/architectures/ds_cnn.py` | True | True | True | - |
| `models/architectures/tcn.py` | True | True | True | - |
| `experiments/03_baselines/train_1dcnn_baseline.py` | True | True | True | - |
| `configs/project.yaml` | True | True | True | False |

## DATE-2025 checkpoint verification

Expected SHA-256: `ee7c0079bfb8555bff45c3077cc24eaa4373c57729045d92a831a1d7a3ea9bb1`

No local artifact with the expected checkpoint hash was found by this scan.

Checkpoint absence would not invalidate the architecture as a research candidate, but it would change the reproducibility path and must be handled explicitly.

## Quantization/export candidates

### Development tree

- `configs/embedded/prospective_model_export_execution_environment_contract_v1.json` `3f9f5f80886e`
- `configs/embedded/prospective_protected_model_export_protocol_v1.json` `5574278d4258`
- `data/manifests/prospective_model_export_dependency_blocker_v1.json` `33c07b0f2a5b`
- `experiments/05_embedded/build_model_export_execution_environment_contract_v1.py` `f6a560e6e6b2`
- `experiments/05_embedded/build_prospective_protected_model_export_protocol_v1.py` `bf2960ee4943`
- `experiments/05_embedded/export_protected_model_onnx_v1.py` `3319c9321c95`
- `tests/test_prospective_model_export_execution_environment_contract_v1.py` `7238b34fd3a9`
- `tests/test_prospective_protected_model_export_protocol_v1.py` `54fd1244eec5`
- `tests/test_protected_model_onnx_export_v1.py` `a5be24cac8f3`

### Clean published tree

- `configs/embedded/prospective_model_export_execution_environment_contract_v1.json` `3f9f5f80886e`
- `configs/embedded/prospective_protected_model_export_protocol_v1.json` `5574278d4258`
- `experiments/05_embedded/build_model_export_execution_environment_contract_v1.py` `f6a560e6e6b2`
- `experiments/05_embedded/build_prospective_protected_model_export_protocol_v1.py` `bf2960ee4943`
- `experiments/05_embedded/export_protected_model_onnx_v1.py` `3319c9321c95`
- `tests/test_prospective_model_export_execution_environment_contract_v1.py` `7238b34fd3a9`
- `tests/test_prospective_protected_model_export_protocol_v1.py` `54fd1244eec5`
- `tests/test_protected_model_onnx_export_v1.py` `a5be24cac8f3`

## Embedded/runtime candidates

### Development tree

- `configs/embedded/embedded_evidence_convergence_ledger_v1.json` `d58f375362fe`
- `configs/embedded/embedded_reference_path_contract_v1.json` `175059822fa1`
- `configs/embedded/portable_c_runtime_semantic_contract_v1.json` `bfe2d56e9981`
- `configs/embedded/prospective_embedded_build_toolchain_blocker_v1.json` `b8ee7a7e1d56`
- `configs/embedded/prospective_model_export_execution_environment_contract_v1.json` `3f9f5f80886e`
- `configs/embedded/prospective_protected_model_export_protocol_v1.json` `5574278d4258`
- `configs/runtime/runtime_integration_contract_v1.json` `332beb6b99c7`
- `configs/runtime/runtime_resource_overhead_protocol_v1.json` `2063c44ba810`
- `data/manifests/embedded_build_toolchain_archaeology_receipt_v1.json` `3b180cfc64f0`
- `data/manifests/embedded_target_archaeology_receipt_v1.json` `14b10131c11f`
- `data/manifests/runtime_resource_overhead_result_receipt_v1.json` `17df6b6abcb4`
- `embedded/reference_runtime/include/imu_reliability_runtime.h` `016255607633`
- `embedded/reference_runtime/src/imu_reliability_runtime.c` `1b7e7fd97f99`
- `embedded/reference_runtime/tests/test_imu_reliability_runtime.c` `e780774b7778`
- `experiments/04_runtime/benchmark_runtime_resource_overhead_v1.py` `a140daa5d12f`
- `experiments/04_runtime/build_runtime_integration_contract_v1.py` `435caba613f1`
- `experiments/04_runtime/build_runtime_resource_overhead_protocol_v1.py` `6d157e7cd207`
- `experiments/05_embedded/build_embedded_build_toolchain_blocker_v1.py` `7fd46043975b`
- `experiments/05_embedded/build_embedded_evidence_convergence_ledger_v1.py` `777924bb7c3c`
- `experiments/05_embedded/build_embedded_reference_path_contract_v1.py` `bc5206aa957b`
- `experiments/05_embedded/build_model_export_execution_environment_contract_v1.py` `f6a560e6e6b2`
- `experiments/05_embedded/build_portable_c_runtime_semantic_contract_v1.py` `729bf8279258`
- `experiments/05_embedded/build_prospective_protected_model_export_protocol_v1.py` `bf2960ee4943`
- `experiments/05_embedded/export_protected_model_onnx_v1.py` `3319c9321c95`
- `paper_final_handoff/04_EFFICIENCY_AND_DEPLOYMENT/efficiency_measurement/runtime_environment_r8c.json` `a5d12cdba63c`
- `paper_final_handoff/04_EFFICIENCY_AND_DEPLOYMENT/efficiency_protocol/runtime_environment_r8a.json` `3c3f9c1e9089`
- `results/paper_implementation_archive_final_r11c/implementation_r11a/frozen_results/v26_efficiency_measurement_r8c/runtime_environment_r8c.json` `a5d12cdba63c`
- `results/paper_implementation_archive_final_r11c/implementation_r11a/frozen_results/v26_efficiency_protocol_r8a/runtime_environment_r8a.json` `3c3f9c1e9089`
- `results/paper_implementation_archive_r11a/frozen_results/v26_efficiency_measurement_r8c/runtime_environment_r8c.json` `a5d12cdba63c`
- `results/paper_implementation_archive_r11a/frozen_results/v26_efficiency_protocol_r8a/runtime_environment_r8a.json` `3c3f9c1e9089`
- `results/raw/runtime_resource_overhead_v1_candidate.json` `1fe4ed729b4a`
- `results/stage25_physical_fault_protocol/clean_reproduction_r4b2/canonical_model_forward_runtime_evidence.json` `6db28774cd21`
- `results/v26_efficiency_measurement_r8c/runtime_environment_r8c.json` `a5d12cdba63c`
- `results/v26_efficiency_protocol_r8a/runtime_environment_r8a.json` `3c3f9c1e9089`
- `src/imu_reliability/runtime/__init__.py` `f8b125ffb9e1`
- `src/imu_reliability/runtime/decision.py` `00391737a5f2`
- `src/imu_reliability/runtime/reliability.py` `aa8fe189301c`
- `tests/test_embedded_evidence_convergence_ledger_v1.py` `05b146da2378`
- `tests/test_embedded_reference_path_contract_v1.py` `82606b37f04f`
- `tests/test_portable_c_runtime_parity_v1.py` `2b4b9dbbe3d1`
- `tests/test_portable_c_runtime_semantic_contract_v1.py` `2ecac772bd38`
- `tests/test_prospective_embedded_build_toolchain_blocker_v1.py` `ca30dcfa5294`
- `tests/test_runtime_integration_contract_v1.py` `d5682430cb92`
- `tests/test_runtime_reliability.py` `794695c1b387`
- `tests/test_runtime_resource_overhead_benchmark_v1.py` `5c6648fb498a`
- `tests/test_runtime_resource_overhead_protocol_v1.py` `ef9b286261c9`

### Clean published tree

- `configs/embedded/embedded_evidence_convergence_ledger_v1.json` `d58f375362fe`
- `configs/embedded/embedded_reference_path_contract_v1.json` `175059822fa1`
- `configs/embedded/portable_c_runtime_semantic_contract_v1.json` `bfe2d56e9981`
- `configs/embedded/prospective_embedded_build_toolchain_blocker_v1.json` `b8ee7a7e1d56`
- `configs/embedded/prospective_model_export_execution_environment_contract_v1.json` `3f9f5f80886e`
- `configs/embedded/prospective_protected_model_export_protocol_v1.json` `5574278d4258`
- `configs/runtime/runtime_integration_contract_v1.json` `332beb6b99c7`
- `configs/runtime/runtime_resource_overhead_protocol_v1.json` `2063c44ba810`
- `embedded/reference_runtime/include/imu_reliability_runtime.h` `016255607633`
- `embedded/reference_runtime/src/imu_reliability_runtime.c` `1b7e7fd97f99`
- `embedded/reference_runtime/tests/test_imu_reliability_runtime.c` `e780774b7778`
- `experiments/04_runtime/benchmark_runtime_resource_overhead_v1.py` `a140daa5d12f`
- `experiments/04_runtime/build_runtime_integration_contract_v1.py` `435caba613f1`
- `experiments/04_runtime/build_runtime_resource_overhead_protocol_v1.py` `6d157e7cd207`
- `experiments/05_embedded/build_embedded_build_toolchain_blocker_v1.py` `7fd46043975b`
- `experiments/05_embedded/build_embedded_evidence_convergence_ledger_v1.py` `777924bb7c3c`
- `experiments/05_embedded/build_embedded_reference_path_contract_v1.py` `bc5206aa957b`
- `experiments/05_embedded/build_model_export_execution_environment_contract_v1.py` `f6a560e6e6b2`
- `experiments/05_embedded/build_portable_c_runtime_semantic_contract_v1.py` `729bf8279258`
- `experiments/05_embedded/build_prospective_protected_model_export_protocol_v1.py` `bf2960ee4943`
- `experiments/05_embedded/export_protected_model_onnx_v1.py` `3319c9321c95`
- `src/imu_reliability/runtime/__init__.py` `f8b125ffb9e1`
- `src/imu_reliability/runtime/decision.py` `00391737a5f2`
- `src/imu_reliability/runtime/reliability.py` `aa8fe189301c`
- `tests/test_embedded_evidence_convergence_ledger_v1.py` `05b146da2378`
- `tests/test_embedded_reference_path_contract_v1.py` `82606b37f04f`
- `tests/test_portable_c_runtime_parity_v1.py` `2b4b9dbbe3d1`
- `tests/test_portable_c_runtime_semantic_contract_v1.py` `2ecac772bd38`
- `tests/test_prospective_embedded_build_toolchain_blocker_v1.py` `ca30dcfa5294`
- `tests/test_runtime_integration_contract_v1.py` `d5682430cb92`
- `tests/test_runtime_reliability.py` `794695c1b387`
- `tests/test_runtime_resource_overhead_benchmark_v1.py` `5c6648fb498a`
- `tests/test_runtime_resource_overhead_protocol_v1.py` `ef9b286261c9`

## Current baseline decision

`DATE2025_CNN_400MS_RECONSTRUCTED` is the current primary candidate because it directly matches the pre-impact fall-detection task.

It is **not frozen**.

The generic compact 1D CNN, DS-CNN and TCN remain comparison/generalization candidates.
