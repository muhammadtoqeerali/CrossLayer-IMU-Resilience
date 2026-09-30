# Phase 1 Source Audit

Status: automated evidence collection complete. Scientific reuse decisions remain pending.

The audit is read-only with respect to inherited projects.

## Source projects

| Project | Exists | Git | Branch | HEAD | Relevant files |
|---|---:|---:|---|---|---:|
| `IMU_Reliability` | yes | yes | `main` | `e2c82586fec6` | 569 |
| `RC-RGD-IMU_publish` | yes | yes | `main` | `e1db7880e17a` | 483 |
| `Protechto-master` | yes | yes | `toqeer-dev` | `8a27377f5bb9` | 537 |
| `Protechto_master` | yes | yes | `main` | `ae861f9840bd` | 566 |
| `Protechto-repo` | yes | yes | `toqeer-dev` | `3691201a22ed` | 399 |
| `Protechto` | yes | yes | `toqeer-dev` | `643246032a68` | 486 |
| `fall_project_code` | yes | no | `-` | `-` | 1 |
| `HR_LR_Fallings` | yes | yes | `main` | `534722b560de` | 205 |
| `TRUST_ROBOT` | yes | yes | `trajectory-association-evaluation-protocol` | `68af6a3a49e8` | 728 |
| `TRUST_ROBOT_RUNTIME` | yes | no | `-` | `-` | 4 |

## Relevant-file category counts

| Project | baseline_model | data_pipeline | embedded_runtime | evaluation | fault_injection | integrity_runtime | quantization_export | training_pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMU_Reliability` | 142 | 51 | 35 | 364 | 52 | 252 | 16 | 34 |
| `RC-RGD-IMU_publish` | 124 | 50 | 33 | 305 | 37 | 209 | 16 | 31 |
| `Protechto-master` | 67 | 446 | 0 | 13 | 3 | 6 | 3 | 7 |
| `Protechto_master` | 77 | 446 | 0 | 24 | 3 | 8 | 9 | 15 |
| `Protechto-repo` | 40 | 352 | 0 | 4 | 0 | 0 | 3 | 7 |
| `Protechto` | 90 | 387 | 0 | 9 | 0 | 0 | 5 | 16 |
| `fall_project_code` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `HR_LR_Fallings` | 109 | 39 | 0 | 98 | 0 | 0 | 2 | 18 |
| `TRUST_ROBOT` | 131 | 110 | 45 | 389 | 143 | 532 | 8 | 58 |
| `TRUST_ROBOT_RUNTIME` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |

## Detected clean-baseline candidates

Automated scores are navigation aids only. They are not scientific model rankings.

| Family | Source | File | SHA-256 | Audit score | Signals |
|---|---|---|---|---:|---|
| `compact_1d_cnn` | `IMU_Reliability` | `models/architectures/baseline_cnn.py` | `259fa4134ee5` | 39 | conv1d, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `models/architectures/ds_cnn.py` | `2730580c212d` | 24 | conv1d, depthwise, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `configs/models/baseline_model_registry_v1.json` | `ba10a15427c1` | 20 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/finalize_historical_decision.py` | `6dfa4a394ba6` | 18 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/lock_and_evaluate_baseline.py` | `ca45785c67fd` | 18 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/verify_400ms_candidate.py` | `d47227e55355` | 18 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_baselines/train_1dcnn_baseline.py` | `4213ba2e4523` | 18 | conv1d, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/models/cnn1d.py` | `3ab41c355c8f` | 18 | conv1d, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/17_benchmark_analysis/scripts/create_baseline_report.py` | `c2226b2b0009` | 18 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `external/snap_uq/snapuq/models/audio/dscnn.py` | `97e79f1966e0` | 18 | depthwise, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `external/snap_uq/snapuq/models/factory.py` | `8b43f1c17f60` | 18 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `external/snap_uq/snapuq/models/vision/lenet.py` | `15ab2b6387bd` | 18 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `external/tcuq/tcuq/TCUQ notebooks and Codes/models/cnn4_mnist.py` | `788d59fdf919` | 18 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `external/tcuq/tcuq/TCUQ notebooks and Codes/models/dscnn_speech.py` | `6e68a6ecba26` | 18 | depthwise, pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `docs/protected_baseline_v1.md` | `81b7d6e5cca7` | 15 | conv1d, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/baseline_freeze_report.txt` | `2251f3bd747c` | 15 | conv1d, onnx, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/baseline_lock_and_clean_eval.txt` | `6dda6feb3d69` | 15 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/baseline_reconstruction_report.txt` | `a77bea25a868` | 15 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/final_historical_decision_report.txt` | `829a8ffd7139` | 15 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/checkpoint_catalog.json` | `952a87e352b2` | 15 | onnx, quantization |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/checkpoint_family_summary.txt` | `b28191c6f177` | 15 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/checkpoint_load_errors.json` | `29d0fed1988d` | 15 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/ori_copy_audit.txt` | `57c9edc3ab6b` | 15 | conv1d, lstm, onnx, pytorch, stm32, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/phase1e_continuation.txt` | `a29822770f74` | 15 | onnx |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/phase1e_final_lineage.txt` | `6014fdaa929a` | 15 | onnx |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/forensics/phase1f_400ms_baseline.txt` | `5fe967541ef4` | 15 | lstm, onnx, pytorch, quantization |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/00_baseline/historical_decision_rule.txt` | `150ee0ffafd9` | 15 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/configs/models/model_registry.yaml` | `b8e3f8905dce` | 15 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/01_integrity_p0/audit_full_signal_lineage.py` | `4ff0ed0525cf` | 13 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/01_integrity_p0/build_real_trial_inventory.py` | `613d550a5090` | 13 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_ood/build_ood_calibration_protocol_v1.py` | `2b70ca6cbc20` | 13 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_ood/build_ood_final_test_protocol_v1.py` | `074fe78197d2` | 13 | - |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_ood/evaluate_ood_calibration_v1.py` | `19dbd4d1d8b6` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_ood/evaluate_ood_calibration_v1b.py` | `6462829b7ffe` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/03_ood/evaluate_ood_final_test_v1.py` | `9805b5483e0f` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/04_runtime/benchmark_runtime_resource_overhead_v1.py` | `a140daa5d12f` | 13 | pytorch, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/04_runtime/build_runtime_integration_contract_v1.py` | `435caba613f1` | 13 | stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/04_runtime/build_runtime_resource_overhead_protocol_v1.py` | `6d157e7cd207` | 13 | stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/05_embedded/build_embedded_evidence_convergence_ledger_v1.py` | `777924bb7c3c` | 13 | cmsis, onnx, quantization, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/05_embedded/build_embedded_reference_path_contract_v1.py` | `bc5206aa957b` | 13 | cmsis, onnx, quantization, stm32, tflite |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/05_embedded/build_model_export_execution_environment_contract_v1.py` | `f6a560e6e6b2` | 13 | onnx, quantization, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/05_embedded/build_prospective_protected_model_export_protocol_v1.py` | `bf2960ee4943` | 13 | onnx, quantization, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/05_embedded/export_protected_model_onnx_v1.py` | `3319c9321c95` | 13 | onnx, pytorch, quantization, stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/06_paper/build_paper_evidence_claim_matrix_v1.py` | `127bebc01b90` | 13 | stm32 |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/10_model_comparison/evaluate_all_models.py` | `78c3a9fcd39c` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/11_efficiency_analysis/run_efficiency_analysis.py` | `bf0d366ddfe2` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/12_robustness_comparison/run_model_robustness_comparison.py` | `0b9b7d18fd8a` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/analyze_benchmark_v3r1.py` | `a6cf855916d9` | 13 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/analyze_benchmark_v3r1_r2.py` | `4b48d4981a03` | 13 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/analyze_v3r1_decision.py` | `fcce64ce0114` | 13 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/benchmark_efficiency_v3r1.py` | `6a6d121d6096` | 13 | lstm, pytorch, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/engine/input_adapter.py` | `e3de53c2258b` | 13 | lstm, pytorch, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/engine/model_forward.py` | `e69b34c79f65` | 13 | conv1d, lstm, pytorch, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/16_benchmark_suite/run_benchmark_v3r1.py` | `fba64468d553` | 13 | lstm, pytorch, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/17_v25_screening/analyze_v25_screening_r1.py` | `ee97829efb3d` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/19_v25_final_confirmation/analyze_v25_final_r2.py` | `1bf32225b92f` | 13 | lstm, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/19_v25_final_confirmation/run_v25_final_r1.py` | `d6bfb61148f7` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/19_v25_final_confirmation/run_v25_final_r2.py` | `ff6d70a2a2bf` | 13 | pytorch |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/20_v25_final_efficiency/analyze_v25_final_efficiency.py` | `aded870acdfc` | 13 | lstm, pytorch, transformer |
| `compact_1d_cnn` | `IMU_Reliability` | `experiments/20_v25_final_efficiency/analyze_v25_final_efficiency_optimized.py` | `31c665eef4b6` | 13 | lstm, pytorch, transformer |

## Cross-project duplicate source groups

Detected duplicate groups across distinct candidate projects: **924**.

### Duplicate group 1

`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

- `IMU_Reliability/experiments/16_benchmark_suite/engine/__init__.py`
- `IMU_Reliability/experiments/16_benchmark_suite/models/__init__.py`
- `IMU_Reliability/experiments/17_benchmark_analysis/engine/__init__.py`
- `IMU_Reliability/external/snap_uq/snapuq/models/audio/__init__.py`
- `IMU_Reliability/external/snap_uq/snapuq/models/vision/__init__.py`
- `IMU_Reliability/external/tcuq/tcuq/TCUQ notebooks and Codes/baselines/__init__.py`
- `IMU_Reliability/external/trusttiny_har/conformal/__init__.py`
- `IMU_Reliability/external/trusttiny_har/evaluation/__init__.py`
- `IMU_Reliability/external/trusttiny_har/models/__init__.py`
- `IMU_Reliability/external/trusttiny_har/privacy/__init__.py`
- `IMU_Reliability/external/trusttiny_har/scripts/__init__.py`
- `IMU_Reliability/external/trusttiny_har/training/__init__.py`
- `IMU_Reliability/external/trusttiny_har/utils/__init__.py`
- `IMU_Reliability/paper_final_handoff/03_EXTERNAL_STORM_RESULTS/external_test_binding/aggregate_model_candidates.txt`
- `IMU_Reliability/paper_final_handoff/03_EXTERNAL_STORM_RESULTS/external_test_binding/condition_manifest_candidates.txt`
- `IMU_Reliability/src/imu_reliability/__init__.py`
- `IMU_Reliability/src/imu_reliability/baseline/__init__.py`
- `RC-RGD-IMU_publish/experiments/16_benchmark_suite/engine/__init__.py`
- `RC-RGD-IMU_publish/experiments/16_benchmark_suite/models/__init__.py`
- `RC-RGD-IMU_publish/experiments/17_benchmark_analysis/engine/__init__.py`
- `RC-RGD-IMU_publish/external/snap_uq/snapuq/models/audio/__init__.py`
- `RC-RGD-IMU_publish/external/snap_uq/snapuq/models/vision/__init__.py`
- `RC-RGD-IMU_publish/external/tcuq/tcuq/TCUQ notebooks and Codes/baselines/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/conformal/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/evaluation/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/models/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/privacy/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/scripts/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/training/__init__.py`
- `RC-RGD-IMU_publish/external/trusttiny_har/utils/__init__.py`
- `RC-RGD-IMU_publish/src/imu_reliability/__init__.py`
- `RC-RGD-IMU_publish/src/imu_reliability/baseline/__init__.py`
- `Protechto-master/converter/__init__.py`
- `Protechto-master/losses/__init__.py`
- `Protechto-master/models/modules/__init__.py`
- `Protechto-master/models/tsai_modules/__init__.py`
- `Protechto-master/models/tsai_modules/layers/__init__.py`
- `Protechto-master/preprocessing/__init__.py`
- `Protechto-master/simulation/__init__.py`
- `Protechto_master/converter/__init__.py`
- `Protechto_master/losses/__init__.py`
- `Protechto_master/models/modules/__init__.py`
- `Protechto_master/models/tsai_modules/__init__.py`
- `Protechto_master/models/tsai_modules/layers/__init__.py`
- `Protechto_master/preprocessing/__init__.py`
- `Protechto_master/simulation/__init__.py`
- `Protechto-repo/converter/__init__.py`
- `Protechto-repo/losses/__init__.py`
- `Protechto-repo/models/modules/__init__.py`
- `Protechto-repo/models/tsai_modules/__init__.py`
- `Protechto-repo/models/tsai_modules/layers/__init__.py`
- `Protechto-repo/preprocessing/__init__.py`
- `Protechto-repo/simulation/__init__.py`
- `Protechto/converter/__init__.py`
- `Protechto/losses/__init__.py`
- `Protechto/models/modules/__init__.py`
- `Protechto/models/tsai_modules/__init__.py`
- `Protechto/models/tsai_modules/layers/__init__.py`
- `Protechto/preprocessing/__init__.py`
- `Protechto/Protechto/converter/__init__.py`
- `Protechto/Protechto/losses/__init__.py`
- `Protechto/Protechto/models/modules/__init__.py`
- `Protechto/Protechto/models/tsai_modules/__init__.py`
- `Protechto/Protechto/models/tsai_modules/layers/__init__.py`
- `Protechto/Protechto/preprocessing/__init__.py`
- `Protechto/Protechto/simulation/__init__.py`
- `Protechto/simulation/__init__.py`
- `HR_LR_Fallings/converter/__init__.py`
- `HR_LR_Fallings/losses/__init__.py`
- `HR_LR_Fallings/models/modules/__init__.py`
- `HR_LR_Fallings/models/tsai_modules/__init__.py`
- `HR_LR_Fallings/models/tsai_modules/layers/__init__.py`
- `HR_LR_Fallings/preprocessing/__init__.py`
- `HR_LR_Fallings/risk_analysis/__init__.py`
- `HR_LR_Fallings/risk_evaluation/__init__.py`
- `HR_LR_Fallings/risk_models/__init__.py`
- `HR_LR_Fallings/simulation/__init__.py`
- `TRUST_ROBOT/experiments/16_benchmark_suite/engine/__init__.py`
- `TRUST_ROBOT/experiments/16_benchmark_suite/models/__init__.py`
- `TRUST_ROBOT/experiments/17_benchmark_analysis/engine/__init__.py`
- `TRUST_ROBOT/src/imu_reliability/__init__.py`
- `TRUST_ROBOT/src/imu_reliability/baseline/__init__.py`

### Duplicate group 2

`059d37fc6a999a943b877538b68061213d8e8bf4b2f44528c7bca0d96d766f44`

- `Protechto-master/models/configs/CNNSplit_config.py`
- `Protechto_master/models/configs/CNNSplit_config.py`
- `Protechto-repo/models/configs/CNNSplit_config.py`
- `Protechto/models/configs/CNNSplit_config.py`
- `Protechto/Protechto/models/configs/CNNSplit_config.py`
- `HR_LR_Fallings/models/configs/CNNSplit_config.py`

### Duplicate group 3

`08e92e4a985c7d183b94513fd5bd36c33e8ae64ee1a22fdb5d9465932344f7a3`

- `Protechto-master/dataloaders/OnFieldDataloader.py`
- `Protechto_master/dataloaders/OnFieldDataloader.py`
- `Protechto-repo/dataloaders/OnFieldDataloader.py`
- `Protechto/dataloaders/OnFieldDataloader.py`
- `Protechto/Protechto/dataloaders/OnFieldDataloader.py`
- `HR_LR_Fallings/dataloaders/OnFieldDataloader.py`

### Duplicate group 4

`17d2e3ec51b182b2e5c25d755a15a9a9194bde42d6b84ebd394c6763c3a998ad`

- `Protechto-master/models/tsai_modules/layers/SameConv1d.py`
- `Protechto_master/models/tsai_modules/layers/SameConv1d.py`
- `Protechto-repo/models/tsai_modules/layers/SameConv1d.py`
- `Protechto/models/tsai_modules/layers/SameConv1d.py`
- `Protechto/Protechto/models/tsai_modules/layers/SameConv1d.py`
- `HR_LR_Fallings/models/tsai_modules/layers/SameConv1d.py`

### Duplicate group 5

`2c5279bfe0161ffb884ecb0f68bfab536c7da4df6dab3b079e8494f511c4f473`

- `Protechto-master/dataloaders/__init__.py`
- `Protechto_master/dataloaders/__init__.py`
- `Protechto-repo/dataloaders/__init__.py`
- `Protechto/dataloaders/__init__.py`
- `Protechto/Protechto/dataloaders/__init__.py`
- `HR_LR_Fallings/dataloaders/__init__.py`

### Duplicate group 6

`30b8f039315dfef2d656fca05415961f748888ee3c3b9f25885a766c3c10f91c`

- `Protechto-master/losses/CostSensitiveLoss.py`
- `Protechto_master/losses/CostSensitiveLoss.py`
- `Protechto-repo/losses/CostSensitiveLoss.py`
- `Protechto/losses/CostSensitiveLoss.py`
- `Protechto/Protechto/losses/CostSensitiveLoss.py`
- `HR_LR_Fallings/losses/CostSensitiveLoss.py`

### Duplicate group 7

`31063f318fbd9fba79e1b05e8fde102709ade9423f1e72c22976ac5774c19456`

- `Protechto-master/models/tsai_modules/layers/Add.py`
- `Protechto_master/models/tsai_modules/layers/Add.py`
- `Protechto-repo/models/tsai_modules/layers/Add.py`
- `Protechto/models/tsai_modules/layers/Add.py`
- `Protechto/Protechto/models/tsai_modules/layers/Add.py`
- `HR_LR_Fallings/models/tsai_modules/layers/Add.py`

### Duplicate group 8

`333ef5526c105a1701b19c00961f3e5ebcacfb3af33ce0e29d32713d55f22603`

- `Protechto-master/models/tsai_modules/layers/Squeeze.py`
- `Protechto_master/models/tsai_modules/layers/Squeeze.py`
- `Protechto-repo/models/tsai_modules/layers/Squeeze.py`
- `Protechto/models/tsai_modules/layers/Squeeze.py`
- `Protechto/Protechto/models/tsai_modules/layers/Squeeze.py`
- `HR_LR_Fallings/models/tsai_modules/layers/Squeeze.py`

### Duplicate group 9

`3c5987efa06b63a34dd89276f467f8327ced12110ce3fa2eb3f6d39e7066cd5a`

- `Protechto-master/preprocessing/time_warping.py`
- `Protechto_master/preprocessing/time_warping.py`
- `Protechto-repo/preprocessing/time_warping.py`
- `Protechto/preprocessing/time_warping.py`
- `Protechto/Protechto/preprocessing/time_warping.py`
- `HR_LR_Fallings/preprocessing/time_warping.py`

### Duplicate group 10

`46dd3489806fd56be3f11f237d3ac0826c8a5dbb49d85e91b13e7caf075dd129`

- `Protechto-master/preprocessing/orientation_changing.py`
- `Protechto_master/preprocessing/orientation_changing.py`
- `Protechto-repo/preprocessing/orientation_changing.py`
- `Protechto/preprocessing/orientation_changing.py`
- `Protechto/Protechto/preprocessing/orientation_changing.py`
- `HR_LR_Fallings/preprocessing/orientation_changing.py`

### Duplicate group 11

`496a9e26e36cc22695e0baf4a84e7a87be9c305145308c574d5922033e31131e`

- `Protechto-master/losses/FocalLossV2.py`
- `Protechto_master/losses/FocalLossV2.py`
- `Protechto-repo/losses/FocalLossV2.py`
- `Protechto/losses/FocalLossV2.py`
- `Protechto/Protechto/losses/FocalLossV2.py`
- `HR_LR_Fallings/losses/FocalLossV2.py`

### Duplicate group 12

`4bf169f97fd7846e952e374f2ea6beb73311ec38dbf1b4c618257c0605a58c08`

- `Protechto-master/losses/FocalLoss.py`
- `Protechto_master/losses/FocalLoss.py`
- `Protechto-repo/losses/FocalLoss.py`
- `Protechto/losses/FocalLoss.py`
- `Protechto/Protechto/losses/FocalLoss.py`
- `HR_LR_Fallings/losses/FocalLoss.py`

### Duplicate group 13

`55961e6b0dd331d6e78469d1eddf8ee87ad389309b6036cbfa17874399fa66a8`

- `Protechto-master/models/tsai_modules/layers/SeparableConv1d.py`
- `Protechto_master/models/tsai_modules/layers/SeparableConv1d.py`
- `Protechto-repo/models/tsai_modules/layers/SeparableConv1d.py`
- `Protechto/models/tsai_modules/layers/SeparableConv1d.py`
- `Protechto/Protechto/models/tsai_modules/layers/SeparableConv1d.py`
- `HR_LR_Fallings/models/tsai_modules/layers/SeparableConv1d.py`

### Duplicate group 14

`5d0b54859d3bf118cb27b855aff2fff132fb16a74243a58ab332083dc072e962`

- `Protechto-master/models/modules/Residual.py`
- `Protechto_master/models/modules/Residual.py`
- `Protechto-repo/models/modules/Residual.py`
- `Protechto/models/modules/Residual.py`
- `Protechto/Protechto/models/modules/Residual.py`
- `HR_LR_Fallings/models/modules/Residual.py`

### Duplicate group 15

`5d629b93e71bb542ed4148f5c714558987340bc226564339fb9002c14582de28`

- `Protechto-master/test_ensemble.py`
- `Protechto_master/test_ensemble.py`
- `Protechto-repo/test_ensemble.py`
- `Protechto/Protechto/test_ensemble.py`
- `Protechto/test_ensemble.py`
- `HR_LR_Fallings/test_ensemble.py`

### Duplicate group 16

`5e1ddac378c80929068b6fb8f62fff0d351bacbf047297829bd0531ad9c43eb2`

- `Protechto-master/models/CNN.py`
- `Protechto_master/models/CNN.py`
- `Protechto-repo/models/CNN.py`
- `Protechto/models/CNN.py`
- `Protechto/Protechto/models/CNN.py`
- `HR_LR_Fallings/models/CNN.py`

### Duplicate group 17

`62ba0ae6bb6fb8d7fede1a9e6569702ad04d4c1497423eec3d58035e2080e1ff`

- `Protechto-master/models/configs/LSTM_config.py`
- `Protechto_master/models/configs/LSTM_config.py`
- `Protechto-repo/models/configs/LSTM_config.py`
- `Protechto/models/configs/LSTM_config.py`
- `Protechto/Protechto/models/configs/LSTM_config.py`
- `HR_LR_Fallings/models/configs/LSTM_config.py`

### Duplicate group 18

`64194dbfc54e11e1dca47fd830f49fd54d741e8fa66867c3056208288e41e6d6`

- `Protechto-master/models/ResNet.py`
- `Protechto_master/models/ResNet.py`
- `Protechto-repo/models/ResNet.py`
- `Protechto/models/ResNet.py`
- `Protechto/Protechto/models/ResNet.py`
- `HR_LR_Fallings/models/ResNet.py`

### Duplicate group 19

`7284f157070b6a71dc8177337e4c537eb22f754d8da341ab377bd337e64ffe86`

- `Protechto-master/models/tsai_modules/layers/ConvBlock.py`
- `Protechto_master/models/tsai_modules/layers/ConvBlock.py`
- `Protechto-repo/models/tsai_modules/layers/ConvBlock.py`
- `Protechto/models/tsai_modules/layers/ConvBlock.py`
- `Protechto/Protechto/models/tsai_modules/layers/ConvBlock.py`
- `HR_LR_Fallings/models/tsai_modules/layers/ConvBlock.py`

### Duplicate group 20

`76ca555f5a94daa501d23f84664dea687008a28384f594a18750dc57b74d37a5`

- `Protechto-master/models/LSTM.py`
- `Protechto_master/models/LSTM.py`
- `Protechto-repo/models/LSTM.py`
- `Protechto/models/LSTM.py`
- `Protechto/Protechto/models/LSTM.py`
- `HR_LR_Fallings/models/LSTM.py`

### Duplicate group 21

`79a6eb3f28e8dca0cc59759155c0f7fd58c43251761f123524712b66566e7904`

- `Protechto-master/models/CNNSplit.py`
- `Protechto_master/models/CNNSplit.py`
- `Protechto-repo/models/CNNSplit.py`
- `Protechto/models/CNNSplit.py`
- `Protechto/Protechto/models/CNNSplit.py`
- `HR_LR_Fallings/models/CNNSplit.py`

### Duplicate group 22

`887f09dd3132460edd489df03d5ab68cf4ca83475de40385485f58b2a735a71e`

- `Protechto-master/models/modules/Inception.py`
- `Protechto_master/models/modules/Inception.py`
- `Protechto-repo/models/modules/Inception.py`
- `Protechto/models/modules/Inception.py`
- `Protechto/Protechto/models/modules/Inception.py`
- `HR_LR_Fallings/models/modules/Inception.py`

### Duplicate group 23

`9b1d7bcd701a29b559bd87dc1a6dfc1a2ce66804d1b00ba9211290cd2615c242`

- `Protechto-master/simulation/SimulationDataloader.py`
- `Protechto_master/simulation/SimulationDataloader.py`
- `Protechto-repo/simulation/SimulationDataloader.py`
- `Protechto/Protechto/simulation/SimulationDataloader.py`
- `Protechto/simulation/SimulationDataloader.py`
- `HR_LR_Fallings/simulation/SimulationDataloader.py`

### Duplicate group 24

`9fc5c83d941baf3a7a8cd2d2f928afb73a922e956eeb6b6e406c56be0b4771c4`

- `Protechto-master/models/tsai_modules/layers/Conv1d.py`
- `Protechto_master/models/tsai_modules/layers/Conv1d.py`
- `Protechto-repo/models/tsai_modules/layers/Conv1d.py`
- `Protechto/models/tsai_modules/layers/Conv1d.py`
- `Protechto/Protechto/models/tsai_modules/layers/Conv1d.py`
- `HR_LR_Fallings/models/tsai_modules/layers/Conv1d.py`

### Duplicate group 25

`a31b679c11152fd80181c5bfe12b010ba4ae4ea655dca5f8c8fa563082afb607`

- `Protechto-master/models/tsai_modules/utilities.py`
- `Protechto_master/models/tsai_modules/utilities.py`
- `Protechto-repo/models/tsai_modules/utilities.py`
- `Protechto/models/tsai_modules/utilities.py`
- `Protechto/Protechto/models/tsai_modules/utilities.py`
- `HR_LR_Fallings/models/tsai_modules/utilities.py`

### Duplicate group 26

`a33dbe6e1310b851b50018653b75b437f712a91ea9f78c3650edcc708fa2b321`

- `Protechto-master/requirements.txt`
- `Protechto_master/requirements.txt`
- `Protechto-repo/requirements.txt`
- `Protechto/Protechto/requirements.txt`
- `Protechto/requirements.txt`
- `HR_LR_Fallings/requirements.txt`

### Duplicate group 27

`a3425d1d163a50c0b9963ee7fb6ee480e3d0b7202de7654d40bf770de7af6f8e`

- `Protechto-master/to_onnx.py`
- `Protechto_master/to_onnx.py`
- `Protechto-repo/to_onnx.py`
- `Protechto/Protechto/to_onnx.py`
- `Protechto/to_onnx.py`
- `HR_LR_Fallings/to_onnx.py`

### Duplicate group 28

`a5b40d8bc4889eaa222750c2cd8892da2de48f52ed7efb4aced0b06c7152d92a`

- `Protechto-master/models/configs/ResNet_config.py`
- `Protechto_master/models/configs/ResNet_config.py`
- `Protechto-repo/models/configs/ResNet_config.py`
- `Protechto/models/configs/ResNet_config.py`
- `Protechto/Protechto/models/configs/ResNet_config.py`
- `HR_LR_Fallings/models/configs/ResNet_config.py`

### Duplicate group 29

`a9c3c106e82376ff8bab0999b659b08dec3e0af7f1a34677f7e451134e95b742`

- `Protechto-master/models/ResCNN.py`
- `Protechto_master/models/ResCNN.py`
- `Protechto-repo/models/ResCNN.py`
- `Protechto/models/ResCNN.py`
- `Protechto/Protechto/models/ResCNN.py`
- `HR_LR_Fallings/models/ResCNN.py`

### Duplicate group 30

`b495165985efdbbe57323c3de2efb0b8004d94384e7f61d79f38af8b1c15aafd`

- `Protechto-master/models/helper.py`
- `Protechto_master/models/helper.py`
- `Protechto-repo/models/helper.py`
- `Protechto/models/helper.py`
- `Protechto/Protechto/models/helper.py`
- `HR_LR_Fallings/models/helper.py`

## Phase-1 decision state

No component is yet classified as REUSE, ADAPT, REFERENCE_ONLY or REJECT.

The next step is expert review of the detected candidates and exact source-file inspection.
