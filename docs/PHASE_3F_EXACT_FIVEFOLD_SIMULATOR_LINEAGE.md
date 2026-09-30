# Phase 3F Exact Five-Fold and Simulator Lineage

## Status

PASS

## KFold subject-order correction

`SORTED_LEXICOGRAPHIC_DIRECTORY_NAMES`

The loader explicitly applies `sorted(os.listdir(root_directory))`.

Therefore fold reconstruction is deterministic for a frozen dataset root.

## Primary model

`DATE2025_CNN_400MS_RECONSTRUCTED` remains the primary baseline.

## Existing complete 400-ms five-fold families

- CNNSplit
- LSTM
- ResNet

These are architecture comparators and are not treated as five-fold replicas of the primary DATE CNN.

## Candidate fold populations

- `/mnt/hdd16T/protechto/data/KFall_oriented/segments/400ms_50ov_npseg_filt_binary`: 32 fold subjects
- `/mnt/hdd16T/protechto/data/UniVrFall_oriented/segments/400ms_50ov_npseg_filt_binary`: 29 fold subjects
- `/mnt/hdd16T/protechto/data/UniVrFall_KFall_OF/segments/400ms_50ov_npseg_filt_binary`: 71 fold subjects
- `/mnt/hdd16T/protechto/data/back/UniVrFall_KFall/segments/400ms_50ov_npseg_filt_binary`: 67 fold subjects

## Simulator candidates

- `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings`: 79 files explicitly mention both UniVR and KFall
- `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/v7_activity_fall_200ms`: 1 files explicitly mention both UniVR and KFall
- `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/v7_activity_fall_200ms`: 1 files explicitly mention both UniVR and KFall
- `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_results/audit_kfall`: 0 files explicitly mention both UniVR and KFall
- `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_results/audit_kfall`: 0 files explicitly mention both UniVR and KFall

## Next decision

Freeze the exact combined real-data root and its five fold memberships once checkpoint/training lineage identifies the historical root.

Then reconcile onset/impact annotations to trial keys before physical lead-time semantics are frozen.
