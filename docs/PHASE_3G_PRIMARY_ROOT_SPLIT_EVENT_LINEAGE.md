# Phase 3G Primary Root, Split and Event-Lineage Audit

## Status

PASS

## Primary frozen baseline

`DATE2025_CNN_400MS_RECONSTRUCTED`

## Primary real-data root candidate

`/mnt/hdd16T/protechto/data/back/UniVrFall_KFall/segments/400ms_50ov_npseg_filt_binary`

Raw subject directories: 69

Real fold subjects: 67

Augmentation-only subjects: 999 and 1000

The 67-subject real population is an exact set match to the historical protected baseline population.

## Outer test lineage

Historical held-out test membership is an exact match to reconstructed five-fold fold 1.

Historical train plus validation membership is also exactly the fold-1 non-test population.

## Important inner-split distinction

Historical frozen checkpoint:

- train: 47 subjects
- validation: 6 subjects
- test: 14 subjects

Prospective five-fold fold 1:

- train: 42 subjects
- validation: 11 subjects
- test: 14 subjects

Therefore the historical baseline and prospective five-fold replication share the outer test partition but do not share the same inner train-validation split.

The historical split remains frozen checkpoint provenance.

The prospective five-fold protocol remains a separate fold-specific replication protocol.

## Curated risk/event assets

- Subject folds: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/SUBJECT_FOLDS_COMBINED_CERTAIN_V1.json`
- Subject folds: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/SUBJECT_FOLDS_COMBINED_CERTAIN_V1.json`
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/FALL_EVENT_INDEX_COMBINED_LABELED.csv` rows=2919
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/FALL_EVENT_INDEX_COMBINED_UNLABELED.csv` rows=2919
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/FALL_EVENT_INDEX_KFALL_UNLABELED.csv` rows=2346
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/HR_LR_Fallings/risk_data_generated/FALL_EVENT_INDEX_UNIVR_UNLABELED.csv` rows=573
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/FALL_EVENT_INDEX_COMBINED_LABELED.csv` rows=2919
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/FALL_EVENT_INDEX_COMBINED_UNLABELED.csv` rows=2919
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/FALL_EVENT_INDEX_KFALL_UNLABELED.csv` rows=2346
- Event index: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori/risk_data_generated/FALL_EVENT_INDEX_UNIVR_UNLABELED.csv` rows=573

## Remaining Phase-3 work

Reconcile UniVR event-count and onset/impact lineage using the curated event index plus official annotations.

Freeze event timing, window stride, calibration partition and five-fold replication contract.
